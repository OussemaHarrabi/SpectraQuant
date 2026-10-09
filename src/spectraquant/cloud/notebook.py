"""Deterministic notebook generation from a :class:`RunSpec` (``design-cloud-adapter.md`` §3).

A generated notebook is *plumbing only*: it records the environment, materialises the pinned
environment, checks out the exact commit (with an assert), verifies dataset checksums, invokes the
repository's own runner, writes a schema-valid ``run_manifest.json``, exports the bundle with a
single machine-readable result line, and prints a teardown marker. No scientific logic lives in a
cell (``AGENTS.md`` §2b rule 1).

Determinism
-----------
``build_notebook`` is a pure function of ``(spec, template version)``: cell sources are static
text parameterised by the spec, cell ids are derived from their position and stage, and no
timestamp is baked in (the environment cell computes the start time at execution).

The digest self-reference problem
---------------------------------
A notebook cannot contain the sha256 of its own bytes. :func:`notebook_digest` therefore returns the
digest of the *canonical serialization*, in which the ``NOTEBOOK_DIGEST`` literal is the fixed
placeholder ``<PENDING>``. :func:`build_notebook` substitutes the real digest into that literal, so
the manifest records a digest a reviewer can recompute from the repository alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spectraquant.cloud.secrets import assert_no_secret, redact

if TYPE_CHECKING:  # pragma: no cover - typing only; the import stays lazy at runtime
    import nbformat

__all__ = [
    "NOTEBOOK_TEMPLATE_VERSION",
    "PENDING_DIGEST",
    "STAGES",
    "build_notebook",
    "cell_stages",
    "notebook_digest",
    "notebook_text",
    "require_nbformat",
    "write_notebook",
]

#: Version of the cell template. A change here changes every generated notebook's digest.
NOTEBOOK_TEMPLATE_VERSION = "1.4.0"

#: Placeholder used for the notebook's own digest in the canonical serialization.
PENDING_DIGEST = "<PENDING>"

#: Mandatory code cells, in the order ``design-cloud-adapter.md`` §3 fixes them.
STAGES: tuple[str, ...] = (
    "environment",
    "install",
    "repo",
    "data",
    "run",
    "manifest",
    "export",
    "teardown",
)

_METADATA_KEY = "spectraquant"


def require_nbformat() -> Any:
    """Import ``nbformat`` lazily and return the module.

    Returns:
        The imported ``nbformat`` module.

    Raises:
        ImportError: ``nbformat`` is not installed; the message names the optional ``cloud`` extra.
    """
    try:
        import nbformat
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatched import
        raise ImportError(
            "generating notebooks requires the optional 'cloud' extra: install it with "
            "`uv sync --all-extras` or `uv pip install 'spectraquant[cloud]'` (nbformat>=5.10)"
        ) from exc
    return nbformat


# --------------------------------------------------------------------------------------
# Cell sources
# --------------------------------------------------------------------------------------
_HEADER = """\
# @@RUN_ID@@ — generated notebook (template v@@TEMPLATE_VERSION@@); DO NOT EDIT.
#
# Produced by `spectraquant cloud notebook` from a frozen RunSpec. Hand edits are overwritten on the
# next regeneration and are not accepted as evidence (AGENTS.md §2b rule 2).
#
# platform: @@PLATFORM@@ | gpu_required: @@GPU_REQUIRED@@ | timeout: @@TIMEOUT_MINUTES@@ min
# experiment: @@EXPERIMENT_CONFIG@@ @@OVERRIDES@@
# runner: @@RUNNER_COMMAND@@
# git_commit: @@GIT_COMMIT@@@@DIRTY_NOTE@@
# measurement_class_expected: @@MEASUREMENT_CLASS@@
# spec_sha256: @@SPEC_SHA256@@
# notebook_digest: @@NOTEBOOK_DIGEST@@
#
# Cells, in order: environment, install, repo, data, run, manifest, export, teardown.
"""

_ENVIRONMENT = """\
# Mandatory cell 1/8 — environment record.
# Nothing scientific happens here: hardware, dependency versions, start time and the notebook's own
# identity are recorded so the manifest can prove where the numbers came from.
import hashlib, json, os, platform as _platform, shlex, shutil, subprocess, sys, time
from pathlib import Path

SPEC = json.loads(@@SPEC_LITERAL@@)
RUN_ID = SPEC["run_id"]
SPEC_SHA256 = @@SPEC_SHA256_LITERAL@@
NOTEBOOK_DIGEST = @@NOTEBOOK_DIGEST_LITERAL@@
NOTEBOOK_TEMPLATE_VERSION = @@TEMPLATE_VERSION_LITERAL@@

def _default_workdir():
    # Scratch directory for the checkout, the environment and the logs.
    # Kaggle exports everything under the notebook's working directory as run output, so putting the
    # repository and its virtual environment there makes a run's output tens of thousands of files
    # (observed: 148 MB of .venv before the download was abandoned) and an unattributable bundle. On
    # those platforms the scratch tree lives outside the exported directory; only EXPORT_DIR stays in.
    override = os.environ.get("SPECTRAQUANT_WORKDIR")
    if override:
        return Path(override)
    for scratch in ("/kaggle/temp", "/tmp"):
        if Path(scratch).is_dir():
            return Path(scratch) / "spectraquant-work" / RUN_ID
    return Path.cwd() / "spectraquant-work" / RUN_ID


WORKDIR = _default_workdir()
REPO_DIR = WORKDIR / "repo"
ARTIFACT_DIR = WORKDIR / "artifacts" / "run"
LOG_DIR = WORKDIR / "logs"
EXPORT_DIR = Path(os.environ.get("SPECTRAQUANT_EXPORT_DIR") or (Path.cwd() / "spectraquant-export" / RUN_ID))
for _directory in (WORKDIR, ARTIFACT_DIR, LOG_DIR, EXPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def _capture(argv, *, cwd=None):
    \"\"\"Run a command and return (returncode, combined stdout+stderr); never raises.\"\"\"
    try:
        proc = subprocess.run(
            [str(item) for item in argv],
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _require_ok(rc, out, what):
    \"\"\"Abort the notebook when a structural step fails (a truncated run must be obvious).\"\"\"
    if rc != 0:
        raise RuntimeError(f"{what} failed (exit {rc}):\\n{out[-4000:]}")


def _ram_bytes():
    try:
        import psutil  # optional
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


STARTED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
_START_MONOTONIC = time.monotonic()
_gpu_rc, _gpu_out = (
    _capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if shutil.which("nvidia-smi")
    else (127, "nvidia-smi not found")
)
_freeze_rc, _freeze_out = _capture([sys.executable, "-m", "pip", "freeze"])
ENVIRONMENT = {
    "run_id": RUN_ID,
    "started_utc": STARTED_UTC,
    "python": sys.version,
    "executable": sys.executable,
    "python_version_requested": SPEC["python_version"],
    "platform": _platform.platform(),
    "machine": _platform.machine(),
    "processor": _platform.processor(),
    "cpu_count": os.cpu_count(),
    "ram_bytes": _ram_bytes(),
    "nvidia_smi": _gpu_out.strip(),
    "gpu_required": SPEC["gpu_required"],
    "pip_freeze": [line for line in _freeze_out.splitlines() if line.strip()],
    "notebook_digest": NOTEBOOK_DIGEST,
    "notebook_template_version": NOTEBOOK_TEMPLATE_VERSION,
    "spec_sha256": SPEC_SHA256,
}
(ARTIFACT_DIR / "environment.json").write_text(json.dumps(ENVIRONMENT, indent=2, sort_keys=True))
print(json.dumps({key: value for key, value in ENVIRONMENT.items() if key != "pip_freeze"}, indent=2, sort_keys=True))
print("workdir:", WORKDIR, "| export:", EXPORT_DIR)
if SPEC["gpu_required"] and _gpu_rc != 0:
    raise RuntimeError("gpu_required=True but nvidia-smi reported no usable device: refusing to continue")
"""

_INSTALL = """\
# Mandatory cell 2/8 — pinned environment.
# INSTALL_SPEC is the exact command recorded in the RunSpec; it must not silently resolve newer
# versions (it materialises the committed uv.lock). uv is bootstrapped first because the platforms
# do not ship it.
INSTALL_SPEC = SPEC["install_spec"]
if shutil.which("uv") is None:
    _require_ok(*_capture([sys.executable, "-m", "pip", "install", "-q", "uv"]), "uv bootstrap")

# The pinned lock lives in the repository, so the install cell needs the checkout at the pinned
# commit. Cell 3 re-checks out that commit and asserts HEAD == spec.git_commit.
if not (REPO_DIR / ".git").is_dir():
    _require_ok(*_capture(["git", "clone", SPEC["repo_url"], str(REPO_DIR)]), "git clone")
_require_ok(*_capture(["git", "-C", str(REPO_DIR), "fetch", "--all", "--tags", "--prune"]), "git fetch")
if SPEC["git_commit"]:
    _require_ok(*_capture(["git", "-C", str(REPO_DIR), "checkout", "--detach", SPEC["git_commit"]]), "git checkout")

print("install_spec:", INSTALL_SPEC)
_install_rc, _install_out = _capture(["bash", "-lc", INSTALL_SPEC], cwd=REPO_DIR)
(LOG_DIR / "install.log").write_text(_install_out)
_require_ok(_install_rc, _install_out, "pinned install")
# The install above materialises the repository's own environment. The notebook's kernel, however,
# runs on the PLATFORM interpreter, so every later cell that imports repository code would fail on a
# dependency the kernel does not have (observed on the first real Kaggle run: "No module named
# 'hydra'"). The locked dependency set is therefore installed into THIS interpreter as well, so the
# kernel runs exactly the pinned versions instead of whatever the platform ships.
_LOCK_FILE = WORKDIR / "requirements.lock.txt"
_export_rc, _export_out = _capture(
    ["bash", "-lc", f"uv export --project {REPO_DIR} --no-hashes --no-emit-project "
                    f"--format requirements-txt -o {_LOCK_FILE}"],
    cwd=REPO_DIR,
)
_require_ok(_export_rc, _export_out, "uv export (locked requirements)")
# `python -m uv` is not available for every interpreter the platforms ship (observed:
# "/usr/bin/python3: No module named uv"), so resolve the uv EXECUTABLE and drive it explicitly.
UV_BIN = shutil.which("uv")
if UV_BIN is None:
    raise RuntimeError(
        "uv is not resolvable on PATH after the bootstrap install; the kernel install step cannot "
        "run (set PATH or install uv as a console script)"
    )
_kernel_rc, _kernel_out = _capture(
    [UV_BIN, "pip", "install", "--python", sys.executable, "-r", str(_LOCK_FILE)]
)
_require_ok(_kernel_rc, _kernel_out, "kernel install (locked dependencies)")
_editable_rc, _editable_out = _capture(
    [UV_BIN, "pip", "install", "--python", sys.executable, "--no-deps", "-e", str(REPO_DIR)]
)
_require_ok(_editable_rc, _editable_out, "kernel install (editable project)")
(LOG_DIR / "kernel_install.log").write_text(_kernel_out + _editable_out)
_freeze_rc, _freeze_out = _capture([sys.executable, "-m", "pip", "freeze"])
(ARTIFACT_DIR / "dependencies.txt").write_text(_freeze_out)
print(_install_out[-2000:])
"""

_REPO = """\
# Mandatory cell 3/8 — repository at the exact commit.
# The assert below is the reproducibility gate: a run whose checkout differs from the recorded SHA
# is not evidence.
if not (REPO_DIR / ".git").is_dir():
    _require_ok(*_capture(["git", "clone", SPEC["repo_url"], str(REPO_DIR)]), "git clone")
_require_ok(*_capture(["git", "-C", str(REPO_DIR), "fetch", "--all", "--tags", "--prune"]), "git fetch")
if SPEC["git_commit"]:
    _require_ok(*_capture(["git", "-C", str(REPO_DIR), "checkout", "--detach", SPEC["git_commit"]]), "git checkout")
_head_rc, _head_out = _capture(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"])
HEAD_SHA = _head_out.strip()
if SPEC["git_commit"] and not SPEC["allow_dirty"]:
    assert HEAD_SHA == SPEC["git_commit"], (
        f"checkout mismatch: HEAD={HEAD_SHA} != spec.git_commit={SPEC['git_commit']}"
    )
elif SPEC["git_commit"] and HEAD_SHA != SPEC["git_commit"]:
    print(f"WARNING: allow_dirty=True and HEAD={HEAD_SHA} != spec.git_commit={SPEC['git_commit']}: this run is not reproducible")
sys.path.insert(0, str(REPO_DIR / "src"))
print("checked out", HEAD_SHA, "into", REPO_DIR)
"""

_DATA = """\
# Mandatory cell 4/8 — datasets, checksum-verified.
# Fetching lives in spectraquant.cloud.remote (repo code at the pinned commit), never inline here.
from spectraquant.cloud.remote import materialise_datasets

DATASET_RECORDS = materialise_datasets(
    config_path=SPEC["experiment_config"],
    overrides=SPEC["overrides"],
    dataset_refs=SPEC["dataset_refs"],
    dest_dir=WORKDIR / "data",
    seed=int(SPEC["seeds"][0]),
)
(ARTIFACT_DIR / "datasets.json").write_text(json.dumps(DATASET_RECORDS, indent=2, sort_keys=True))
print(json.dumps(DATASET_RECORDS, indent=2, sort_keys=True))
"""

_RUN = """\
# Mandatory cell 5/8 — the declared experiment, executed by repo code.
# The runner comes from the spec (`runner_command`, design note section 11): a plan runs
# `spectraquant run-plan --plan <plan>`, a Hydra experiment runs `spectraquant run --config ...`.
# The notebook adds no logic of its own. A non-zero exit is NOT raised here: the failure is recorded
# in the manifest (AGENTS.md section 2b rule 4).
RUNNER_COMMAND = SPEC.get("runner_command")
if RUNNER_COMMAND:
    RUN_COMMAND = [sys.executable, "-m", *shlex.split(RUNNER_COMMAND)]
else:
    RUN_CONFIG = SPEC["experiment_config"]
    RUN_OVERRIDES = list(SPEC["overrides"])
    RUN_COMMAND = [sys.executable, "-m", "spectraquant", "run", "--config", RUN_CONFIG, *RUN_OVERRIDES]
RUN_LOG = LOG_DIR / "run.log"
RUN_RC, RUN_OUT = _capture(RUN_COMMAND, cwd=REPO_DIR)
RUN_LOG.write_text(RUN_OUT)
print("command:", " ".join(RUN_COMMAND))
print("exit:", RUN_RC)
print(RUN_OUT[-4000:])
if RUN_RC != 0 and "No such command" in RUN_OUT:
    print("HINT: this checkout has no such `spectraquant` subcommand; see scripts/cloud/README.md")
"""

_MANIFEST = """\
# Mandatory cell 6/8 — run_manifest.json (validated against artifacts/schemas/ before writing).
import time as _time

from spectraquant.cloud.remote import build_run_manifest, collect_artifact_checksums, find_run_manifest, write_run_manifest

FINISHED_UTC = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
WALL_TIME_S = round(_time.monotonic() - _START_MONOTONIC, 3)
# The run's own outputs (its manifest, checkpoints, tables) are part of the bundle.
if (REPO_DIR / "artifacts").is_dir():
    shutil.copytree(REPO_DIR / "artifacts", ARTIFACT_DIR / "artifacts", dirs_exist_ok=True)

_found_manifest = find_run_manifest(REPO_DIR, RUN_ID)
RUN_MANIFEST = json.loads(_found_manifest.read_text()) if _found_manifest else None
RUN_STATUS = "success" if RUN_RC == 0 else "failed"
FAILURE_REASON = None if RUN_RC == 0 else f"spectraquant run exited {RUN_RC}: {RUN_OUT[-500:].strip()}"
GPU_HOURS = round(WALL_TIME_S / 3600.0, 6) if SPEC["gpu_required"] else 0.0

MANIFEST_DOCUMENT = build_run_manifest(
    spec=SPEC,
    workdir=REPO_DIR,
    artifacts_dir=ARTIFACT_DIR,
    status=RUN_STATUS,
    started_utc=STARTED_UTC,
    finished_utc=FINISHED_UTC,
    environment=ENVIRONMENT,
    run_manifest=RUN_MANIFEST,
    failure_reason=FAILURE_REASON,
    log_path=RUN_LOG,
    remote_run_id=os.environ.get("SPECTRAQUANT_REMOTE_RUN_ID"),
    submitted_by=os.environ.get("SPECTRAQUANT_SUBMITTED_BY"),
    notebook_digest=NOTEBOOK_DIGEST,
    gpu_hours=GPU_HOURS,
    metrics={"cloud.wall_time_s": WALL_TIME_S},
)
CLOUD_CONTEXT = {
    "run_id": RUN_ID,
    "platform": SPEC["platform"],
    "spec_sha256": SPEC_SHA256,
    "notebook_digest": NOTEBOOK_DIGEST,
    "notebook_template_version": NOTEBOOK_TEMPLATE_VERSION,
    "git_commit": SPEC["git_commit"],
    "head_sha": HEAD_SHA,
    "status": RUN_STATUS,
    "started_utc": STARTED_UTC,
    "finished_utc": FINISHED_UTC,
    "wall_time_s": WALL_TIME_S,
    "gpu_hours": GPU_HOURS,
    "environment": ENVIRONMENT,
    "datasets": DATASET_RECORDS,
    "artifact_checksums": collect_artifact_checksums(ARTIFACT_DIR),
    "run_manifest_found": None if _found_manifest is None else str(_found_manifest),
    "install_spec": INSTALL_SPEC,
}
MANIFEST_PATH = write_run_manifest(out_dir=ARTIFACT_DIR, document=MANIFEST_DOCUMENT, cloud_context=CLOUD_CONTEXT)
print("manifest:", MANIFEST_PATH, "status:", RUN_STATUS)
"""

_EXPORT = """\
# Mandatory cell 7/8 — export the bundle and print the machine-readable result line.
import nbformat  # provided by the pinned environment


def _is_executed(path):
    \"\"\"True when a notebook file carries execution counts (i.e. it actually ran).\"\"\"
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    cells = document.get("cells") or []
    return any(
        cell.get("cell_type") == "code" and cell.get("execution_count") is not None
        for cell in cells
        if isinstance(cell, dict)
    )


def _capture_executed_notebook():
    \"\"\"Best-effort copy of the *executed* notebook into the export bundle.\"\"\"
    destination = EXPORT_DIR / "notebook" / f"{RUN_ID}.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    explicit = os.environ.get("SPECTRAQUANT_NOTEBOOK_PATH")
    if explicit and Path(explicit).is_file() and _is_executed(explicit):
        shutil.copy2(explicit, destination)
        return destination
    for candidate in sorted(Path.cwd().glob("*.ipynb")):
        if _is_executed(candidate):
            shutil.copy2(candidate, destination)
            return destination
    try:  # Colab: ask the front end for the live notebook (best effort, never fatal)
        from google.colab import _message
        payload = _message.blocking_request("get_ipynb", timeout_sec=30)["ipynb"]
        nbformat.write(nbformat.from_dict(payload), str(destination))
        return destination
    except Exception as exc:
        print(f"could not capture the executed notebook automatically: {type(exc).__name__}: {exc}")
    print(
        "ACTION REQUIRED: download the executed notebook from the platform into "
        f"{destination} — `spectraquant cloud collect` rejects a bundle without it."
    )
    return None


EXECUTED_NOTEBOOK = _capture_executed_notebook()
BUNDLE_FILES = {}
for _item in sorted(ARTIFACT_DIR.rglob("*")):
    if _item.is_file():
        _relative = _item.relative_to(ARTIFACT_DIR)
        _target = EXPORT_DIR / _relative
        _target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_item, _target)
        BUNDLE_FILES[str(_relative)] = hashlib.sha256(_item.read_bytes()).hexdigest()
if EXECUTED_NOTEBOOK is not None:
    BUNDLE_FILES[str(EXECUTED_NOTEBOOK.relative_to(EXPORT_DIR))] = hashlib.sha256(
        EXECUTED_NOTEBOOK.read_bytes()
    ).hexdigest()

_drive = Path("/content/drive/MyDrive/spectraquant-runs")
if _drive.parent.is_dir():
    shutil.copytree(EXPORT_DIR, _drive / RUN_ID, dirs_exist_ok=True)
    print("mirrored to", _drive / RUN_ID)

RESULT = {
    "run_id": RUN_ID,
    "platform": SPEC["platform"],
    "status": RUN_STATUS,
    "export_dir": str(EXPORT_DIR),
    "manifest": str(EXPORT_DIR / "run_manifest.json"),
    "executed_notebook": None if EXECUTED_NOTEBOOK is None else str(EXECUTED_NOTEBOOK),
    "notebook_digest": NOTEBOOK_DIGEST,
    "spec_sha256": SPEC_SHA256,
    "git_commit": SPEC["git_commit"],
    "head_sha": HEAD_SHA,
    "gpu_hours": GPU_HOURS,
    "finished_utc": FINISHED_UTC,
    "files": BUNDLE_FILES,
}
print("SPECTRAQUANT_RESULT_JSON=" + json.dumps(RESULT, sort_keys=True, separators=(",", ":")))
"""

_TEARDOWN = """\
# Mandatory cell 8/8 — teardown marker: a truncated run is detectable because this line is missing.
TEARDOWN = {
    "run_id": RUN_ID,
    "status": RUN_STATUS,
    "teardown_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "manifest": str(MANIFEST_PATH),
}
print("SPECTRAQUANT_TEARDOWN_OK " + json.dumps(TEARDOWN, sort_keys=True))
"""


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def _spec_literals(spec: Any) -> dict[str, str]:
    spec_json = json.dumps(spec.to_json(), sort_keys=True, separators=(",", ":"))
    return {
        "SPEC_LITERAL": repr(spec_json),
        "SPEC_SHA256_LITERAL": repr(spec.sha256()),
        "TEMPLATE_VERSION_LITERAL": repr(NOTEBOOK_TEMPLATE_VERSION),
    }


def _sub(template: str, values: dict[str, str]) -> str:
    """Substitute ``@@TOKEN@@`` placeholders (templates contain literal braces, so no ``format``)."""
    rendered = template
    for token, value in values.items():
        rendered = rendered.replace(f"@@{token}@@", value)
    return rendered


def _cell_sources(spec: Any, *, digest: str) -> list[tuple[str, str]]:
    """Return ``(stage, source)`` pairs for the notebook, in the frozen order."""
    overrides = " ".join(spec.overrides) or "(none)"
    header_values = {
        "RUN_ID": spec.run_id,
        "TEMPLATE_VERSION": NOTEBOOK_TEMPLATE_VERSION,
        "PLATFORM": spec.platform,
        "GPU_REQUIRED": str(spec.gpu_required),
        "TIMEOUT_MINUTES": str(spec.timeout_minutes),
        "EXPERIMENT_CONFIG": spec.experiment_config,
        "RUNNER_COMMAND": spec.runner_command or "spectraquant run --config <experiment_config>",
        "OVERRIDES": overrides,
        "GIT_COMMIT": spec.git_commit or "(unset)",
        "DIRTY_NOTE": " (DIRTY — not reproducible)" if spec.allow_dirty else "",
        "MEASUREMENT_CLASS": str(spec.measurement_class_expected),
        "SPEC_SHA256": spec.sha256(),
        "NOTEBOOK_DIGEST": digest,
    }
    environment_values = {
        **_spec_literals(spec),
        "NOTEBOOK_DIGEST_LITERAL": repr(digest),
    }
    sources = [
        ("header", _sub(_HEADER, header_values)),
        ("environment", _sub(_ENVIRONMENT, environment_values)),
        ("install", _INSTALL),
        ("repo", _REPO),
        ("data", _DATA),
        ("run", _RUN),
        ("manifest", _MANIFEST),
        ("export", _EXPORT),
        ("teardown", _TEARDOWN),
    ]
    return sources


def _build_node(spec: Any, *, digest: str) -> nbformat.NotebookNode:
    nbformat = require_nbformat()
    cells = []
    for index, (stage, source) in enumerate(_cell_sources(spec, digest=digest)):
        cell_id = f"sq-{index:02d}-{stage}"
        if stage == "header":
            cell = nbformat.v4.new_markdown_cell(source=source, id=cell_id)
        else:
            cell = nbformat.v4.new_code_cell(source=source, id=cell_id)
        cell.metadata[_METADATA_KEY] = {
            "stage": stage,
            "run_id": spec.run_id,
            "template_version": NOTEBOOK_TEMPLATE_VERSION,
            "generated": True,
        }
        cells.append(cell)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
        "language_info": {"name": "python"},
        _METADATA_KEY: {
            "run_id": spec.run_id,
            "platform": spec.platform,
            "template_version": NOTEBOOK_TEMPLATE_VERSION,
            "spec_sha256": spec.sha256(),
            "generated": True,
            "generator": "spectraquant.cloud.notebook",
        },
    }
    return notebook


def _serialize(notebook: nbformat.NotebookNode) -> str:
    nbformat = require_nbformat()
    return str(nbformat.writes(notebook, version=4))


def notebook_digest(spec: Any) -> str:
    """Return ``sha256:<hex>`` of the notebook's canonical serialization.

    The canonical serialization is the generated notebook with the ``NOTEBOOK_DIGEST`` literal set to
    :data:`PENDING_DIGEST` (a notebook cannot contain the hash of its own bytes). :func:`build_notebook`
    substitutes this digest into that literal, so the value is recomputable from the repository.
    """
    text = redact(_serialize(_build_node(spec, digest=PENDING_DIGEST)))
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def build_notebook(spec: Any) -> nbformat.NotebookNode:
    """Build the notebook for ``spec`` (deterministic; pure function of spec + template version).

    Args:
        spec: a validated :class:`spectraquant.cloud.spec.RunSpec`.

    Returns:
        An ``nbformat.NotebookNode`` with the eight mandatory cells in the frozen order.
    """
    node = _build_node(spec, digest=notebook_digest(spec))
    # Defense in depth: no credential value may reach a generated cell.
    for cell in node.cells:
        cell.source = redact(str(cell.source))
    return node


def notebook_text(spec: Any) -> str:
    """Return the serialized, redacted notebook text that :func:`write_notebook` writes."""
    text = redact(_serialize(build_notebook(spec)))
    assert_no_secret(text, where=f"generated notebook for {spec.run_id}")
    return text


def cell_stages(spec: Any) -> list[dict[str, Any]]:
    """Return ``[{"index", "stage", "cell_type", "chars"}]`` for the notebook's cells."""
    node = build_notebook(spec)
    described: list[dict[str, Any]] = []
    for index, cell in enumerate(node.cells):
        stage = str(cell.get("metadata", {}).get(_METADATA_KEY, {}).get("stage", "?"))
        described.append(
            {
                "index": index,
                "stage": stage,
                "cell_type": str(cell.get("cell_type", "code")),
                "chars": len(str(cell.get("source", ""))),
            }
        )
    return described


def write_notebook(spec: Any, out_path: str | Path) -> str:
    """Write the generated notebook to ``out_path`` and return its digest.

    Args:
        spec: a validated :class:`RunSpec`.
        out_path: destination ``.ipynb`` path; parents are created.

    Returns:
        :func:`notebook_digest` of ``spec``.
    """
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = notebook_text(spec)
    target.write_text(text, encoding="utf-8")
    return notebook_digest(spec)
