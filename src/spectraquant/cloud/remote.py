"""Remote-side helpers imported by generated notebooks (``design-cloud-adapter.md`` §3).

Everything a notebook cell needs to do beyond shelling out lives here, so the notebook stays
plumbing-only ("thin notebooks, thick modules", ``AGENTS.md`` §2b rule 1). These functions execute
*on the remote host*, inside the checked-out repository at the pinned commit, and they are also
exercised locally by the unit tests (the synthetic dataset path needs no network).

Public API:

* :func:`synthetic_split_checksum` — content checksum of one synthetic split, used both when the
  spec is built locally and when the dataset is materialised remotely.
* :func:`materialise_datasets` — fetch/rebuild every ``DatasetRef`` and verify its checksum.
* :func:`collect_artifact_checksums` — ``sha256:<hex>`` per file in an artifact directory.
* :func:`find_run_manifest` / :func:`build_run_manifest` / :func:`write_run_manifest` — the
  schema-validated ``run_manifest.json`` plus its ``cloud_context.json`` sidecar.
* :func:`parse_result_line` — parse the single ``SPECTRAQUANT_RESULT_JSON=`` log line.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "RESULT_LINE_PREFIX",
    "TEARDOWN_LINE_PREFIX",
    "ChecksumMismatch",
    "build_run_manifest",
    "collect_artifact_checksums",
    "find_run_manifest",
    "has_teardown_marker",
    "materialise_datasets",
    "parse_result_line",
    "sha256_file",
    "sha256_of_files",
    "synthetic_split_checksum",
    "write_run_manifest",
]

#: Prefix of the single machine-readable result line printed by the export cell.
RESULT_LINE_PREFIX = "SPECTRAQUANT_RESULT_JSON="

#: Prefix of the teardown marker that makes a truncated run detectable.
TEARDOWN_LINE_PREFIX = "SPECTRAQUANT_TEARDOWN_OK "

_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


class ChecksumMismatch(ValueError):
    """Raised when a materialised dataset's checksum differs from the declared one."""

    def __init__(self, name: str, expected: str, actual: str) -> None:
        self.name = name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"dataset {name!r} checksum mismatch: declared {expected}, materialised {actual} "
            "(refusing to run on unverified data)"
        )


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
def synthetic_split_checksum(data_cfg: Any, *, seed: int, split: str = "train") -> str:
    """Return ``sha256:<hex>`` over the raw bytes of one split of the synthetic corpus.

    Args:
        data_cfg: validated ``spectraquant.config.DataConfig``.
        seed: master seed for the deterministic LCG corpus.
        split: ``train``/``val``/``validation``/``test`` (aliases accepted).

    Returns:
        The digest of that split's contiguous int64 token bytes.

    Raises:
        ValueError: ``split`` is not a known split name.
    """
    from spectraquant.data.synthetic import build_corpus, sequence_checksum

    corpus = build_corpus(data_cfg, seed=int(seed))
    key = _SPLIT_ALIASES.get(split.strip().lower())
    if key is None:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(_SPLIT_ALIASES)}")
    if key == "train":
        return sequence_checksum(corpus.train)
    if key == "val":
        return sequence_checksum(corpus.val)
    return sequence_checksum(corpus.train, corpus.val)


def materialise_datasets(
    *,
    config_path: str | Path,
    overrides: Sequence[str],
    dataset_refs: Sequence[Mapping[str, Any]],
    dest_dir: str | Path,
    seed: int,
) -> list[dict[str, Any]]:
    """Materialise every dataset ref and verify its declared checksum.

    Two shapes exist, dispatched on the config document (``design-cloud-adapter.md`` §11):

    * a **plan** (``configs/{tier1,tier2,repro}/*.yaml``) — its datasets are pinned by immutable
      Hugging Face revisions, verified by :func:`spectraquant.cloud.plan_data.materialise_plan_datasets`
      against the spec's declaration digests. Bulk download is deliberately skipped (C4 train is
      ~1 TB); the runner streams the splits it consumes.
    * a **Hydra experiment config** — only the ``synthetic`` data kind exists in the Tier-0/1
      scaffold; an unknown kind raises ``NotImplementedError`` rather than silently skipping the
      verification.

    Args:
        config_path: plan or experiment config composed on the remote host.
        overrides: Hydra overrides, recorded verbatim (unused by the plan path).
        dataset_refs: the spec's ``dataset_refs`` as plain mappings.
        dest_dir: directory to write the materialised payload into.
        seed: master seed used to rebuild generated data.

    Returns:
        One record per ref.

    Raises:
        ChecksumMismatch: a materialised split does not match its declared checksum.
        PinnedRevisionError: a plan asset pin is floating or disagrees with the spec.
        NotImplementedError: the config's data kind has no remote fetcher.
    """
    from spectraquant.cloud.plan_data import is_plan_document, materialise_plan_datasets

    if is_plan_document(config_path):
        return materialise_plan_datasets(config_path, dataset_refs, dest_dir)

    from spectraquant.config import load_experiment_config

    cfg = load_experiment_config(config_path, list(overrides))
    kind = str(getattr(cfg.data, "kind", "synthetic"))
    if kind != "synthetic":
        raise NotImplementedError(
            f"data.kind={kind!r} has no remote fetcher in this milestone; only 'synthetic' "
            "materialises without a network (see docs/coordination/design-cloud-adapter.md §3)"
        )

    from spectraquant.data.synthetic import build_corpus

    corpus = build_corpus(cfg.data, seed=int(seed))
    target = Path(dest_dir)
    target.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for ref in dataset_refs:
        name = str(ref["name"])
        split = str(ref["split"])
        payload = target / name
        payload.mkdir(parents=True, exist_ok=True)
        key = _SPLIT_ALIASES.get(split.strip().lower())
        if key is None:
            raise ValueError(f"unknown split {split!r} for dataset {name!r}")
        tensor = corpus.train if key == "train" else corpus.val
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        out_file = payload / f"{key}.bin"
        out_file.write_bytes(raw)
        digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        declared = str(ref["checksum"])
        if declared.removeprefix("sha256:") != digest.removeprefix("sha256:"):
            raise ChecksumMismatch(name, declared, digest)
        records.append(
            {
                "name": name,
                "split": split,
                "checksum": digest,
                "path": str(out_file),
                "bytes": len(raw),
            }
        )
    return records


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    """Return ``sha256:<hex>`` of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def collect_artifact_checksums(directory: str | Path) -> dict[str, str]:
    """Return ``{relative posix path: sha256:<hex>}`` for every file under ``directory``.

    Directories are walked in sorted order so the mapping is deterministic.
    """
    root = Path(directory)
    checksums: dict[str, str] = {}
    if not root.exists():
        return checksums
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        checksums[path.relative_to(root).as_posix()] = sha256_file(path)
    return checksums


def sha256_of_files(paths: Sequence[str | Path], *, root: str | Path | None = None) -> str:
    """Return ``sha256:<hex>`` over the sorted ``relative-path + bytes`` stream of ``paths``.

    Used for dataset payloads that consist of more than one file; the digest covers the file names
    as well as their bytes, so a rename is a checksum change.
    """
    base = Path(root) if root is not None else None
    digest = hashlib.sha256()
    ordered = sorted(Path(p) for p in paths)
    for path in ordered:
        label = path.relative_to(base).as_posix() if base is not None else path.name
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(Path(path).read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def find_run_manifest(workdir: str | Path, run_id: str) -> Path | None:
    """Return the newest ``run_manifest.json`` written by the run, or ``None``.

    Searched in ``<workdir>/artifacts/runs/<run_id>/`` and ``<workdir>/artifacts/**`` so the cloud
    step works with either the run's own layout or the export directory.
    """
    root = Path(workdir)
    candidates: list[Path] = []
    exact = root / "artifacts" / "runs" / run_id / "run_manifest.json"
    if exact.is_file():
        candidates.append(exact)
    for base in (root / "artifacts", root):
        if base.is_dir():
            candidates.extend(path for path in base.rglob("run_manifest.json") if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _git_head(workdir: Path) -> tuple[str | None, bool | None]:
    """Return ``(HEAD sha, dirty)`` for the checkout, or ``(None, None)`` when unavailable."""

    def _run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(workdir), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - git always present
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    head = _run(["rev-parse", "HEAD"])
    status = _run(["status", "--porcelain", "--untracked-files=no"])
    return (head or None), (None if status is None else bool(status))


def _runtime_backend(run_manifest: Mapping[str, Any] | None) -> str:
    """Return the manifest ``runtime_backend``: the run's own value, else the detected device."""
    if run_manifest:
        declared = run_manifest.get("runtime_backend")
        if isinstance(declared, str) and declared:
            return declared
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "not-applicable"
    return "torch-cuda-fp16" if torch.cuda.is_available() else "torch-cpu-fp32"


def build_run_manifest(
    *,
    spec: Mapping[str, Any],
    workdir: str | Path,
    artifacts_dir: str | Path,
    status: str,
    started_utc: str,
    finished_utc: str,
    environment: Mapping[str, Any] | None = None,
    run_manifest: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    failure_reason: str | None = None,
    log_path: str | Path | None = None,
    remote_run_id: str | None = None,
    submitted_by: str | None = None,
    notebook_digest: str | None = None,
    gpu_hours: float | None = None,
    extra_artifact_paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the cloud ``run_manifest.json`` document (schema-validated by the caller).

    The run's own manifest, when the run wrote one, is the base: its measured values are never
    replaced by estimates. Cloud-specific provenance (GPU-hours, notebook digest, remote run id,
    artifact checksums) is recorded in ``metrics`` because the published manifest schema forbids
    additional top-level properties, and mirrored in ``cloud_context.json``.
    """
    plan_note = (
        "A plan spec (``design-cloud-adapter.md`` §11) is resolved as a plan rather than a Hydra "
        "composition: this wrapper manifest is provenance only (``method='none'``, no class), and "
        "the per-arm manifests the runner wrote — class 2 for quality, class 1/3 for bytes — are "
        "the evidence."
    )
    from spectraquant.cloud.plan_data import is_plan_document
    from spectraquant.config import load_experiment_config
    from spectraquant.experiment_plan import load_plan
    from spectraquant.reporting.environment import collect_hardware, collect_software
    from spectraquant.reporting.manifests import utc_timestamp

    root = Path(workdir)
    config_path = str(spec["experiment_config"])
    plan_metrics: dict[str, Any] = {}
    if is_plan_document(config_path):
        plan = load_plan(config_path)
        resolved_config = plan.model_dump(mode="json")
        compression = {
            "method": "none",
            "ranks": None,
            "bits": None,
            "group_size": None,
            "exclusions": [],
        }
        measurement_class: int | None = None
        plan_metrics = {
            "cloud.plan.name": plan.name,
            "cloud.plan.tier": int(plan.tier),
            "cloud.plan.substrate": plan.substrate,
            "cloud.plan.arms": [arm.name for arm in plan.arms],
            "cloud.plan.measurement_classes_authorised": [
                int(value) for value in plan.measurement_classes
            ],
            "cloud.plan.dataset_checksum_semantics": "revision-declaration-digest",
            "cloud.plan.per_arm_manifests": f"artifacts/runs/{plan.name}-plan/",
            "cloud.plan.note": plan_note,
        }
    else:
        cfg = load_experiment_config(config_path, list(spec.get("overrides") or []))
        resolved_config = cfg.model_dump(mode="json")
        resolved_config.pop("config_path", None)
        compression = {
            "method": cfg.method.compression.method,
            "ranks": cfg.method.compression.ranks,
            "bits": cfg.method.compression.bits,
            "group_size": cfg.method.compression.group_size,
            "exclusions": list(cfg.method.compression.exclusions),
        }
        declared_class = cfg.method.measurement_class
        measurement_class = None if cfg.method.compression.method == "none" else declared_class
        if measurement_class is None and cfg.method.compression.method != "none":
            measurement_class = int(spec["measurement_class_expected"])
    plan_branch = bool(plan_metrics)

    commit, dirty = _git_head(root)
    if commit is None:
        commit = str(spec.get("git_commit") or "") or None
        dirty = bool(spec.get("allow_dirty", False))

    checksums = collect_artifact_checksums(artifacts_dir)
    artifact_paths = sorted(checksums) + sorted(extra_artifact_paths)

    merged_metrics: dict[str, Any] = {}
    if run_manifest:
        raw_metrics = run_manifest.get("metrics")
        if isinstance(raw_metrics, Mapping):
            merged_metrics.update(raw_metrics)
    merged_metrics.update(plan_metrics)
    merged_metrics.update(
        {
            "cloud.started_utc": started_utc,
            "cloud.finished_utc": finished_utc,
            "cloud.gpu_hours": 0.0 if gpu_hours is None else round(float(gpu_hours), 6),
            "cloud.run_id": str(spec["run_id"]),
            "cloud.platform": str(spec["platform"]),
            "cloud.gpu_required": bool(spec.get("gpu_required", False)),
            "cloud.notebook_digest": notebook_digest,
            "cloud.remote_run_id": remote_run_id,
            "cloud.submitted_by": submitted_by,
            "cloud.spec_sha256": _spec_sha256(spec),
            "cloud.artifact_checksums": [f"{name}={digest}" for name, digest in checksums.items()],
        }
    )
    if metrics:
        merged_metrics.update({str(key): value for key, value in metrics.items()})
    merged_metrics = {key: value for key, value in merged_metrics.items() if value is not None}

    training: dict[str, Any] = {"steps": 0, "tokens": 0, "wall_time_s": 0.0, "peak_mem_mb": None}
    if run_manifest and isinstance(run_manifest.get("training"), Mapping):
        training.update(dict(run_manifest["training"]))

    dataset_refs = list(spec.get("dataset_refs") or [])
    document: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(spec["run_id"]),
        "timestamp_utc": started_utc or utc_timestamp(),
        "git_commit": commit,
        "git_dirty": dirty,
        "config_path": config_path,
        "resolved_config": resolved_config,
        "model_id": _from_run(run_manifest, "model_id") if plan_branch else None,
        "model_revision": _from_run(run_manifest, "model_revision") if plan_branch else None,
        "dataset_ids": [str(ref["name"]) for ref in dataset_refs],
        "dataset_revisions": [str(ref.get("revision")) for ref in dataset_refs],
        "dataset_checksums": [str(ref["checksum"]) for ref in dataset_refs],
        "split": str(dataset_refs[0]["split"]) if dataset_refs else "train",
        "seed": int((spec.get("seeds") or [0])[0]),
        "hardware": collect_hardware(),
        "software": collect_software(),
        "compression": compression,
        "theoretical_bits": _from_run(run_manifest, "theoretical_bits"),
        "packed_bytes": _from_run(run_manifest, "packed_bytes"),
        "runtime_backend": _runtime_backend(run_manifest),
        "measurement_class": measurement_class,
        "training": training,
        "metrics": merged_metrics,
        "status": status,
        "failure_reason": failure_reason,
        "log_path": None if log_path is None else str(log_path),
        "artifact_paths": artifact_paths,
    }
    if environment is not None:
        document["software"] = {
            **document["software"],
            "remote.python": str(environment.get("python") or "") or None,
            "remote.platform": str(environment.get("platform") or "") or None,
            "remote.nvidia_smi": _truncate(environment.get("nvidia_smi")),
        }
    return document


def write_run_manifest(
    *,
    out_dir: str | Path,
    document: Mapping[str, Any],
    cloud_context: Mapping[str, Any] | None = None,
) -> Path:
    """Validate ``document`` against the published schema and write it, plus ``cloud_context.json``.

    Args:
        out_dir: directory that receives ``run_manifest.json`` and ``cloud_context.json``.
        document: the manifest document.
        cloud_context: full provenance sidecar (not constrained by the manifest schema).

    Returns:
        Path of the written ``run_manifest.json``.

    Raises:
        ManifestValidationError: the document is not a valid manifest.
    """
    from spectraquant.reporting.manifests import validate_manifest_dict

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    validate_manifest_dict(document, source="cloud run_manifest.json")
    manifest_path = target_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if cloud_context is not None:
        (target_dir / "cloud_context.json").write_text(
            json.dumps(cloud_context, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return manifest_path


def parse_result_line(text: str) -> dict[str, Any] | None:
    """Return the parsed ``SPECTRAQUANT_RESULT_JSON=`` payload from ``text``, or ``None``.

    The last matching line wins (a re-run appends a second line), and a malformed payload is
    ignored rather than raising: the collector re-verifies the manifest anyway.
    """
    payload: dict[str, Any] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(RESULT_LINE_PREFIX):
            continue
        try:
            parsed = json.loads(stripped[len(RESULT_LINE_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
    return payload


def has_teardown_marker(text: str) -> bool:
    """True when the log contains the teardown marker (a truncated run does not)."""
    return any(line.strip().startswith(TEARDOWN_LINE_PREFIX) for line in text.splitlines())


def _from_run(run_manifest: Mapping[str, Any] | None, key: str) -> Any:
    return None if not run_manifest else run_manifest.get(key)


def _spec_sha256(spec: Mapping[str, Any]) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _truncate(value: Any, limit: int = 400) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit] or None
