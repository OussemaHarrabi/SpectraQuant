"""The frozen ``RunSpec`` — the only input to the cloud execution adapters.

Contract: ``docs/coordination/design-cloud-adapter.md`` §2. Field names, types and defaults are
frozen; validators below encode the rules that document states in prose:

* ``max_cost_authorized_usd`` is **required** on a paid platform (hard error, not a warning);
* ``git_commit`` is required — a full 40-character SHA — unless ``allow_dirty=True``;
* ``measurement_class_expected`` is in ``1..4`` (classes 4-GPU/5 are not claimable from this
  workstation and are labelled 4 per the platform's hardware, see ``measurement-taxonomy.md`` §1);
* ``platform`` is one of ``colab``, ``kaggle``, ``colab_enterprise``, ``local_cpu``;
* ``local_cpu`` with ``gpu_required=True`` is an error (the workstation has no CUDA device).

The spec is immutable (``frozen=True``) and cannot carry unknown keys (``extra="forbid"``): a typo
in a submitted spec must fail loudly rather than silently default.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spectraquant.cloud.secrets import redact_mapping

__all__ = [
    "PAID_PLATFORMS",
    "PLATFORMS",
    "PLATFORM_MAX_TIMEOUT_MINUTES",
    "DatasetRef",
    "ExpectedArtifact",
    "Platform",
    "RunSpec",
    "checksum_hex",
    "dataset_declaration_digest",
    "load_spec",
    "plan_to_run_spec",
    "save_spec",
    "spec_from_config",
    "spec_sha256",
]

Platform = Literal["colab", "kaggle", "colab_enterprise", "local_cpu"]

#: Every supported platform, in the order the design note lists them.
PLATFORMS: tuple[str, ...] = ("colab", "kaggle", "colab_enterprise", "local_cpu")

#: Platforms that bill the user. Everything else is a documented free tier.
PAID_PLATFORMS: frozenset[str] = frozenset({"colab_enterprise"})

#: Documented free-tier / platform session ceilings (minutes). A spec may not exceed these.
PLATFORM_MAX_TIMEOUT_MINUTES: dict[str, int] = {
    "colab": 720,  # 12 h consumer Colab session ceiling
    "kaggle": 720,  # 12 h Kaggle notebook run ceiling
    "colab_enterprise": 1440,
    "local_cpu": 1440,
}

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def checksum_hex(value: str) -> str:
    """Return the bare 64-character hex digest of a ``sha256:<hex>`` (or bare hex) string.

    Raises:
        ValueError: ``value`` is not a sha256 digest in either accepted spelling.
    """
    if not _SHA256.match(value):
        raise ValueError(
            f"not a sha256 digest: {value!r} (expected 'sha256:<64 hex>' or '<64 hex>')"
        )
    return value.split(":", 1)[1] if value.startswith("sha256:") else value


def _validate_checksum(value: str) -> str:
    checksum_hex(value)
    return value


def dataset_declaration_digest(name: str, revision: str, split: str) -> str:
    """Return the ``sha256:<hex>`` **declaration digest** of one plan dataset reference.

    A frozen plan pins its datasets by *immutable repository revision* (a Hugging Face commit), not
    by a payload checksum: the payload cannot be hashed locally without downloading it, and a dataset
    a run must not be re-derived silently. The declaration digest therefore covers the triple the
    plan authorises — ``name``, ``revision``, ``split`` — so a remote run can prove that the spec it
    received is the spec the plan froze, and a mismatch is a loud failure.

    It is **not** a content checksum: it must never be reported as a class-3 byte or payload figure
    (``AGENTS.md`` §5), and the *measured* dataset fingerprint a run records is separate. Datasets
    that this repository generates itself (the synthetic Tier-0 corpus) are pinned by a real payload
    checksum instead; see :func:`spectraquant.cloud.remote.synthetic_split_checksum`.

    Args:
        name: dataset identifier (an HF repo id).
        revision: the immutable revision the plan pins.
        split: the split the run consumes.

    Returns:
        ``sha256:<64 hex>`` over the ``\\n``-joined triple, UTF-8 encoded.
    """
    payload = "\n".join((name, revision, split)).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class DatasetRef(BaseModel):
    """One dataset input: identity, immutable revision, split and content checksum.

    ``checksum`` has two spellings, distinguished by how the dataset is pinned (design note §11):

    * *payload checksum* — the sha256 of the bytes a run must reproduce. Used for the synthetic
      Tier-0 corpus this repository generates itself (the remote run rebuilds it and compares).
    * *declaration digest* — :func:`dataset_declaration_digest` over ``(name, revision, split)``,
      used for the frozen plans' Hugging Face datasets, whose integrity anchor is the immutable
      repository revision rather than a locally computable payload hash. A run that cannot honour
      the pinned revision MUST fail loudly; it must never load an unpinned revision and report the
      declared digest as if the payload had been verified.

    Attributes:
        name: dataset identifier (an HF repo id, or a locally generated dataset name).
        revision: immutable revision (commit/tag) or a content version for generated data.
        split: split this run consumes (``train``/``validation``/...).
        checksum: ``sha256:<64 hex>``; see the two spellings above.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    split: str = Field(min_length=1)
    checksum: str

    @field_validator("checksum")
    @classmethod
    def _checksum_ok(cls, value: str) -> str:
        return _validate_checksum(value)

    def to_json(self) -> dict[str, str]:
        """Return a JSON-serializable mapping."""
        return self.model_dump(mode="json")


class ExpectedArtifact(BaseModel):
    """An artifact the remote run must produce, with the checks it must satisfy.

    Attributes:
        name: path of the artifact relative to the run's export directory.
        sha256: expected digest (``sha256:<hex>`` or bare hex), or ``None`` to check size only.
        min_bytes: minimum acceptable size in bytes (guards against truncated/empty exports).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    sha256: str | None = None
    min_bytes: int = Field(default=0, ge=0)

    @field_validator("sha256")
    @classmethod
    def _sha256_ok(cls, value: str | None) -> str | None:
        return None if value is None else _validate_checksum(value)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return self.model_dump(mode="json")


class RunSpec(BaseModel):
    """Frozen description of one cloud run (``design-cloud-adapter.md`` §2).

    Attributes:
        run_id: slug identifying the run; unique and stable across retries.
        experiment_config: path under ``configs/experiment/`` to compose remotely.
        overrides: Hydra overrides, recorded verbatim.
        platform: execution substrate.
        repo_url: clone URL of this repository.
        git_commit: resolved SHA of the code to execute.
        allow_dirty: permit an empty/partial ``git_commit`` for a non-reproducible local run.
        python_version: interpreter requested on the remote host.
        install_spec: the exact, pinned command that materialises the environment.
        dataset_refs: datasets to materialise and checksum-verify before the run.
        seeds: master seeds; ``seeds[0]`` is the manifest's master seed.
        gpu_required: whether the run needs a CUDA device.
        timeout_minutes: hard wall-clock ceiling for the remote run.
        max_cost_authorized_usd: user-authorized spend ceiling (required on paid platforms).
        expected_artifacts: artifacts the collection step must find and verify.
        measurement_class_expected: the class the run intends to report (1..4).
        runner_command: the repository CLI invocation the run cell executes, as a
            ``spectraquant`` invocation without the interpreter (e.g.
            ``"spectraquant run-plan --plan configs/tier1/smollm2_135m.yaml"``). ``None`` means the
            frozen default (``spectraquant run --config <experiment_config> <overrides>``). Added by
            the 2026-10-09 amendment (``design-cloud-adapter.md`` §11) so a plan can carry its own
            runner: the plan runner is a different entry point from the Hydra composition runner.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, max_length=128)
    experiment_config: str = Field(min_length=1)
    runner_command: str | None = None
    overrides: list[str]
    platform: Platform
    repo_url: str = Field(min_length=1)
    git_commit: str
    allow_dirty: bool = False
    python_version: str = "3.11"
    install_spec: str = Field(min_length=1)
    dataset_refs: list[DatasetRef]
    seeds: list[int]
    gpu_required: bool
    timeout_minutes: int = Field(gt=0)
    max_cost_authorized_usd: float | None = Field(default=None, ge=0.0)
    expected_artifacts: list[ExpectedArtifact]
    measurement_class_expected: int = Field(ge=1, le=4)

    @field_validator("run_id")
    @classmethod
    def _run_id_is_a_slug(cls, value: str) -> str:
        if not _RUN_ID.match(value):
            raise ValueError(
                "run_id must be a filesystem-safe slug matching ^[A-Za-z0-9][A-Za-z0-9._-]*$"
            )
        return value

    @field_validator("seeds")
    @classmethod
    def _seeds_are_non_negative(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("seeds must list at least one master seed")
        if any(seed < 0 for seed in value):
            raise ValueError("seeds must be non-negative")
        return value

    @field_validator("runner_command")
    @classmethod
    def _runner_command_is_a_spectraquant_invocation(cls, value: str | None) -> str | None:
        """A spec may only route the run cell to this repository's own CLI (design note §11).

        The string is the argv *without* the interpreter: the run cell prepends
        ``sys.executable, "-m"``, so the first token must be ``spectraquant``. Anything else
        (``bash -c ...``, a path to a downloaded script, a raw python heredoc) would put logic in a
        generated cell and is refused.
        """
        if value is None:
            return None
        tokens = shlex.split(value)
        if not tokens or tokens[0] != "spectraquant":
            raise ValueError(
                "runner_command must be a 'spectraquant ...' invocation without the interpreter "
                f"(got {value!r}); the generated run cell executes "
                "`python -m spectraquant <runner_command>`"
            )
        if len(tokens) < 2:
            raise ValueError(f"runner_command {value!r} names no subcommand")
        return value

    @model_validator(mode="after")
    def _platform_rules(self) -> RunSpec:
        if self.git_commit:
            if not self.allow_dirty and not _SHA40.match(self.git_commit):
                raise ValueError(
                    "git_commit must be a full 40-character lowercase SHA unless allow_dirty=True "
                    f"(got {self.git_commit!r})"
                )
        elif not self.allow_dirty:
            raise ValueError("git_commit is required unless allow_dirty=True")

        if self.platform in PAID_PLATFORMS and self.max_cost_authorized_usd is None:
            raise ValueError(
                f"platform {self.platform!r} is paid: max_cost_authorized_usd is required before "
                "anything may be emitted or submitted (AGENTS.md §2b rule 8)"
            )

        if self.platform == "local_cpu" and self.gpu_required:
            raise ValueError(
                "platform 'local_cpu' cannot satisfy gpu_required=True: the workstation has no CUDA "
                "device (AGENTS.md §2)"
            )

        ceiling = PLATFORM_MAX_TIMEOUT_MINUTES[self.platform]
        if self.timeout_minutes > ceiling:
            raise ValueError(
                f"timeout_minutes={self.timeout_minutes} exceeds the documented {self.platform} "
                f"ceiling of {ceiling} minutes"
            )
        return self

    # -- serialization ------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        """Return the spec as canonical, JSON-serializable data (redacted)."""
        return dict(redact_mapping(self.model_dump(mode="json")))

    def to_json_text(self) -> str:
        """Return canonical JSON: sorted keys, compact separators, trailing newline."""
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")) + "\n"

    def sha256(self) -> str:
        """Return ``sha256:<hex>`` of the canonical JSON form (the registry's ``config hash``)."""
        return spec_sha256(self)

    def is_paid(self) -> bool:
        """True when the platform bills the user."""
        return self.platform in PAID_PLATFORMS

    def requires_gpu(self) -> bool:
        """True when the run needs a CUDA device."""
        return self.gpu_required

    def notebook_path(self, out_dir: Path | str | None = None) -> Path:
        """Conventional notebook location for this run (``notebooks/generated/<run_id>.ipynb``).

        ``$SPECTRAQUANT_NOTEBOOK_DIR`` overrides the directory (used by CI and by tests so they never
        write into the repository).
        """
        if out_dir is not None:
            return Path(out_dir) / f"{self.run_id}.ipynb"
        override = os.environ.get("SPECTRAQUANT_NOTEBOOK_DIR")
        base = Path(override) if override else _default_notebook_dir()
        return base / f"{self.run_id}.ipynb"


def _default_notebook_dir() -> Path:
    from spectraquant.paths import repo_root

    return repo_root() / "notebooks" / "generated"


def spec_sha256(spec: RunSpec) -> str:
    """Return the ``sha256:<hex>`` digest of a spec's canonical JSON form."""
    digest = hashlib.sha256(spec.to_json_text().encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def save_spec(spec: RunSpec, path: Path | str) -> Path:
    """Write ``spec`` as canonical JSON to ``path`` and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(spec.to_json_text(), encoding="utf-8")
    return target


def load_spec(path: Path | str) -> RunSpec:
    """Load and validate a :class:`RunSpec` from a JSON file."""
    return RunSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def spec_from_config(
    config_path: str | Path,
    *,
    platform: str,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    repo_url: str | None = None,
    git_commit: str | None = None,
    allow_dirty: bool | None = None,
    install_spec: str | None = None,
    gpu_required: bool | None = None,
    timeout_minutes: int | None = None,
    max_cost_authorized_usd: float | None = None,
    expected_artifacts: list[ExpectedArtifact] | None = None,
    measurement_class_expected: int | None = None,
    seeds: list[int] | None = None,
) -> RunSpec:
    """Build a :class:`RunSpec` from a versioned experiment config plus the git state.

    Everything derivable from the repository is derived (git SHA, dataset identities and checksums,
    seeds, expected measurement class); everything host-specific is an explicit argument so the
    resulting spec is fully recorded and reproducible.

    Args:
        config_path: path to ``configs/experiment/<x>.yaml`` (resolved like the CLI does).
        platform: one of :data:`PLATFORMS`.
        run_id: explicit run id; defaults to ``"<experiment-name>-<utcstamp>-<sha12>"``.
        overrides: Hydra overrides recorded verbatim.
        repo_url: clone URL; defaults to ``$SPECTRAQUANT_REPO_URL`` or the local ``origin``.
        git_commit: resolved SHA; defaults to the checkout's ``HEAD``.
        allow_dirty: permit a dirty tree / unresolved SHA; defaults to the checkout's dirty state.
        install_spec: pinned environment materialisation command.
        gpu_required: whether the run needs CUDA; defaults to ``True`` for GPU platforms.
        timeout_minutes: ceiling; defaults to the platform's documented maximum.
        max_cost_authorized_usd: authorized spend (required for paid platforms).
        expected_artifacts: artifacts collection must verify.
        measurement_class_expected: declared class; defaults to the config's ``method`` class, or 1
            for a ``method=none`` harness config (which records no class in its manifest).
        seeds: master seeds; defaults to ``[config.experiment.seed]``.

    Returns:
        A validated :class:`RunSpec`.

    Raises:
        FileNotFoundError: the config does not exist.
        ValueError: the config or the derived spec violates a rule.
    """
    from spectraquant.cloud.remote import synthetic_split_checksum
    from spectraquant.config import load_experiment_config
    from spectraquant.reporting.gitinfo import git_info

    cfg = load_experiment_config(config_path, list(overrides or []))
    resolved_config_path = Path(str(cfg.config_path))
    recorded_config = _recordable_config_path(resolved_config_path)

    info = git_info()
    commit = git_commit if git_commit is not None else (info.commit or "")
    explicit_allow = allow_dirty is not None
    dirty = bool(allow_dirty) if explicit_allow else bool(info.dirty)
    if info.commit is None and not dirty:
        raise ValueError(
            "not inside a git checkout and allow_dirty is False: cannot pin git_commit "
            "(pass allow_dirty=True for a non-reproducible run)"
        )
    if not explicit_allow and info.dirty and commit == info.commit:
        raise ValueError(
            "the working tree is dirty: a cloud run must execute an exact commit "
            "(commit your changes, or pass allow_dirty=True deliberately)"
        )

    master_seed = seeds[0] if seeds else int(cfg.experiment.seed)
    data = cfg.data
    dataset_refs = [
        DatasetRef(
            name=str(data.name),
            revision=_data_revision(data),
            split="train",
            checksum=synthetic_split_checksum(data, seed=master_seed, split="train"),
        )
    ]

    platform_value = platform
    platform_max = PLATFORM_MAX_TIMEOUT_MINUTES.get(platform_value)
    if platform_max is None:
        raise ValueError(f"unknown platform {platform!r}; expected one of {', '.join(PLATFORMS)}")

    declared_class = cfg.method.measurement_class
    if measurement_class_expected is None:
        measurement_class_expected = 1 if declared_class is None else int(declared_class)
    if measurement_class_expected not in (1, 2, 3, 4):
        raise ValueError(
            f"measurement_class_expected={measurement_class_expected} is not in 1..4; classes "
            "4-GPU/5 are not claimable from this workstation (measurement-taxonomy.md §1)"
        )

    url = repo_url if repo_url is not None else _default_repo_url()
    install = install_spec if install_spec is not None else _default_install_spec()
    needs_gpu = gpu_required if gpu_required is not None else platform_value != "local_cpu"

    name = str(cfg.experiment.name)
    return RunSpec(
        run_id=run_id or _default_run_id(name, commit),
        experiment_config=recorded_config,
        overrides=list(overrides or []),
        platform=cast_platform(platform_value),
        repo_url=url,
        git_commit=commit,
        allow_dirty=dirty,
        install_spec=install,
        dataset_refs=dataset_refs,
        seeds=[master_seed],
        gpu_required=needs_gpu,
        timeout_minutes=timeout_minutes if timeout_minutes is not None else platform_max,
        max_cost_authorized_usd=max_cost_authorized_usd,
        expected_artifacts=list(expected_artifacts or []),
        measurement_class_expected=measurement_class_expected,
    )


#: Roles a plan dataset may carry, in the order used to pick the split a spec records.
_PLAN_SPLIT_BY_ROLE: tuple[tuple[str, str], ...] = (
    ("test_perplexity", "test"),
    ("development", "validation"),
    ("calibration", "train"),
    ("train", "train"),
    ("test_downstream", "test"),
)

#: Default repo-relative directory the plan runner writes its per-arm results into.
_PLAN_RUN_SUBDIR = "artifacts/runs"


def _plan_dataset_split(dataset: Any) -> str:
    """Pick the split a plan dataset contributes to the spec (first matching role wins)."""
    roles = set(dataset.roles)
    for role, split in _PLAN_SPLIT_BY_ROLE:
        if role in roles:
            return split
    raise ValueError(
        f"plan dataset {dataset.name!r} carries no role this slice can map to a split "
        f"(roles: {sorted(roles)})"
    )


def _plan_out_dir(out_dir: str | Path | None, plan_name: str) -> str:
    """Return the recorded output directory for a plan run (repo-relative unless absolute)."""
    if out_dir is None:
        return f"{_PLAN_RUN_SUBDIR}/{plan_name}-plan"
    return Path(out_dir).as_posix()


def plan_to_run_spec(
    plan: str | Path,
    *,
    platform: str,
    repo_url: str | None = None,
    git_commit: str | None = None,
    allow_dirty: bool | None = None,
    out_dir: str | Path | None = None,
) -> RunSpec:
    """Build a :class:`RunSpec` from a frozen cloud **plan** (``configs/{tier1,tier2,repro}/*.yaml``).

    A plan is the frozen definition of a remote experiment (``docs/research/preregistration.md``);
    a :class:`~spectraquant.experiment_plan.PlanConfig` is *not* a Hydra composition config, so
    :func:`spec_from_config` cannot consume it. This function is the missing bridge: it maps the
    plan's identity, model/dataset revisions, seed set, cost ceiling and authorised measurement
    classes into the frozen :class:`RunSpec`, and points the notebook's run cell at
    ``spectraquant run-plan`` through ``runner_command``.

    Mapping decisions (each is a place the plan is narrower than the spec, so the *plan* wins):

    * ``experiment_config`` records the plan path — it is the document the spec composes from, and
      the notebook's manifest cell resolves it as a plan (see ``design-cloud-adapter.md`` §11).
    * ``timeout_minutes`` = ``cost.platform_hours_max * 60`` and ``max_cost_authorized_usd`` =
      ``cost.max_cost_authorized_usd``; a plan may never exceed its own authorization envelope, and
      a free-tier-only plan may not target a paid platform.
    * ``gpu_required`` is ``True`` unless the plan's own substrate is ``local_cpu``: the plan's
      substrate is where it was authorised to run, and research training never runs locally
      (``AGENTS.md`` §2.3).
    * ``seeds`` records the plan's reported seed list (the *n* of its statistical plan).
    * ``measurement_class_expected`` is the **maximum** class the plan authorises, capped at 4
      (classes 4-GPU/5 are not claimable from this workstation).
    * ``dataset_refs`` records one entry per plan dataset, split-selected by role, pinned by the
      plan's immutable revision through :func:`dataset_declaration_digest`.
    * ``expected_artifacts`` are the run's manifest, its aggregated ``metrics.json`` and the
      *executed* notebook — the three things a collected plan run must contain.

    Args:
        plan: plan name or path (resolved like :func:`spectraquant.experiment_plan.plan_path`).
        platform: execution substrate, one of :data:`PLATFORMS`.
        repo_url: clone URL; defaults to ``$SPECTRAQUANT_REPO_URL`` or the local ``origin``.
        git_commit: resolved SHA; defaults to the checkout's ``HEAD``.
        allow_dirty: permit a dirty tree / unresolved SHA; defaults to the checkout's dirty state.
        out_dir: repo-relative directory the runner writes into; recorded in ``runner_command``.
            Defaults to ``artifacts/runs/<plan-name>-plan``.

    Returns:
        A validated :class:`RunSpec` whose ``runner_command`` runs ``spectraquant run-plan``.

    Raises:
        FileNotFoundError: the plan file does not exist.
        ValueError: the plan, the git state, or the platform contradicts a spec rule.
    """
    from spectraquant.experiment_plan import load_plan, plan_path
    from spectraquant.reporting.gitinfo import git_info

    plan_file = plan_path(plan)
    config = load_plan(plan_file)
    recorded_config = _recordable_config_path(plan_file)

    info = git_info()
    commit = git_commit if git_commit is not None else (info.commit or "")
    explicit_allow = allow_dirty is not None
    dirty = bool(allow_dirty) if explicit_allow else bool(info.dirty)
    if info.commit is None and not dirty:
        raise ValueError(
            "not inside a git checkout and allow_dirty is False: cannot pin git_commit "
            "(pass allow_dirty=True for a non-reproducible run)"
        )
    if not explicit_allow and info.dirty and commit == info.commit:
        raise ValueError(
            "the working tree is dirty: a cloud run must execute an exact commit "
            "(commit your changes, or pass allow_dirty=True deliberately)"
        )

    platform_value = cast_platform(platform)
    if config.cost.free_tier_only and platform_value in PAID_PLATFORMS:
        raise ValueError(
            f"plan {config.name!r} declares cost.free_tier_only=true and may not target the paid "
            f"platform {platform_value!r}; no paid resource without prior authorization "
            "(AGENTS.md §2b rule 8)"
        )

    timeout_minutes = round(config.cost.platform_hours_max * 60)
    if timeout_minutes <= 0:
        raise ValueError(
            f"plan {config.name!r} declares platform_hours_max={config.cost.platform_hours_max}, "
            "which is not a usable wall-clock ceiling"
        )

    authorised = [int(cls) for cls in config.measurement_classes]
    if not authorised:
        raise ValueError(f"plan {config.name!r} authorises no measurement class")
    measurement_class_expected = min(4, max(authorised))

    dataset_refs = []
    for dataset in config.datasets:
        split = _plan_dataset_split(dataset)
        dataset_refs.append(
            DatasetRef(
                name=dataset.name,
                revision=dataset.revision,
                split=split,
                checksum=dataset_declaration_digest(dataset.name, dataset.revision, split),
            )
        )

    resolved_out_dir = _plan_out_dir(out_dir, config.name)
    runner_command = f"spectraquant run-plan --plan {recorded_config} --out {resolved_out_dir}"

    expected = [
        ExpectedArtifact(name="run_manifest.json", min_bytes=2),
        ExpectedArtifact(name=f"{resolved_out_dir}/metrics.json", min_bytes=2),
        ExpectedArtifact(name=f"notebook/{config.name}-cloud.ipynb", min_bytes=64),
    ]

    return RunSpec(
        run_id=f"{config.name}-cloud",
        experiment_config=recorded_config,
        runner_command=runner_command,
        overrides=[],
        platform=platform_value,
        repo_url=repo_url if repo_url is not None else _default_repo_url(),
        git_commit=commit,
        allow_dirty=dirty,
        install_spec=_default_plan_install_spec(),
        dataset_refs=dataset_refs,
        seeds=[int(seed) for seed in config.seeds.reported],
        gpu_required=config.substrate != "local_cpu",
        timeout_minutes=timeout_minutes,
        max_cost_authorized_usd=float(config.cost.max_cost_authorized_usd),
        expected_artifacts=expected,
        measurement_class_expected=measurement_class_expected,
    )


def _default_plan_install_spec() -> str:
    """The pinned environment a plan run needs: the cloud adapter **and** the model stack."""
    return "uv sync --frozen --extra cloud --extra models"


def cast_platform(value: str) -> Platform:
    """Validate ``value`` and narrow it to :data:`Platform`.

    Raises:
        ValueError: ``value`` is not a supported platform.
    """
    if value not in PLATFORMS:
        raise ValueError(f"unknown platform {value!r}; expected one of {', '.join(PLATFORMS)}")
    return value  # type: ignore[return-value]


def _recordable_config_path(resolved: Path) -> str:
    """Return the config path as recorded in the spec: repo-relative when possible.

    A cloud host has the repository, not this workstation's absolute paths, so a resolved path under
    the repository root is recorded relative to it (``configs/experiment/<x>.yaml``) and resolves
    against the remote checkout. A path outside the repository is recorded verbatim and is
    host-specific.
    """
    from spectraquant.paths import repo_root

    try:
        return resolved.relative_to(repo_root()).as_posix()
    except ValueError:
        return str(resolved)


def _data_revision(data: Any) -> str:
    """Derive a stable, immutable revision string for a generated dataset config."""
    payload = data.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"config-sha256:{digest}"


def _default_repo_url() -> str:
    configured = os.environ.get("SPECTRAQUANT_REPO_URL")
    if configured:
        return configured
    from spectraquant.paths import repo_root
    from spectraquant.reporting.gitinfo import _run_git

    ok, out = _run_git(["remote", "get-url", "origin"], repo_root())
    if ok and out:
        return out
    return str(repo_root())


def _default_install_spec() -> str:
    """The pinned environment materialisation used when the caller does not override it."""
    return "uv sync --frozen --extra cloud"


def _default_run_id(name: str, commit: str) -> str:
    from spectraquant.reporting.manifests import build_run_id

    return build_run_id(name, git_commit=commit or None, suffix="cloud")
