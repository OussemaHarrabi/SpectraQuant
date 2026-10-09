"""Typed **cloud experiment plans** — the frozen Tier-1/Tier-2/ reproduction definitions.

These are not Hydra composition configs (that is :mod:`spectraquant.config`): a plan is a single
self-contained YAML document that fully specifies a *remote* experiment — model identity and
revision, dataset roles, seed set, compression grid, byte-budget ladder, method arms, substrate and
cost ceiling. The cloud notebook generator consumes a plan; the preregistration references plans as
the frozen values of its checklists, so the two cannot drift.

Why a separate schema instead of reusing ``ExperimentConfig``: a plan must express things the local
fixture config cannot (a pretrained model with an immutable revision, dataset revisions and split
roles, a *set* of seeds, a budget ladder, a cost ceiling). ``extra="forbid"`` is kept so a typo is an
error rather than a silently ignored field.

Usage::

    plan = load_plan("configs/tier1/smollm2_135m.yaml")
    plan.seeds
    list_plans()
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from spectraquant.paths import configs_dir

__all__ = [
    "PLAN_DIRS",
    "ArmSpec",
    "CompressionGrid",
    "CostCeiling",
    "DatasetRole",
    "ModelRef",
    "PlanConfig",
    "SeedPlan",
    "SubstrateName",
    "list_plans",
    "load_plan",
    "plan_path",
]

#: Directories (relative to ``configs/``) that may contain plans.
PLAN_DIRS: tuple[str, ...] = ("tier1", "tier2", "repro")

SubstrateName = Literal["colab", "kaggle", "colab_enterprise", "local_cpu"]


class _Strict(BaseModel):
    """Base: unknown keys are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRef(_Strict):
    """A model with an immutable revision and a recorded licence.

    ``revision`` is a Hugging Face revision hash or a local path; an empty revision is rejected
    because an unpinned model cannot be reproduced (``docs/research/model-dataset-licenses.md``).
    """

    id: str = Field(min_length=1)
    revision: str = Field(min_length=7)
    license: str = Field(min_length=1)
    parameters: int = Field(gt=0)
    role: str = Field(min_length=1)


class DatasetRole(_Strict):
    """One dataset revision bound to one or more protocol roles."""

    name: str = Field(min_length=1)
    revision: str = Field(min_length=7)
    license: str = Field(min_length=1)
    config: str | None = None
    roles: list[
        Literal["train", "calibration", "development", "test_perplexity", "test_downstream"]
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_roles(self) -> DatasetRole:
        if len(set(self.roles)) != len(self.roles):
            raise ValueError(f"duplicate roles for dataset {self.name!r}: {self.roles}")
        if "test_perplexity" in self.roles and not self.config:
            # A repository that exposes several configs cannot be loaded without one (observed on the
            # first real cloud run: "Config name is missing" from the datasets library, after the
            # platform had already spent minutes building the environment). Fail here, locally.
            raise ValueError(
                f"dataset {self.name!r} is used for the perplexity split but declares no config; "
                "pin the config name (for example config: wikitext-2-raw-v1)"
            )
        return self


class SeedPlan(_Strict):
    """Master seed plus the derived streams and the reported seed list.

    ``reported`` is the list actually run and reported; its length is the *n* of the statistical
    plan, so it must be at least the preregistered floor (5 for cheap cells, 3 for trainable ones).
    """

    master: int = Field(ge=0)
    derived: dict[str, int] = Field(default_factory=dict)
    reported: list[int] = Field(min_length=3)
    floor: int = Field(ge=3)

    @model_validator(mode="after")
    def _floor_respected(self) -> SeedPlan:
        if len(self.reported) < self.floor:
            raise ValueError(
                f"reported seed count {len(self.reported)} is below the preregistered floor "
                f"{self.floor}: reduce scope only through a preregistration amendment"
            )
        if len(set(self.reported)) != len(self.reported):
            raise ValueError(f"duplicate seeds: {self.reported}")
        return self


class CompressionGrid(_Strict):
    """The predeclared compression grid: bit widths, ranks, group sizes and the byte-budget ladder."""

    bits: list[int] = Field(min_length=1)
    ranks: list[int] = Field(min_length=1)
    group_sizes: list[int] = Field(min_length=1)
    budget_ladder_bytes: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> CompressionGrid:
        for field, values in (
            ("bits", self.bits),
            ("ranks", self.ranks),
            ("group_sizes", self.group_sizes),
            ("budget_ladder_bytes", self.budget_ladder_bytes),
        ):
            if list(values) != sorted(values):
                raise ValueError(f"{field} must be ascending: {values}")
            if len(set(values)) != len(values):
                raise ValueError(f"{field} has duplicates: {values}")
        return self


class ArmSpec(_Strict):
    """One method arm of the experiment matrix."""

    name: str = Field(min_length=1)
    kind: Literal[
        "fp16_reference",
        "ptq_uniform",
        "low_rank_only",
        "rank_then_quant",
        "quant_then_residual",
        "proxy_allocated",
        "proxy_allocated_regularized",
        "qlora",
        "loftq",
        "lr_qat",
        "spectraquant",
    ]
    trainable: bool
    #: The grid point this arm is bound to, when the plan binds one (``{"rank": 8, "bits": 4}``).
    #: A frozen plan that leaves its arms unbound is not self-contained: the point then lives only in
    #: the invocation, so the notebook's runner command becomes the sole record of what ran, and two
    #: arms can be given the same point by accident (observed: `--bits=4` made `ptq_uniform_8` and
    #: `ptq_uniform_4` identical). Binding here keeps the plan the single source of truth.
    point: dict[str, int] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def _point_keys_are_known(self) -> ArmSpec:
        unknown = set(self.point) - {"rank", "bits", "group_size"}
        if unknown:
            raise ValueError(
                f"arm {self.name!r} binds unknown grid point key(s) {sorted(unknown)}; "
                "allowed: bits, rank, group_size"
            )
        for key, value in self.point.items():
            if int(value) <= 0:
                raise ValueError(f"arm {self.name!r} binds {key}={value}, which is not positive")
        return self


class CostCeiling(_Strict):
    """The authorization envelope for a cloud run (AGENTS.md section 2b rule 8)."""

    platform_hours_max: float = Field(gt=0)
    max_cost_authorized_usd: float = Field(ge=0)
    free_tier_only: bool


class PlanConfig(_Strict):
    """A fully specified remote experiment.

    Attributes:
        name: plan identity; also the generated notebook's stem.
        substrate: where the plan runs (``local_cpu`` is refused for trainable plans).
        device: the torch device the runner uses. ``cpu`` is the default and the only device the
            pinned environment supports (``pyproject.toml`` pins the CPU torch index), so a plan that
            declares ``cuda`` also declares that it needs a CUDA torch build; ``gpu_required`` in the
            derived :class:`~spectraquant.cloud.spec.RunSpec` follows this field, not the tier.
        tier: scope-ladder tier this plan belongs to.
        models: one or more pinned models (a reproduction plan may compare two).
        datasets: dataset revisions with their protocol roles.
        seeds: the seed plan.
        grid: the compression grid.
        arms: the method matrix.
        harness: evaluation harness commit and the predeclared task list.
        cost: the authorization envelope.
        measurement_classes: classes the plan may produce, so a run cannot label a number with a
            class the plan never authorised.
    """

    name: str = Field(min_length=1)
    substrate: SubstrateName
    tier: int = Field(ge=0, le=5)
    device: Literal["cpu", "cuda"] = "cpu"
    models: list[ModelRef] = Field(min_length=1)
    datasets: list[DatasetRole] = Field(min_length=1)
    seeds: SeedPlan
    grid: CompressionGrid
    arms: list[ArmSpec] = Field(min_length=1)
    harness: dict[str, str | list[str]]
    cost: CostCeiling
    measurement_classes: list[int] = Field(min_length=1)
    notes: str = ""

    @model_validator(mode="after")
    def _consistency(self) -> PlanConfig:
        if self.substrate == "local_cpu" and any(arm.trainable for arm in self.arms):
            raise ValueError(
                "local_cpu may not host a trainable arm: research training runs on the cloud "
                "substrate (AGENTS.md sections 2.3 and 2b)"
            )
        if any(cls not in (1, 2, 3, 4, 5) for cls in self.measurement_classes):
            raise ValueError(f"invalid measurement class in {self.measurement_classes}")
        if not self.cost.free_tier_only and self.cost.max_cost_authorized_usd <= 0:
            raise ValueError(
                "a non-free-tier plan must state a positive max_cost_authorized_usd "
                "(no paid resource without prior authorization)"
            )
        if "commit" not in self.harness:
            raise ValueError("harness must pin a commit")
        return self


def plan_path(plan: str | Path) -> Path:
    """Resolve a plan name or path to an existing file under the repository ``configs`` tree.

    Raises:
        FileNotFoundError: no candidate exists; the message lists what was tried.
    """
    raw = Path(plan)
    candidates: list[Path] = []
    if raw.is_file():
        return raw.resolve()
    if raw.suffix in (".yaml", ".yml"):
        candidates.append(raw)
    else:
        candidates += [Path(f"{raw}.yaml"), Path(f"{raw}.yml")]
    for directory in PLAN_DIRS:
        for candidate in list(candidates):
            candidates.append(configs_dir() / directory / candidate.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"plan not found: {plan} (tried: {tried})")


def load_plan(plan: str | Path) -> PlanConfig:
    """Load and validate one cloud experiment plan.

    Raises:
        FileNotFoundError: the plan file does not exist.
        pydantic.ValidationError: the document violates the schema (unknown keys included).
    """
    path = plan_path(plan)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"plan must be a mapping, got {type(document).__name__}")
    return PlanConfig.model_validate(document)


def list_plans() -> Sequence[Path]:
    """Every committed plan file, sorted by path."""
    found: list[Path] = []
    for directory in PLAN_DIRS:
        base = configs_dir() / directory
        if base.is_dir():
            found.extend(sorted(base.glob("*.yaml")))
    return tuple(found)
