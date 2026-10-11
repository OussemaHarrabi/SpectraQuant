"""The pre-submission guards: the disk projection and the local CPU dry-run.

Both guards exist because two Tier-1 submissions burned ~13 h of GPU budget on defects a local check
could have found: one run filled the worker's disk and died at export, another surfaced a
serialization and a granularity defect only on the GPU. These tests exercise the *guards*, not the
science: the dry-run's runner is injected, so nothing here spawns a training subprocess or touches
the network.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from _cloud_fixtures import make_manifest, make_spec
from typer.testing import CliRunner

import spectraquant.cli.main as cli_main
from spectraquant.cli.main import app
from spectraquant.cloud.adapters.base import CommandResult
from spectraquant.cloud.dryrun import (
    TINY_CALIBRATION_DOCUMENTS,
    TINY_MAX_DOCUMENTS,
    TINY_MAX_TOKENS,
    TINY_SEQ_LEN,
    TINY_TRAINING,
    arm_binding_args,
    dry_run_plan,
    select_grid_point,
    tiny_plan_document,
    write_tiny_plan,
)
from spectraquant.cloud.plan_runner import resolve_arm_compression
from spectraquant.cloud.preflight import (
    DISK_ESTIMATE_LABEL,
    PLATFORM_DISK_HEADROOM_BYTES,
    WorkerDiskExceeded,
    assert_plan_fits,
    assert_spec_fits,
    checkpoints_per_seed,
    headroom_for_platform,
    project_plan_disk,
    projection_for_spec,
    recorded_plan_file,
)
from spectraquant.cloud.remote import RESULT_LINE_PREFIX
from spectraquant.cloud.spec import save_spec
from spectraquant.experiment_plan import load_plan

TIER1_PLAN = "configs/tier1/smollm2_135m.yaml"
SMOKE_CONFIG = "configs/experiment/smoke.yaml"

runner = CliRunner()


def _combined(result: Any) -> str:
    return str(getattr(result, "output", "")) + str(getattr(result, "stderr", "") or "")


def _plan_document(**overrides: Any) -> dict[str, Any]:
    """Return the frozen Tier-1 plan document with ``overrides`` applied."""
    document = yaml.safe_load(Path(TIER1_PLAN).read_text(encoding="utf-8"))
    document.update(overrides)
    return document


def _write_plan(tmp_path: Path, name: str = "plan.yaml", **overrides: Any) -> Path:
    """Write a plan document (the frozen Tier-1 plan unless overridden) and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(_plan_document(**overrides), sort_keys=False), encoding="utf-8")
    return path


def _training_schedule() -> dict[str, Any]:
    """Return the frozen Tier-1 plan's training schedule as a mutable mapping."""
    schedule = load_plan(TIER1_PLAN).training
    assert schedule is not None  # the plan declares a trainable arm, so the block must be pinned
    return dict(schedule.model_dump(mode="json"))


def _trainable_arms(count: int) -> list[dict[str, Any]]:
    """Return ``count`` distinct trainable arms (the M3 shape: several arms that all write weights)."""
    return [
        {"name": f"trained_{index}", "kind": "proxy_allocated_regularized", "trainable": True}
        for index in range(count)
    ]


# --------------------------------------------------------------------------------------
# preflight: the projection
# --------------------------------------------------------------------------------------
def test_projection_itemises_labelled_estimates_with_visible_arithmetic() -> None:
    projection = project_plan_disk(TIER1_PLAN, platform="colab")

    assert projection.terms, "the projection must itemise its terms"
    for term in projection.terms:
        assert term.label == DISK_ESTIMATE_LABEL
        assert term.bytes_estimate > 0, term.name
        assert term.basis.strip(), f"{term.name} must show the arithmetic behind its estimate"
    # The peak is exactly the sum of the terms the retention policy can have resident.
    assert projection.peak_bytes == sum(
        term.bytes_estimate for term in projection.terms if term.counted_in_peak
    )
    # The unpruned total is every term: it can never be smaller than the peak.
    assert projection.unpruned_total_bytes >= projection.peak_bytes
    assert projection.largest_term.counted_in_peak


def test_projection_itemises_per_arm_checkpoints_and_the_resident_set() -> None:
    plan = load_plan(TIER1_PLAN)
    projection = project_plan_disk(TIER1_PLAN, platform="colab")

    # "trained checkpoints per arm x arms" must be visible, per arm.
    per_arm = next(
        term for term in projection.terms if term.name == "checkpoints:spectraquant_regularized"
    )
    per_seed = per_arm.bytes_estimate // len(plan.seeds.reported)
    assert per_arm.bytes_estimate == per_seed * len(plan.seeds.reported)
    assert "(arm, seed)" in per_arm.basis
    assert "checkpoint_every=250" in per_arm.basis

    # Exactly one (arm, seed) set is resident: the runner prunes each arm's checkpoints (the term
    # the guard's decision rests on, so it must be explicit and no larger than the unpruned total).
    resident = next(term for term in projection.terms if term.name == "checkpoints_resident_peak")
    assert resident.counted_in_peak
    assert resident.bytes_estimate == per_seed
    assert not per_arm.counted_in_peak

    # The artifacts the run writes are itemised with their own arithmetic.
    artifacts = next(
        term for term in projection.terms if term.name == "run_artifacts:spectraquant_regularized"
    )
    assert "manifest" in artifacts.basis and "metrics.json" in artifacts.basis
    assert "log" in artifacts.basis
    assert {term.name for term in projection.terms} >= {
        "environment",
        "hub_model_snapshot:HuggingFaceTB/SmolLM2-135M",
        "hub_dataset_cache",
        "executed_notebook",
        "export_manifest_and_aggregate",
    }


def test_projection_of_a_plan_without_checkpointing_has_no_checkpoint_terms(tmp_path: Path) -> None:
    schedule = _training_schedule()
    schedule["checkpoint_every"] = 0
    plan_file = _write_plan(tmp_path, training=schedule)

    projection = project_plan_disk(plan_file, platform="colab")

    assert not [term for term in projection.terms if term.name.startswith("checkpoints")]
    assert checkpoints_per_seed(load_plan(plan_file), "proxy_allocated_regularized") == 0


def test_guard_accepts_the_frozen_tier1_plan_on_both_notebook_platforms() -> None:
    for platform in ("colab", "kaggle"):
        projection = assert_plan_fits(TIER1_PLAN, platform=platform)
        assert projection.platform == platform
        assert projection.headroom_bytes == PLATFORM_DISK_HEADROOM_BYTES[platform]
        assert projection.peak_bytes < projection.headroom_bytes


def test_guard_refuses_a_plan_that_does_not_fit_and_names_the_offending_term() -> None:
    limit = 1_000_000_000  # 1 GB: far below any arm's checkpoint set
    with pytest.raises(WorkerDiskExceeded) as excinfo:
        assert_plan_fits(TIER1_PLAN, platform="kaggle", headroom_bytes=limit)

    message = str(excinfo.value)
    expected = project_plan_disk(TIER1_PLAN, platform="kaggle", headroom_bytes=limit)
    assert expected.exceeds
    # The message names the projected size, the limit and the largest term.
    assert f"{expected.peak_bytes:,} B" in message
    assert f"{limit:,} B" in message
    assert expected.largest_term.name in message
    assert "itemised" in message.lower()


def test_many_trainable_arms_exceed_even_the_pruned_peak(tmp_path: Path) -> None:
    schedule = _training_schedule()
    schedule.update({"steps": 4, "warmup_steps": 1, "checkpoint_every": 1})
    plan_file = _write_plan(
        tmp_path, name="many-arms.yaml", arms=_trainable_arms(6), training=schedule
    )

    with pytest.raises(WorkerDiskExceeded) as excinfo:
        assert_plan_fits(plan_file, platform="kaggle", headroom_bytes=2_000_000_000)

    assert "checkpoints_resident_peak" in str(excinfo.value)


def test_six_trainable_arms_fit_at_the_peak_but_their_unpruned_total_does_not(
    tmp_path: Path,
) -> None:
    schedule = _training_schedule()
    schedule.update({"steps": 4, "warmup_steps": 1, "checkpoint_every": 1})
    plan_file = _write_plan(
        tmp_path, name="many-arms.yaml", arms=_trainable_arms(6), training=schedule
    )

    projection = project_plan_disk(plan_file, platform="kaggle")
    # The M3 shape, in miniature: six trained arms x five seeds of checkpoints do *not* fit a 20 GB
    # worker (this is the projection of the run that died at export)...
    assert projection.unpruned_total_bytes > PLATFORM_DISK_HEADROOM_BYTES["kaggle"]
    assert projection.unpruned_exceeds
    # ... yet the run is collectable, because the runner prunes each (arm, seed) checkpoint set when
    # that arm finishes. The guard therefore decides on the peak and *reports* the unpruned figure.
    assert projection.peak_bytes < PLATFORM_DISK_HEADROOM_BYTES["kaggle"]
    assert not projection.exceeds
    assert projection.to_json()["unpruned_exceeds"] is True


def test_the_decision_is_the_peak_and_one_byte_below_it_is_refused(tmp_path: Path) -> None:
    schedule = _training_schedule()
    schedule.update({"steps": 4, "warmup_steps": 1, "checkpoint_every": 1})
    plan_file = _write_plan(
        tmp_path, name="many-arms.yaml", arms=_trainable_arms(6), training=schedule
    )

    projection = project_plan_disk(plan_file, platform="kaggle")
    strict = project_plan_disk(plan_file, platform="kaggle", headroom_bytes=projection.peak_bytes)
    assert not strict.exceeds
    with pytest.raises(WorkerDiskExceeded) as excinfo:
        assert_plan_fits(plan_file, platform="kaggle", headroom_bytes=projection.peak_bytes - 1)
    assert "checkpoints_resident_peak" in str(excinfo.value)


def test_headroom_comes_from_the_documented_run_book_figure() -> None:
    limit, source = headroom_for_platform("kaggle")

    assert limit == 20_000_000_000
    assert "run book" in source


def test_an_undocumented_platform_requires_an_explicit_headroom() -> None:
    with pytest.raises(ValueError, match="not documented"):
        headroom_for_platform("colab_enterprise")

    limit, source = headroom_for_platform("colab_enterprise", override=5_000_000)
    assert limit == 5_000_000
    assert "override" in source


def test_local_platform_headroom_is_measured_not_invented(tmp_path: Path) -> None:
    limit, source = headroom_for_platform("local_cpu", root=tmp_path)

    assert limit > 0
    assert "measured free space" in source


def test_seeds_outside_the_plan_are_refused() -> None:
    with pytest.raises(ValueError, match="not in plan"):
        project_plan_disk(TIER1_PLAN, platform="colab", seeds=[99])


def test_projection_for_a_spec_reads_the_plan_it_records(tmp_path: Path) -> None:
    spec = make_spec(
        run_id="tier1_smollm2_135m-cloud",
        experiment_config=TIER1_PLAN,
        platform="colab",
        seeds=[0, 1],
    )

    assert recorded_plan_file(spec) is not None
    projection = projection_for_spec(spec, root=Path.cwd())
    assert projection is not None
    assert projection.plan_name == "tier1_smollm2_135m"
    assert projection.seeds == (0, 1)
    assert projection.peak_bytes < projection.headroom_bytes

    composition = make_spec(run_id="smoke-cloud", experiment_config=SMOKE_CONFIG, platform="colab")
    assert recorded_plan_file(composition) is None
    assert projection_for_spec(composition, root=Path.cwd()) is None
    assert assert_spec_fits(composition, root=Path.cwd()) is None


def test_assert_spec_fits_refuses_an_oversized_plan_spec() -> None:
    spec = make_spec(
        run_id="tier1_smollm2_135m-cloud",
        experiment_config=TIER1_PLAN,
        platform="colab",
        seeds=[0, 1],
    )

    with pytest.raises(WorkerDiskExceeded) as excinfo:
        assert_spec_fits(spec, root=Path.cwd(), headroom_bytes=1_000)
    assert "tier1_smollm2_135m" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# dry-run: tiny plan derivation and per-arm bindings
# --------------------------------------------------------------------------------------
def test_tiny_plan_replaces_the_schedule_and_stays_a_valid_plan(tmp_path: Path) -> None:
    plan = load_plan(TIER1_PLAN)

    document = tiny_plan_document(plan, rank=8, bits=4)
    assert document["training"]["steps"] == TINY_TRAINING["steps"]
    assert document["training"]["checkpoint_every"] == TINY_TRAINING["checkpoint_every"]
    assert document["training"]["corpus_documents"] == TINY_TRAINING["corpus_documents"]
    # Everything that identifies the plan is kept: the arms, the model and dataset pins.
    assert [arm["name"] for arm in document["arms"]] == [arm.name for arm in plan.arms]
    assert document["models"] == plan.model_dump(mode="json")["models"]
    assert document["datasets"] == plan.model_dump(mode="json")["datasets"]
    assert "[dry-run]" in document["notes"]

    tiny = write_tiny_plan(plan, tmp_path, rank=8, bits=4)
    assert tiny.is_file()
    reloaded = load_plan(tiny)  # the copy must pass the plan schema
    assert reloaded.training is not None
    assert reloaded.training.steps == TINY_TRAINING["steps"]


def test_the_narrowed_grid_keeps_every_arm_bindable(tmp_path: Path) -> None:
    """The tiny grid may drop candidate space, but never a point an arm must resolve to."""
    plan = load_plan(TIER1_PLAN)
    tiny = load_plan(write_tiny_plan(plan, tmp_path, rank=8, bits=4))

    # The narrowing is real (the frozen grid is much wider)...
    assert tiny.grid.ranks == [8]
    assert tiny.grid.bits == [4, 8]  # 8 and 4 are named by the ptq arms; 4 is the dry-run's point
    assert plan.grid.ranks == [2, 4, 8, 16, 32]
    # ... and the budget ladder and group sizes are untouched: the allocated arms still solve against
    # a declared rung.
    assert tiny.grid.budget_ladder_bytes == plan.grid.budget_ladder_bytes
    assert tiny.grid.group_sizes == plan.grid.group_sizes

    # Every declared arm resolves against the narrowed grid with the dry-run's own bindings.
    for arm in tiny.arms:
        bindings = arm_binding_args(arm, rank=8, bits=4, budget_bytes=212336640)
        kwargs: dict[str, Any] = {}
        if "--rank" in bindings:
            kwargs["rank"] = 8
        if "--bits" in bindings:
            kwargs["bits"] = 4
        compression = resolve_arm_compression(
            tiny, arm, granularity="per_group", group_size=128, **kwargs
        )
        if arm.kind == "proxy_allocated" or arm.kind == "proxy_allocated_regularized":
            assert compression.bits is None and compression.rank is None
        else:
            assert compression.granularity == "per_group"


def test_grid_point_selection_takes_the_closest_declared_value() -> None:
    assert select_grid_point([2, 4, 8, 16, 32], 8) == 8
    assert select_grid_point([2, 6, 20], 8) == 6  # ties and gaps fall to the smaller value
    assert select_grid_point([12, 20], 8) == 12
    with pytest.raises(ValueError, match="empty grid"):
        select_grid_point([], 8)


def test_arm_bindings_match_the_runners_own_vocabulary() -> None:
    plan = load_plan(TIER1_PLAN)
    arms = {arm.name: arm for arm in plan.arms}

    allocated = plan.arms[0]
    allocated = arms["proxy_allocated"]
    assert arm_binding_args(allocated, rank=8, bits=4, budget_bytes=1_000) == [
        "--budget-bytes",
        "1_000".replace("_", ""),
    ]
    # A rank-and-bit arm gets both, and an fp16 reference gets neither.
    assert arm_binding_args(arms["quant_then_residual"], rank=8, bits=4, budget_bytes=1) == [
        "--rank",
        "8",
        "--bits",
        "4",
    ]
    assert arm_binding_args(arms["fp16_reference"], rank=8, bits=4, budget_bytes=1) == []
    # `ptq_uniform_8` names its own width, so no flag may override it.
    assert arm_binding_args(arms["ptq_uniform_8"], rank=8, bits=4, budget_bytes=1) == []
    # The plan binds `low_rank_only` to rank 8 through its own `point`, but the dry-run still binds
    # it explicitly: the runner validates the value against the plan's grid either way.
    assert arm_binding_args(arms["low_rank_only"], rank=8, bits=4, budget_bytes=1) == [
        "--rank",
        "8",
        "--bits",
        "4",
    ]


# --------------------------------------------------------------------------------------
# dry-run: execution with an injected runner
# --------------------------------------------------------------------------------------
def _value_after(argv: Sequence[str], flag: str) -> str:
    return str(argv[list(argv).index(flag) + 1])


def _fake_runner(
    *,
    failing: Sequence[str] = (),
    silent: Sequence[str] = (),
    mismatched: Sequence[str] = (),
    recorded: list[list[str]] | None = None,
) -> Any:
    """Return a runner that writes a valid (or deliberately broken) bundle per arm.

    Args:
        failing: arms whose command exits non-zero.
        silent: arms that exit zero but write no manifest (a truncated export).
        mismatched: arms whose ``metrics.json`` disagrees with the manifest.
        recorded: list collecting every argv, for command-shape assertions.
    """

    def run(argv: Sequence[str], timeout: float) -> CommandResult:
        del timeout
        if recorded is not None:
            recorded.append(list(argv))
        arm = _value_after(argv, "--arms")
        out_dir = Path(_value_after(argv, "--out"))
        seed = _value_after(argv, "--seed")
        target = out_dir / arm / f"seed-{seed}"
        target.mkdir(parents=True, exist_ok=True)
        if arm in failing:
            return CommandResult(argv=list(argv), returncode=1, stderr=f"{arm} exploded")
        if arm in silent:
            return CommandResult(argv=list(argv), returncode=0, stdout="nothing written")
        document = make_manifest(
            make_spec(run_id=f"{arm}-seed{seed}"), metrics={"perplexity": 12.5}
        )
        manifest_path = target / "run_manifest.json"
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
        metrics_path = target / "metrics.json"
        bundle = {"perplexity": 1.0} if arm in mismatched else document["metrics"]
        metrics_path.write_text(json.dumps(bundle), encoding="utf-8")
        payload = {
            "status": "success",
            "arms_run": [
                {
                    "arm": arm,
                    "seed": int(seed),
                    "manifest": str(manifest_path),
                    "metrics": str(metrics_path),
                    "perplexity": 12.5,
                }
            ],
        }
        return CommandResult(
            argv=list(argv),
            returncode=0,
            stdout=RESULT_LINE_PREFIX + json.dumps(payload, sort_keys=True) + "\n",
        )

    return run


def test_dry_run_passes_every_arm_and_asserts_the_bundle(tmp_path: Path) -> None:
    recorded: list[list[str]] = []
    report = dry_run_plan(
        TIER1_PLAN,
        work_dir=tmp_path / "work",
        runner=_fake_runner(recorded=recorded),
        progress=lambda message: None,
    )

    assert report.ok
    assert [outcome.status for outcome in report.outcomes] == ["passed"] * len(
        load_plan(TIER1_PLAN).arms
    )
    assert all(outcome.manifest and outcome.metrics for outcome in report.outcomes)
    assert "metrics bundle matches" in report.outcomes[0].detail
    assert len(recorded) == len(report.outcomes)
    # The tiny plan the report names is the one the arms were told to run.
    for argv in recorded:
        assert _value_after(argv, "--plan") == report.tiny_plan
        assert "--device" in argv and _value_after(argv, "--device") == "cpu"
        assert _value_after(argv, "--max-tokens") == str(TINY_MAX_TOKENS)
        assert _value_after(argv, "--max-documents") == str(TINY_MAX_DOCUMENTS)
        assert _value_after(argv, "--calibration-documents") == str(TINY_CALIBRATION_DOCUMENTS)
        assert _value_after(argv, "--seq-len") == str(TINY_SEQ_LEN)
    # The allocated arm is bound by a budget and never by a rank/bit width (the runner refuses that).
    allocated_argv = next(argv for argv in recorded if "proxy_allocated" in argv)
    assert "--budget-bytes" in allocated_argv
    assert "--rank" not in allocated_argv and "--bits" not in allocated_argv


def test_dry_run_fails_the_report_when_one_arm_exits_non_zero(tmp_path: Path) -> None:
    report = dry_run_plan(
        TIER1_PLAN,
        work_dir=tmp_path / "work",
        runner=_fake_runner(failing=["rank_then_quant"]),
        progress=lambda message: None,
    )

    assert not report.ok
    assert [outcome.arm for outcome in report.failures] == ["rank_then_quant"]
    assert "exited 1" in report.failures[0].detail
    assert "rank_then_quant exploded" in report.failures[0].detail
    assert len(report.passed) == len(load_plan(TIER1_PLAN).arms) - 1


def test_dry_run_fails_an_arm_that_writes_no_manifest(tmp_path: Path) -> None:
    report = dry_run_plan(
        TIER1_PLAN,
        work_dir=tmp_path / "work",
        runner=_fake_runner(silent=["fp16_reference"]),
        progress=lambda message: None,
    )

    assert not report.ok
    assert [outcome.arm for outcome in report.failures] == ["fp16_reference"]
    assert "bundle check failed" in report.failures[0].detail


def test_dry_run_fails_an_arm_whose_metrics_disagree_with_its_manifest(tmp_path: Path) -> None:
    report = dry_run_plan(
        TIER1_PLAN,
        work_dir=tmp_path / "work",
        runner=_fake_runner(mismatched=["ptq_uniform_4"]),
        progress=lambda message: None,
    )

    assert not report.ok
    assert [outcome.arm for outcome in report.failures] == ["ptq_uniform_4"]
    assert "does not match the manifest" in report.failures[0].detail


def test_dry_run_reports_an_unimplemented_arm_as_skipped_and_not_a_pass(tmp_path: Path) -> None:
    plan_file = _write_plan(
        tmp_path,
        name="unimplemented.yaml",
        arms=[
            {"name": "fp16_reference", "kind": "fp16_reference", "trainable": False},
            {"name": "qlora_arm", "kind": "qlora", "trainable": False},
        ],
        training=None,
        regularizer=None,
    )

    report = dry_run_plan(
        plan_file, work_dir=tmp_path / "work", runner=_fake_runner(), progress=lambda message: None
    )

    assert not report.ok
    assert [outcome.arm for outcome in report.skipped] == ["qlora_arm"]
    assert "M5" in report.skipped[0].detail
    assert [outcome.arm for outcome in report.passed] == ["fp16_reference"]


def test_dry_run_rejects_arguments_that_cannot_belong(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be flags"):
        dry_run_plan(TIER1_PLAN, runner_args=("rank 8",), progress=lambda message: None)
    with pytest.raises(ValueError, match="not a member of plan"):
        dry_run_plan(TIER1_PLAN, group_size=7, progress=lambda message: None)
    with pytest.raises(ValueError, match="requires group_size"):
        dry_run_plan(
            TIER1_PLAN, granularity="per_group", group_size=None, progress=lambda message: None
        )
    with pytest.raises(ValueError, match="not in plan"):
        dry_run_plan(TIER1_PLAN, seed=99, progress=lambda message: None)


def test_dry_run_report_serializes_per_arm_outcomes(tmp_path: Path) -> None:
    report = dry_run_plan(
        TIER1_PLAN,
        work_dir=tmp_path / "work",
        runner=_fake_runner(failing=["low_rank_only"]),
        progress=lambda message: None,
    )

    payload = report.to_json()
    assert payload["ok"] is False
    assert payload["plan_name"] == "tier1_smollm2_135m"
    assert payload["device"] == "cpu"
    assert len(payload["outcomes"]) == len(load_plan(TIER1_PLAN).arms)
    statuses = {entry["arm"]: entry["status"] for entry in payload["outcomes"]}
    assert statuses["low_rank_only"] == "failed"
    assert statuses["fp16_reference"] == "passed"
    # The report records what the dry-run bound and how, so a pass is reproducible.
    assert payload["settings"]["rank"] == 8
    assert payload["settings"]["bits"] == 4
    assert payload["settings"]["granularity"] == "per_group"
    assert payload["settings"]["tiny_training"] == TINY_TRAINING
    # A failed arm carries its own output, so the failure can be diagnosed without re-running it.
    failed = next(entry for entry in payload["outcomes"] if entry["status"] == "failed")
    assert "low_rank_only exploded" in failed["output_tail"]
    assert "low_rank_only exploded" in report.summary()


# --------------------------------------------------------------------------------------
# the CLI surface: dry-run / preflight / submit ordering
# --------------------------------------------------------------------------------------
def test_cli_dry_run_exits_zero_when_every_arm_passes(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli_main,
        "dry_run_plan",
        lambda *args, **kwargs: dry_run_plan(
            TIER1_PLAN, work_dir=tmp_path / "work", runner=_fake_runner(), progress=lambda m: None
        ),
    )

    result = runner.invoke(app, ["cloud", "dry-run", "--plan", TIER1_PLAN])

    assert result.exit_code == 0, _combined(result)
    assert "PASSED" in _combined(result)


def test_cli_dry_run_exits_non_zero_when_an_arm_is_broken(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli_main,
        "dry_run_plan",
        lambda *args, **kwargs: dry_run_plan(
            TIER1_PLAN,
            work_dir=tmp_path / "work",
            runner=_fake_runner(failing=["spectraquant_regularized"]),
            progress=lambda m: None,
        ),
    )

    result = runner.invoke(app, ["cloud", "dry-run", "--plan", TIER1_PLAN])

    assert result.exit_code == 1
    combined = _combined(result)
    assert "FAILED" in combined
    assert "spectraquant_regularized" in combined


def test_cli_dry_run_json_reports_every_outcome(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli_main,
        "dry_run_plan",
        lambda *args, **kwargs: dry_run_plan(
            TIER1_PLAN,
            work_dir=tmp_path / "work",
            runner=_fake_runner(failing=["ptq_uniform_4"]),
            progress=lambda m: None,
        ),
    )

    result = runner.invoke(app, ["cloud", "dry-run", "--plan", TIER1_PLAN, "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    statuses = {entry["arm"]: entry["status"] for entry in payload["outcomes"]}
    assert statuses["ptq_uniform_4"] == "failed"
    assert statuses["fp16_reference"] == "passed"
    assert payload["settings"]["rank"] == 8


def test_cli_preflight_prints_the_projection_and_runs_the_checks(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli_main,
        "dry_run_plan",
        lambda *args, **kwargs: dry_run_plan(
            TIER1_PLAN,
            work_dir=kwargs.get("work_dir"),
            runner=_fake_runner(),
            progress=lambda m: None,
        ),
    )

    result = runner.invoke(
        app, ["cloud", "preflight", "--plan", TIER1_PLAN, "--platform", "kaggle"]
    )

    combined = _combined(result)
    assert result.exit_code == 0, combined
    assert "pre-flight disk projection" in combined
    assert "preflight passed" in combined


def test_cli_preflight_refuses_before_dry_running(monkeypatch: Any) -> None:
    called: list[str] = []

    def _never(*args: Any, **kwargs: Any) -> Any:
        called.append("dry-run")
        raise AssertionError("the dry-run must not run for a plan that cannot fit")

    monkeypatch.setattr(cli_main, "dry_run_plan", _never)

    result = runner.invoke(
        app,
        [
            "cloud",
            "preflight",
            "--plan",
            TIER1_PLAN,
            "--platform",
            "kaggle",
            "--headroom-bytes",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert called == []
    assert "checkpoints_resident_peak" in _combined(result)


def test_cli_submit_refuses_an_oversized_plan_before_any_remote_call(
    tmp_path: Path, monkeypatch: Any
) -> None:
    spec = make_spec(
        run_id="tier1_smollm2_135m-cloud",
        experiment_config=TIER1_PLAN,
        platform="colab",
        seeds=[0, 1],
    )
    spec_path = save_spec(spec, tmp_path / "spec.json")
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    adapter_calls: list[str] = []
    monkeypatch.setattr(cli_main, "adapter_for", lambda platform: adapter_calls.append(platform))

    result = runner.invoke(
        app,
        [
            "cloud",
            "submit",
            "--spec",
            str(spec_path),
            "--notebook",
            str(notebook),
            "--headroom-bytes",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "pre-flight refused" in _combined(result)
    assert "checkpoints_resident_peak" in _combined(result)
    assert adapter_calls == [], "the adapter must never be reached when the projection does not fit"


def test_cli_submit_prints_the_projection_before_submitting(
    tmp_path: Path, monkeypatch: Any
) -> None:
    spec = make_spec(
        run_id="tier1_smollm2_135m-cloud",
        experiment_config=TIER1_PLAN,
        platform="colab",
        seeds=[0, 1],
    )
    spec_path = save_spec(spec, tmp_path / "spec.json")
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    class _Stub:
        def submit(self, spec: Any, notebook_path: str) -> str:
            del spec, notebook_path
            return "remote-1"

    monkeypatch.setattr(cli_main, "adapter_for", lambda platform: _Stub())

    result = runner.invoke(
        app, ["cloud", "submit", "--spec", str(spec_path), "--notebook", str(notebook)]
    )

    combined = _combined(result)
    assert result.exit_code == 0, combined
    assert "pre-flight disk projection" in combined
    assert "submitted: tier1_smollm2_135m-cloud" in combined
    assert combined.index("pre-flight disk projection") < combined.index("submitted:")
