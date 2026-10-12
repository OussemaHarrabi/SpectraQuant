"""The plan runner's wall-clock deadline: a run that runs out of time stops cleanly and says so.

Every test here drives the deadline through a **scripted clock** (``plan_runner._monotonic``): the
budget is proved on a CPU box without the result depending on how fast the machine is, and without a
single ``sleep``. Everything else is the real runner, on the offline fixture model/corpus
(``_plan_fixtures.plan_runner_env``) and the real frozen plan.

The properties under test are the ones that decide whether a capped GPU run is claimable at all:

* it stops **between** whole units of work (never mid-write), before it has spent the session limit;
* every completed arm's artifacts are intact and schema-valid;
* the run-level manifest is schema-valid, carries the partial marker, the completed/not-completed
  lists, the stop reason and the elapsed wall-clock - so a partial run is never read as a complete
  one;
* a run that finishes inside its budget is marked complete, not partial;
* the notebook's manifest stage turns a partial runner into a ``partial`` cloud manifest, because a
  cleanly stopped run still exits zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _plan_fixtures import plan_runner_env

from spectraquant.cloud import plan_runner
from spectraquant.cloud.notebook_stages import run_stage
from spectraquant.cloud.plan_runner import run_plan
from spectraquant.cloud.remote import RESULT_LINE_PREFIX, parse_result_line
from spectraquant.experiment_plan import load_plan
from spectraquant.reporting.manifests import (
    ManifestValidationError,
    validate_manifest_dict,
    validate_manifest_file,
)

REPRO_PLAN = "configs/repro/lr_qat_smollm2_135m.yaml"

#: The repro plan's arms, in declaration order.
ARMS: tuple[str, ...] = ("fp16_reference", "lr_qat", "loftq", "rank_then_quant")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every arm offline and deterministically (fake model, tokenizer and corpus)."""
    plan_runner_env(monkeypatch)


class _ScriptedClock:
    """A monotonic clock the test drives, so the budget never depends on machine speed."""

    def __init__(self, *, tick: float = 0.0) -> None:
        self._now = 0.0
        self._tick = float(tick)

    def __call__(self) -> float:
        self._now += self._tick
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)


def _install_clock(monkeypatch: pytest.MonkeyPatch, *, tick: float = 0.0) -> _ScriptedClock:
    """Replace the runner's clock (the module-level seam) with a scripted one."""
    clock = _ScriptedClock(tick=tick)
    monkeypatch.setattr(plan_runner, "_monotonic", clock)
    return clock


def _advance_after_the_first_arm(monkeypatch: pytest.MonkeyPatch, clock: _ScriptedClock) -> None:
    """Let the first arm's compression finish, then spend the whole budget.

    The hook is the arm's own compression step: it means "arm 1 fits in the budget, arm 2 does not",
    which is exactly the boundary the between-arms check has to catch - and it stays true no matter
    how many times the runner reads the clock elsewhere.
    """
    original = plan_runner.apply_arm
    calls = {"count": 0}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        applied = original(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 1:
            clock.advance(1_000.0)
        return applied

    monkeypatch.setattr(plan_runner, "apply_arm", wrapper)


def _plan_steps() -> int:
    """The repro plan's pinned schedule length, as an interrupted arm reports it."""
    schedule = load_plan(REPRO_PLAN).training
    assert schedule is not None
    return int(schedule.steps)


def _advance_when_training_starts(
    monkeypatch: pytest.MonkeyPatch, clock: _ScriptedClock, *, seconds: float
) -> None:
    """Spend the budget as the arm's training loop starts, so the loop's own check fires."""
    from spectraquant.training import loop as training_loop

    original = training_loop.train_language_model
    calls = {"count": 0}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            clock.advance(seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(training_loop, "train_language_model", wrapper)


# --------------------------------------------------------------------------------------
# The deadline stops the run and the record says so
# --------------------------------------------------------------------------------------
def test_the_deadline_stops_a_multi_arm_run_and_marks_it_partial(
    tmp_path: Path,
    offline: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Acceptance: the run stops early, exports the finished arm, and records the partial marker."""
    clock = _install_clock(monkeypatch)
    _advance_after_the_first_arm(monkeypatch, clock)
    messages: list[str] = []

    result = run_plan(
        REPRO_PLAN,
        out_dir=tmp_path,
        max_wall_seconds=5.0,
        progress=messages.append,
    )

    # (a) it stopped before completing all the arms - and before starting the next one.
    assert result.partial is True
    assert [run.arm for run in result.runs] == ["fp16_reference"]
    assert [entry["cell"] for entry in result.not_completed] == [
        "lr_qat/seed-0",
        "loftq/seed-0",
        "rank_then_quant/seed-0",
    ]
    assert {entry["arm"] for entry in result.not_completed} == set(ARMS[1:])
    for entry in result.not_completed:
        assert entry["steps_completed"] is None, "a cell that never started has no step count"
        assert "budget" in str(entry["reason"])

    # (b) the run-level manifest is schema-valid and carries the partial marker + both lists.
    assert result.run_manifest_path is not None
    manifest = validate_manifest_file(result.run_manifest_path)
    assert manifest["status"] == "partial"
    assert manifest["failure_reason"], "a partial run must name what stopped it"
    metrics = manifest["metrics"]
    assert metrics["run.level"] == "plan"
    assert metrics["run.partial"] is True
    assert metrics["run.status"] == "partial"
    assert metrics["run.max_wall_seconds"] == pytest.approx(5.0)
    assert metrics["run.arms_completed"] == ["fp16_reference"]
    assert metrics["run.arms_not_completed"] == ["loftq", "lr_qat", "rank_then_quant"]
    assert metrics["run.cells_completed"] == ["fp16_reference/seed-0"]
    assert metrics["run.cells_not_completed"] == [
        "lr_qat/seed-0",
        "loftq/seed-0",
        "rank_then_quant/seed-0",
    ]
    assert len(metrics["run.not_completed_reasons"]) == 3
    assert metrics["run.stop_reason"] == manifest["failure_reason"]

    # A partial manifest is *enforced* to carry its reason: without it the document is invalid, so a
    # partial run cannot be laundered into a complete one by deleting the marker.
    broken = dict(manifest)
    broken["failure_reason"] = None
    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(broken)

    # (c) every completed arm's artifacts are intact and readable.
    completed = result.runs[0]
    assert completed.manifest_path.is_file() and completed.metrics_path.is_file()
    arm_manifest = validate_manifest_file(completed.manifest_path)
    assert arm_manifest["status"] == "success"
    payload = json.loads(completed.metrics_path.read_text(encoding="utf-8"))
    assert payload["plan.arm"] == "fp16_reference"
    assert payload["perplexity"] == completed.perplexity

    # ... and no artifact of a cell the budget stopped exists (nothing half-written, nothing faked).
    for arm in ARMS[1:]:
        assert not (tmp_path / arm / "seed-0" / "run_manifest.json").exists()
        assert not (tmp_path / arm / "seed-0" / "metrics.json").exists()

    # (d) the elapsed wall-clock is recorded, in the manifest and on the result.
    assert result.elapsed_wall_seconds == pytest.approx(1_000.0)
    assert metrics["run.elapsed_wall_seconds"] == pytest.approx(1_000.0)
    assert manifest["training"]["wall_time_s"] == pytest.approx(1_000.0)

    # (5) the log distinguishes a deadline stop from a finished run, and the result line - the channel
    # the notebook's manifest stage reads - carries the partial marker.
    assert any("DEADLINE REACHED" in message for message in messages)
    line = parse_result_line(capsys.readouterr().out)
    assert line is not None
    assert line["status"] == "partial" and line["partial"] is True
    assert line["cells_not_completed"] == [
        "lr_qat/seed-0",
        "loftq/seed-0",
        "rank_then_quant/seed-0",
    ]
    assert str(line["stop_reason"]) == result.stop_reason


def test_a_run_inside_its_budget_is_marked_complete(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance: finishing inside the budget is ``complete``, never ``partial``."""
    _install_clock(monkeypatch, tick=0.001)
    messages: list[str] = []

    result = run_plan(
        REPRO_PLAN,
        out_dir=tmp_path,
        max_wall_seconds=600.0,
        progress=messages.append,
    )

    assert result.partial is False
    assert result.not_completed == ()
    assert result.stop_reason is None
    assert [run.arm for run in result.runs] == list(ARMS)
    assert result.run_manifest_path is not None
    manifest = validate_manifest_file(result.run_manifest_path)
    assert manifest["status"] == "success"
    assert manifest["failure_reason"] is None
    metrics = manifest["metrics"]
    assert metrics["run.partial"] is False
    assert metrics["run.status"] == "complete"
    assert metrics["run.arms_not_completed"] == []
    assert metrics["run.cells_not_completed"] == []
    assert metrics["run.stop_reason"] is None
    assert metrics["run.arms_completed"] == sorted(ARMS)
    # The elapsed wall-clock is recorded even when nothing was cut short, and it is under the budget.
    assert 0.0 < result.elapsed_wall_seconds < 600.0
    assert metrics["run.elapsed_wall_seconds"] == pytest.approx(result.elapsed_wall_seconds)
    assert not any("DEADLINE REACHED" in message for message in messages)
    assert any("run complete" in message for message in messages)


def test_a_deadline_inside_an_arm_stops_at_a_checkpoint_boundary(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget is also polled inside an arm, and a cut-short arm is not a measurement."""
    clock = _install_clock(monkeypatch)
    _advance_when_training_starts(monkeypatch, clock, seconds=5.0)
    messages: list[str] = []

    result = run_plan(
        REPRO_PLAN,
        arms=["lr_qat"],
        out_dir=tmp_path,
        max_wall_seconds=5.0,
        progress=messages.append,
    )

    steps_requested = _plan_steps()
    assert result.partial is True
    assert result.runs == (), "an interrupted arm is not a completed cell"
    (entry,) = result.not_completed
    assert entry["arm"] == "lr_qat" and entry["seed"] == 0
    assert entry["steps_requested"] == steps_requested
    assert entry["steps_completed"] is not None
    assert 0 < int(entry["steps_completed"]) < steps_requested, (
        "the stop must land at a checkpoint boundary inside the schedule, not after it"
    )
    assert "checkpoint boundary" in str(entry["reason"])
    assert any("STOPPED" in message for message in messages)
    # Nothing is measured for an incompletely trained arm: no quality number, no arm manifest, no
    # metrics file - the run manifest is the only record of what happened.
    assert not (tmp_path / "lr_qat" / "seed-0" / "run_manifest.json").exists()
    assert not (tmp_path / "lr_qat" / "seed-0" / "metrics.json").exists()
    assert result.run_manifest_path is not None
    manifest = validate_manifest_file(result.run_manifest_path)
    assert manifest["status"] == "partial"
    assert manifest["metrics"]["run.cells_completed"] == []
    assert manifest["metrics"]["run.arms_not_completed"] == ["lr_qat"]
    assert manifest["metrics"]["run.not_completed_steps_completed"] == [
        int(entry["steps_completed"])
    ]


def test_a_budget_that_cannot_be_a_deadline_is_refused(tmp_path: Path, offline: None) -> None:
    """A deadline is either ``None`` or a finite, non-negative number of seconds."""
    for unusable in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_wall_seconds"):
            run_plan(REPRO_PLAN, out_dir=tmp_path, max_wall_seconds=unusable)


# --------------------------------------------------------------------------------------
# The notebook chain: a cleanly stopped run is not a successful run
# --------------------------------------------------------------------------------------
def _stage_context(tmp_path: Path, *, run_out: str, run_rc: int = 0) -> Path:
    """A stage context shaped like the notebook's, with the runner's own output recorded."""
    work = tmp_path / "work"
    for relative in ("artifacts/run", "repo", "logs"):
        (work / relative).mkdir(parents=True, exist_ok=True)
    document = {
        "run_id": "unit-plan-run",
        "spec_sha256": "sha256:" + "b" * 64,
        "spec": {
            "run_id": "unit-plan-run",
            "platform": "kaggle",
            "experiment_config": REPRO_PLAN,
            "overrides": [],
            "dataset_refs": [],
            "seeds": [0],
            "git_commit": "0" * 40,
            "allow_dirty": True,
            "install_spec": "uv sync --all-extras",
            "measurement_class_expected": 2,
            "gpu_required": False,
        },
        "paths": {
            "repo": str(work / "repo"),
            "work": str(work),
            "artifact": str(work / "artifacts" / "run"),
            "log": str(work / "logs" / "run.log"),
            "data": str(work / "data"),
            "export": str(work / "export"),
        },
        "scalars": {
            "started_utc": "2026-10-11T00:00:00Z",
            "finished_utc": "2026-10-11T00:05:00Z",
            "run_rc": run_rc,
            "run_out": run_out,
            "wall_time_s": 12.5,
            "environment": {"python": "3.11"},
            "notebook_digest": "sha256:" + "a" * 64,
            "notebook_template_version": "2.0.2",
            "head_sha": "0" * 40,
        },
    }
    path = work / "stage-context.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_manifest_stage_records_a_partial_runner_as_partial(tmp_path: Path) -> None:
    """A runner that stopped at its deadline exits zero: the stage must not call that success."""
    payload = {
        "status": "partial",
        "partial": True,
        "stop_reason": "the wall-clock budget (5s) was exhausted after 1000.0s",
        "cells_completed": ["fp16_reference/seed-0"],
        "cells_not_completed": ["lr_qat/seed-0"],
    }
    path = _stage_context(tmp_path, run_out=RESULT_LINE_PREFIX + json.dumps(payload))

    summary = run_stage("manifest", path)

    assert "status partial" in summary
    written = json.loads(
        (Path(json.loads(path.read_text())["paths"]["artifact"]) / "run_manifest.json").read_text()
    )
    assert written["status"] == "partial"
    assert "wall-clock" in str(written["failure_reason"])
    assert written["metrics"]["cloud.run_partial"] is True
    # ... while a run whose result line does not say partial is still a success.
    ok_path = _stage_context(
        tmp_path / "ok", run_out=RESULT_LINE_PREFIX + json.dumps({"status": "success"})
    )
    assert "status success" in run_stage("manifest", ok_path)
