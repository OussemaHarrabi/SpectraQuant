"""The Milestone-7 result-registry gate: cell keys, seed matrices, duplicates, incomparability.

Each test writes real manifests to a temporary runs directory and drives the real pass, so the
behaviour under test is the behaviour the gate has - not a mock of it. The manifests are built with
:class:`~spectraquant.reporting.manifests.RunManifest` and written through
:func:`~spectraquant.reporting.manifests.write_manifest`, which validates them against the published
schema, so a test can only exercise states the schema admits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from spectraquant.cli.main import app
from spectraquant.reporting.manifests import (
    CompressionBlock,
    HardwareBlock,
    RunManifest,
    TrainingBlock,
    utc_timestamp,
    write_manifest,
)
from spectraquant.reporting.run_registry import (
    CODE_DUPLICATE_RUN,
    CODE_INCOMPARABLE,
    CODE_INTERRUPTED_RUN,
    CODE_MEMORY_GATE_REFUSED,
    CODE_MISSING_SEEDS,
    CODE_RUNS_DIR_MISSING,
    CODE_SCHEMA_INVALID,
    CODE_UNEQUAL_MEMORY,
    CODE_UNPARSEABLE,
    SEVERITY_BLOCKING,
    SEVERITY_WARNING,
    comparability_key,
    validate_registry,
)

runner = CliRunner()

SERIALIZER = "spectraquant-sqpack-v1"


def build_manifest(
    run_id: str,
    *,
    seed: int = 0,
    substrate: str | None = "colab",
    model_id: str | None = "HuggingFaceTB/SmolLM2-135M",
    model_revision: str | None = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    dataset_ids: tuple[str, ...] = ("EleutherAI/wikitext_document_level",),
    dataset_revisions: tuple[str | None, ...] = ("647234772b9554e208af6c826f23b99e3cac88c8",),
    split: str = "test",
    arm: str | None = "ptq_uniform_4",
    method: str = "rtn",
    bits: int | None = 4,
    measured_bytes: int | None = 1_001_200,
    accounted_bytes: int | None = 1_000_000,
    tolerance: int | float | None = 200,
    serializer: str | None = SERIALIZER,
    source: str | None = "measured",
    measurement_class: int | None = 3,
    status: str = "success",
    failure_reason: str | None = None,
    expected_seeds: tuple[int, ...] | None = (0,),
    seed_floor: int | None = 1,
    resolved_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    model_name: str = "tiny-char-transformer",
) -> RunManifest:
    """A schema-valid plan-style manifest, every comparability field overridable."""
    recorded_metrics: dict[str, Any] = {"parameter_count": 2_000_000}
    if substrate is not None:
        recorded_metrics["plan.substrate"] = substrate
    if arm is not None:
        recorded_metrics["plan.arm"] = arm
        recorded_metrics["plan.arm_kind"] = "ptq_uniform"
    if expected_seeds is not None:
        recorded_metrics["plan.reported_seeds"] = list(expected_seeds)
    if seed_floor is not None:
        recorded_metrics["plan.seed_floor"] = seed_floor
    if metrics is not None:
        recorded_metrics.update(metrics)
    return RunManifest(
        run_id=run_id,
        timestamp_utc=utc_timestamp(),
        git_commit="0" * 40,
        git_dirty=False,
        config_path="configs/tier1/smollm2_135m.yaml",
        resolved_config=resolved_config or {"model": {"name": model_name}, "plan": {"name": "t"}},
        model_id=model_id,
        model_revision=model_revision,
        dataset_ids=list(dataset_ids),
        dataset_revisions=list(dataset_revisions),
        dataset_checksums=[],
        split=split,
        seed=seed,
        hardware=HardwareBlock(
            platform="Linux-6.6.0",
            system="Linux",
            release="6.6.0",
            machine="x86_64",
            processor="Intel Xeon",
            gpu_devices=["Tesla T4"],
            torch_device="cpu",
            torch_threads=1,
        ),
        software={"python": "3.11.16", "torch": "2.14.1"},
        compression=CompressionBlock(
            method=method,
            bits=bits,
            group_size=None,
            accounted_bytes=accounted_bytes,
            measured_bytes=measured_bytes,
            measured_bytes_tolerance=tolerance,
            serializer=serializer,
            nominal_bits_per_param=float(bits) if bits is not None else None,
            measured_bits_per_param=8.0 if measured_bytes is not None else None,
            bytes_source=source,  # type: ignore[arg-type]
        ),
        theoretical_bits=float(bits) if bits is not None else None,
        packed_bytes=measured_bytes,
        runtime_backend="torch-cpu-fp32",
        measurement_class=measurement_class,
        training=TrainingBlock(steps=0, tokens=0, wall_time_s=0.0, peak_mem_mb=None),
        metrics=recorded_metrics,
        status=status,
        failure_reason=failure_reason,
        log_path=None,
        artifact_paths=[],
    )


def write_run(
    directory: Path,
    manifest: RunManifest,
    *,
    subdir: str | None = None,
    name: str | None = None,
) -> Path:
    """Write a manifest into ``<directory>/<subdir>/run_manifest.json`` (the plan layout).

    ``subdir`` defaults to ``<arm>/seed-<n>``; pass it explicitly when two runs of one arm must land
    in the same cell (they would otherwise overwrite each other).
    """
    relative = (
        subdir
        if subdir is not None
        else f"{manifest.metrics.get('plan.arm') or 'no-arm'}/seed-{manifest.seed}"
    )
    return write_manifest(manifest, directory / relative / (name or "run_manifest.json"))


def finding_codes(report: Any, code: str) -> list[Any]:
    """All findings of one code."""
    return [finding for finding in report.findings if finding.code == code]


# --------------------------------------------------------------------------------------
# cell keys
# --------------------------------------------------------------------------------------
def test_cell_key_is_stable_and_order_independent() -> None:
    first = build_manifest("run-a", metrics={"a": 1, "b": 2})
    second = build_manifest("run-b", metrics={"b": 2, "a": 1})
    # The set-like lists (device set, exclusion set) must not make the key depend on enumeration
    # order, and the unrelated run id / metric dict order must not enter it at all.
    first = first.model_copy(
        update={
            "hardware": first.hardware.model_copy(
                update={"gpu_devices": ["Tesla T4", "A100-SXM4-40GB"]}
            ),
            "compression": first.compression.model_copy(
                update={"exclusions": ["lm_head", "embed"]}
            ),
        }
    )
    second = second.model_copy(
        update={
            "hardware": second.hardware.model_copy(
                update={"gpu_devices": ["A100-SXM4-40GB", "Tesla T4"]}
            ),
            "compression": second.compression.model_copy(
                update={"exclusions": ["embed", "lm_head"]}
            ),
        }
    )
    key = comparability_key(first)

    assert key == comparability_key(second)
    assert key.digest == comparability_key(second).digest
    assert key == comparability_key(first)  # pure function: same input, same key
    assert key.digest != comparability_key(build_manifest("run-c", substrate="kaggle")).digest


def test_cell_key_separates_every_frozen_field() -> None:
    base = comparability_key(build_manifest("run"))
    variants = {
        "substrate": build_manifest("run", substrate="kaggle"),
        "model_revision": build_manifest("run", model_revision="f" * 40),
        "dataset_revision": build_manifest("run", dataset_revisions=("e" * 40,)),
        "split": build_manifest("run", split="validation"),
        "arm": build_manifest("run", arm="ptq_uniform_8"),
        "compression": build_manifest("run", bits=8),
    }
    for field, manifest in variants.items():
        assert comparability_key(manifest).digest != base.digest, field


def test_hardware_change_alone_changes_the_cell_key() -> None:
    key = comparability_key(build_manifest("run"))
    other = build_manifest("run")
    other = other.model_copy(
        update={"hardware": other.hardware.model_copy(update={"gpu_devices": ["A100-SXM4-40GB"]})}
    )

    assert comparability_key(other).digest != key.digest
    assert comparability_key(other).family_digest == key.family_digest
    assert comparability_key(other).group_digest != key.group_digest


# --------------------------------------------------------------------------------------
# the seed matrix
# --------------------------------------------------------------------------------------
def test_missing_seed_is_detected(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-seed0", seed=0, expected_seeds=(0, 1, 2), seed_floor=3))

    report = validate_registry(tmp_path)

    assert not report.ok
    missing = finding_codes(report, CODE_MISSING_SEEDS)
    assert len(missing) == 1
    assert missing[0].severity == SEVERITY_BLOCKING
    assert "missing [1, 2]" in missing[0].detail
    cell = report.cells[0]
    assert cell.expected_seeds == (0, 1, 2)
    assert cell.present_seeds == (0,)
    assert cell.missing_seeds == (1, 2)


def test_complete_seed_matrix_is_not_reported(tmp_path: Path) -> None:
    for seed in (0, 1, 2):
        write_run(
            tmp_path,
            build_manifest(f"arm-seed{seed}", seed=seed, expected_seeds=(0, 1, 2), seed_floor=3),
        )

    report = validate_registry(tmp_path)

    assert report.ok, report.findings
    assert report.cells[0].missing_seeds == ()
    assert report.cells[0].present_seeds == (0, 1, 2)


def test_zero_seed_is_a_real_seed(tmp_path: Path) -> None:
    """Falsy-seed bugs are classic: seed 0 must count as present, not as unset."""
    write_run(tmp_path, build_manifest("arm-seed0", seed=0, expected_seeds=(0,)))

    report = validate_registry(tmp_path)

    assert report.ok, report.findings
    assert report.cells[0].present_seeds == (0,)


def test_aborted_seed_does_not_count_as_present(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        build_manifest(
            "arm-seed0",
            seed=0,
            status="aborted",
            failure_reason="wall-clock cap exceeded",
            expected_seeds=(0, 1, 2),
            seed_floor=3,
        ),
    )

    report = validate_registry(tmp_path)

    interrupted = finding_codes(report, CODE_INTERRUPTED_RUN)
    assert len(interrupted) == 1
    assert interrupted[0].severity == SEVERITY_WARNING
    assert "wall-clock cap exceeded" in interrupted[0].detail
    missing = finding_codes(report, CODE_MISSING_SEEDS)
    assert [finding.severity for finding in missing] == [SEVERITY_BLOCKING]
    assert "missing [0, 1, 2]" in missing[0].detail


def test_interrupted_run_alone_is_only_a_warning(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        build_manifest(
            "arm-seed0",
            seed=0,
            status="failed",
            failure_reason="OOM",
            expected_seeds=None,
            seed_floor=None,
        ),
    )

    report = validate_registry(tmp_path)

    assert report.ok, report.blocking
    assert [finding.code for finding in report.findings] == [CODE_INTERRUPTED_RUN]
    assert report.warnings[0].severity == SEVERITY_WARNING


def test_conflicting_seed_declaration_is_reported(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-seed0", seed=0, expected_seeds=(0, 1, 2)))
    write_run(tmp_path, build_manifest("arm-seed1", seed=1, expected_seeds=(0, 1)))

    report = validate_registry(tmp_path)

    conflicts = finding_codes(report, "conflicting_seed_declaration")
    assert len(conflicts) == 1
    assert "disagree" in conflicts[0].detail


# --------------------------------------------------------------------------------------
# duplicates
# --------------------------------------------------------------------------------------
def test_duplicate_cell_seed_is_detected(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-run-1", seed=0), subdir="run-1/seed-0")
    write_run(tmp_path, build_manifest("arm-run-2", seed=0), subdir="run-2/seed-0")

    report = validate_registry(tmp_path)

    assert not report.ok
    duplicates = finding_codes(report, CODE_DUPLICATE_RUN)
    assert len(duplicates) == 1
    assert "byte-identical (a replication)" in duplicates[0].detail
    assert set(duplicates[0].runs) == {"arm-run-1", "arm-run-2"}
    assert report.cells[0].duplicates[0].seed == 0
    assert report.cells[0].duplicates[0].identical_metrics


def test_duplicate_with_conflicting_metrics_is_marked_as_a_conflict(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-run-1", seed=0), subdir="run-1/seed-0")
    write_run(
        tmp_path,
        build_manifest("arm-run-2", seed=0, metrics={"perplexity": 41.0}),
        subdir="run-2/seed-0",
    )

    report = validate_registry(tmp_path)

    assert not report.cells[0].duplicates[0].identical_metrics
    assert "DIFFER" in finding_codes(report, CODE_DUPLICATE_RUN)[0].detail


def test_failed_run_beside_a_success_is_not_a_duplicate(tmp_path: Path) -> None:
    """A recorded failure plus a successful re-run is the documented failure policy, not a dupe."""
    write_run(
        tmp_path,
        build_manifest("arm-run-1", seed=0, status="failed", failure_reason="OOM"),
        subdir="run-1/seed-0",
    )
    write_run(tmp_path, build_manifest("arm-run-2", seed=0), subdir="run-2/seed-0")

    report = validate_registry(tmp_path)

    assert finding_codes(report, CODE_DUPLICATE_RUN) == []
    assert len(finding_codes(report, CODE_INTERRUPTED_RUN)) == 1


# --------------------------------------------------------------------------------------
# broken files
# --------------------------------------------------------------------------------------
def test_schema_invalid_file_is_reported_without_crashing_the_pass(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-seed0", seed=0, expected_seeds=(0,)))
    broken = tmp_path / "broken.manifest.json"
    broken.write_text('{"run_id": "x", "schema_version": 2}', encoding="utf-8")

    report = validate_registry(tmp_path)

    invalid = finding_codes(report, CODE_SCHEMA_INVALID)
    assert len(invalid) == 1
    assert invalid[0].severity == SEVERITY_BLOCKING
    assert "broken.manifest.json" in invalid[0].detail
    # The pass continued: the valid manifest is still indexed.
    assert len(report.runs) == 1
    assert report.cells[0].present_seeds == (0,)
    assert len(report.files) == 2


def test_unparseable_file_is_reported_without_crashing_the_pass(tmp_path: Path) -> None:
    (tmp_path / "truncated.manifest.json").write_text('{"run_id": "x",', encoding="utf-8")

    report = validate_registry(tmp_path)

    unparseable = finding_codes(report, CODE_UNPARSEABLE)
    assert len(unparseable) == 1
    assert unparseable[0].severity == SEVERITY_BLOCKING
    assert "invalid JSON" in unparseable[0].detail
    assert report.cells == ()


def test_orphan_metrics_file_is_reported(tmp_path: Path) -> None:
    target = tmp_path / "ptq_uniform_4" / "seed-0"
    target.mkdir(parents=True)
    (target / "metrics.json").write_text('{"perplexity": 42.0}', encoding="utf-8")

    report = validate_registry(tmp_path)

    assert finding_codes(report, "orphan_metrics")[0].severity == SEVERITY_WARNING
    assert report.ok


def test_missing_runs_dir_is_a_warning(tmp_path: Path) -> None:
    report = validate_registry(tmp_path / "not-there")

    assert report.ok
    assert [finding.code for finding in report.findings] == [CODE_RUNS_DIR_MISSING]
    assert report.cells == ()


# --------------------------------------------------------------------------------------
# incomparability
# --------------------------------------------------------------------------------------
def test_cells_differing_only_in_substrate_are_incomparable(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("colab-run", substrate="colab"), subdir="colab/seed-0")
    write_run(tmp_path, build_manifest("kaggle-run", substrate="kaggle"), subdir="kaggle/seed-0")

    report = validate_registry(tmp_path)

    assert len(report.cells) == 2
    incomparable = finding_codes(report, CODE_INCOMPARABLE)
    assert len(incomparable) == 1
    assert incomparable[0].axis == "substrate"
    assert incomparable[0].severity == SEVERITY_WARNING
    assert "must not be pooled" in incomparable[0].detail
    assert all(cell.incomparable_axes == ("substrate",) for cell in report.cells)
    assert report.ok


def test_cells_differing_only_in_hardware_are_incomparable(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("t4-run"), subdir="t4/seed-0")
    a100 = build_manifest("a100-run")
    a100 = a100.model_copy(
        update={"hardware": a100.hardware.model_copy(update={"gpu_devices": ["A100-SXM4-40GB"]})}
    )
    write_run(tmp_path, a100, subdir="a100/seed-0")

    report = validate_registry(tmp_path)

    incomparable = finding_codes(report, CODE_INCOMPARABLE)
    assert [finding.axis for finding in incomparable] == ["hardware"]


def test_different_arms_of_one_plan_are_not_incomparable(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("fp16", arm="fp16_reference", method="none", bits=None))
    write_run(tmp_path, build_manifest("ptq4", arm="ptq_uniform_4"))

    report = validate_registry(tmp_path)

    assert len(report.cells) == 2
    assert finding_codes(report, CODE_INCOMPARABLE) == []


# --------------------------------------------------------------------------------------
# equal memory
# --------------------------------------------------------------------------------------
def test_unequal_memory_arm_pair_is_flagged(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-a", arm="ptq_uniform_4", measured_bytes=1_001_200))
    write_run(
        tmp_path,
        build_manifest("arm-b", arm="rank_then_quant", measured_bytes=1_020_000),
    )

    report = validate_registry(tmp_path)

    flagged = finding_codes(report, CODE_UNEQUAL_MEMORY)
    assert len(flagged) == 1
    assert flagged[0].severity == SEVERITY_BLOCKING
    assert "more than the 200-byte tolerance" in flagged[0].detail
    assert not report.ok
    assert report.equal_memory == ()


def test_pair_within_tolerance_is_recorded_as_an_equal_memory_verdict(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-a", arm="ptq_uniform_4", measured_bytes=1_001_200))
    write_run(
        tmp_path,
        build_manifest("arm-b", arm="rank_then_quant", measured_bytes=1_001_300),
    )

    report = validate_registry(tmp_path)

    assert finding_codes(report, CODE_UNEQUAL_MEMORY) == []
    assert len(report.equal_memory) == 1
    verdict = report.equal_memory[0]
    assert verdict["equal"] is True
    assert verdict["difference_bytes"] == 100
    assert verdict["measurement_class"] == 3


def test_pair_the_gate_cannot_gate_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    """Two arms with no declared tolerance are not a declared equal-memory comparison."""
    write_run(
        tmp_path,
        build_manifest("arm-a", arm="ptq_uniform_4", tolerance=None, measured_bytes=1_001_200),
    )
    write_run(
        tmp_path,
        build_manifest("arm-b", arm="rank_then_quant", tolerance=None, measured_bytes=1_020_000),
    )

    report = validate_registry(tmp_path)

    refused = finding_codes(report, CODE_MEMORY_GATE_REFUSED)
    assert len(refused) == 1
    assert refused[0].severity == SEVERITY_WARNING
    assert "no tolerance" in refused[0].detail
    assert report.ok, report.blocking


def test_runs_of_one_cell_are_not_gated_against_each_other(tmp_path: Path) -> None:
    """Two seeds of one arm are not an arm pair; only cross-arm pairs are gated."""
    write_run(tmp_path, build_manifest("arm-seed0", seed=0, accounted_bytes=1_000_000))
    write_run(tmp_path, build_manifest("arm-seed1", seed=1, accounted_bytes=999_000))

    report = validate_registry(tmp_path)

    assert report.equal_memory == ()
    assert report.ok, report.findings


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
FIXTURE_CONFIG = {
    "fixture": {
        "kind": "hand-authored gate fixture",
        "not_a_real_run": True,
        "note": "no experiment was executed",
    }
}


def test_fixture_is_not_a_cell_and_never_blocks(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        build_manifest(
            "fixture-a",
            resolved_config=dict(FIXTURE_CONFIG),
            arm="fixture-a",
            measured_bytes=1_001_200,
        ),
    )
    write_run(
        tmp_path,
        build_manifest(
            "fixture-b",
            resolved_config=dict(FIXTURE_CONFIG),
            arm="fixture-b",
            measured_bytes=1_020_000,
        ),
    )

    report = validate_registry(tmp_path)

    assert report.cells == ()
    assert report.runs == ()
    assert len(report.fixtures) == 2
    assert report.ok
    flagged = finding_codes(report, CODE_UNEQUAL_MEMORY)
    assert len(flagged) == 1
    assert flagged[0].severity == SEVERITY_WARNING
    assert flagged[0].fixture_only


def test_fixture_only_directory_is_reported_as_such(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        build_manifest(
            "fixture-a", resolved_config=dict(FIXTURE_CONFIG), arm="fixture-a", expected_seeds=None
        ),
    )

    report = validate_registry(tmp_path)

    assert report.directories[0].fixture_only
    assert report.directories[0].fixtures == 1
    assert finding_codes(report, CODE_MISSING_SEEDS) == []


def test_fixture_gate_is_per_comparison_group(tmp_path: Path) -> None:
    """Fixtures under different substrate/hardware are different groups: no cross-group pairing."""
    write_run(
        tmp_path,
        build_manifest(
            "fixture-colab",
            resolved_config=dict(FIXTURE_CONFIG),
            substrate="colab",
            arm=None,
            tolerance=None,
            measured_bytes=1_001_200,
        ),
        subdir="colab/seed-0",
    )
    write_run(
        tmp_path,
        build_manifest(
            "fixture-kaggle",
            resolved_config=dict(FIXTURE_CONFIG),
            substrate="kaggle",
            arm=None,
            tolerance=None,
            measured_bytes=1_500_000,
        ),
        subdir="kaggle/seed-0",
    )

    report = validate_registry(tmp_path)

    assert len(report.fixtures) == 2
    assert report.equal_memory == ()
    assert [
        finding.code
        for finding in report.findings
        if finding.code in {CODE_UNEQUAL_MEMORY, CODE_MEMORY_GATE_REFUSED}
    ] == []
    assert report.ok


# --------------------------------------------------------------------------------------
# JSON index
# --------------------------------------------------------------------------------------
def test_json_index_is_deterministic(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-seed0", seed=0))
    write_run(tmp_path, build_manifest("arm-seed1", seed=1))
    (tmp_path / "broken.manifest.json").write_text("{}", encoding="utf-8")

    first = validate_registry(tmp_path).to_json_dict()
    second = validate_registry(tmp_path).to_json_dict()

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["summary"]["cells"] == 1
    assert first["summary"]["ok"] is False
    assert first["blocking"] is True


def test_report_paths_are_relative_to_the_runs_dir(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-seed0", seed=0))

    report = validate_registry(tmp_path)

    assert report.runs[0].path == "ptq_uniform_4/seed-0/run_manifest.json"
    assert report.directories[0].path == "ptq_uniform_4/seed-0"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _combined(result: Any) -> str:
    return result.output + (getattr(result, "stderr", "") or "")


def test_cli_exits_non_zero_on_a_blocking_problem(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-run-1", seed=0), subdir="run-1/seed-0")
    write_run(tmp_path, build_manifest("arm-run-2", seed=0), subdir="run-2/seed-0")

    result = runner.invoke(app, ["registry-validate", "--runs", str(tmp_path)])

    assert result.exit_code == 1, _combined(result)
    assert "blocking" in _combined(result)
    assert CODE_DUPLICATE_RUN in _combined(result)


def test_cli_exits_zero_for_warnings_only(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        build_manifest(
            "arm-seed0",
            seed=0,
            status="aborted",
            failure_reason="pre-empted session",
            expected_seeds=None,
            seed_floor=None,
        ),
    )

    result = runner.invoke(app, ["registry-validate", "--runs", str(tmp_path)])

    assert result.exit_code == 0, _combined(result)
    assert CODE_INTERRUPTED_RUN in _combined(result)


def test_cli_exits_zero_on_the_committed_fixtures(comparability_fixtures_dir: Path) -> None:
    """The committed equal-memory fixtures are declared non-runs: reported, never blocking."""
    result = runner.invoke(app, ["registry-validate", "--runs", str(comparability_fixtures_dir)])

    assert result.exit_code == 0, _combined(result)
    assert "fixture-only" in _combined(result)
    assert CODE_UNEQUAL_MEMORY in _combined(result)


def test_cli_json_output_and_out_file(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-seed0", seed=0, expected_seeds=(0,)))
    out = tmp_path / "index.json"

    result = runner.invoke(
        app, ["registry-validate", "--runs", str(tmp_path), "--json", "--out", str(out)]
    )

    assert result.exit_code == 0, _combined(result)
    assert json.loads(result.output)["summary"]["cells"] == 1
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(result.output)


def test_cli_json_exit_code_follows_the_blocking_findings(tmp_path: Path) -> None:
    write_run(tmp_path, build_manifest("arm-seed0", seed=0, expected_seeds=(0, 1, 2)))

    result = runner.invoke(app, ["registry-validate", "--runs", str(tmp_path), "--json"])

    assert result.exit_code == 1, _combined(result)
    assert json.loads(result.output)["blocking"] is True
