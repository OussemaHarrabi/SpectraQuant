"""Unit tests for the Tier-0 coefficient-sweep reduction and report (Milestone 5, H3).

The sweep's *training* is exercised end-to-end by `scripts/experiments/regularizer_sweep.py` (its
output is the committed artifact); these tests pin the parts a regression would silently break: cell
deduplication, the direction-check reduction, and the statements the report must always carry
(measurement classes, substrate, and the frozen H3 falsifier with the cloud cell marked NOT RUN).
"""

from __future__ import annotations

import json

from spectraquant.regularizers.sweep import (
    ARM_NAMES,
    H3_FALSIFIER,
    SWEEP_FORMAT,
    RunRecord,
    SweepPlan,
    build_document,
    gather_arm_configs,
    predictive_cells,
    render_report,
    write_artifacts,
)


def _record(cell: str, group: str, seed: int, **overrides: object) -> RunRecord:
    payload: dict[str, object] = {
        "cell": cell,
        "group": group,
        "seed": seed,
        "weights": {
            "lambda_factor": 1.0,
            "lambda_round": 0.0,
            "lambda_residual": 0.0,
            "lambda_spectrum": 0.0,
        },
        "tail_rank": None,
        "train_loss_initial": 3.0,
        "train_loss_final": 1.0,
        "dev_nll_prepared": 1.5,
        "dev_nll_quantized": 1.6,
        "output_error_vs_dense": 2.0,
        "output_error_vs_prepared": 0.2,
        "factor_rounding_residual": 0.05,
        "product_rounding_residual": 0.08,
        "measured_tail_energy": 0.4,
        "accounted_bytes": 1234,
        "term_means": {"task": 1.0, "rounding": 0.01},
        "wall_time_s": 0.5,
    }
    payload.update(overrides)
    return RunRecord(**payload)  # type: ignore[arg-type]


def test_gather_arm_configs_validates_every_arm() -> None:
    configs = gather_arm_configs()
    assert set(configs) == set(ARM_NAMES)
    assert configs["none"].method.regularizer.is_noop()
    assert not configs["full"].method.regularizer.is_noop()


def test_predictive_cells_cover_the_arms_and_deduplicate_the_sweeps() -> None:
    plan = SweepPlan()
    cells = predictive_cells(plan, gather_arm_configs())
    labels = {cell.label for cell in cells}
    assert set(ARM_NAMES) <= labels
    # lambda_round = 0 is the `factorization` arm and lambda_round = 1 the `rounding` arm, so the
    # sweep grids collapse onto them instead of duplicating the identical run.
    lambdas = [cell.lambdas for cell in cells]
    assert len(lambdas) == len(set(lambdas))
    # The swept grids are *covered* by the cell set; the grid endpoints are carried by the arms
    # (lambda_round = 0 is `factorization`, lambda_round = 1 is `rounding`, and the same for the
    # spectral sweep), which is exactly why the direction checks match on the coefficient pattern
    # rather than on the cell group.
    round_family = [
        cell for cell in cells if cell.lambdas[0] == 1.0 and cell.lambdas[2:] == (0.0, 0.0)
    ]
    assert set(plan.round_lambdas) <= {cell.lambdas[1] for cell in round_family}
    spectrum_family = [cell for cell in cells if cell.lambdas[:3] == (1.0, 0.0, 0.0)]
    assert set(plan.spectrum_lambdas) <= {cell.lambdas[3] for cell in spectrum_family}
    for cell in cells:
        assert (cell.tail_rank == plan.tail_rank) == (cell.lambdas[3] > 0.0)


def test_document_reduces_a_monotone_family_into_a_direction_check() -> None:
    plan = SweepPlan(seeds=(0, 1))
    records = []
    for seed in plan.seeds:
        for lam in (0.0, 0.3, 3.0):
            weights = {
                "lambda_factor": 1.0,
                "lambda_round": lam,
                "lambda_residual": 0.0,
                "lambda_spectrum": 0.0,
            }
            records.append(
                _record(
                    f"round={lam:g}",
                    "lambda_round",
                    seed,
                    weights=weights,
                    factor_rounding_residual=0.08 / (1.0 + lam),
                    product_rounding_residual=0.10 / (1.0 + lam),
                )
            )
    document = build_document(
        records, {}, {"config_path": "c", "compression": {}}, plan, wall_time_s=1.0
    )
    check = document["direction_checks"]["lambda_round_vs_factor_rounding_residual"]
    assert check["lambdas"] == [0.0, 0.3, 3.0]
    assert check["non_increasing"] and check["strictly_decreasing"]
    assert check["relative_change"] < -0.5
    product = document["direction_checks"]["lambda_round_vs_product_rounding_residual"]
    assert product["strictly_decreasing"]
    assert document["confirmatory_cell"]["status"] == "NOT RUN"
    assert document["measurement_classes"]["accounted_bytes"] == 1


def test_report_states_the_falsifier_substrate_and_measurement_classes() -> None:
    plan = SweepPlan(seeds=(0,))
    records = [
        _record(
            "none",
            "arm",
            0,
            weights={
                "lambda_factor": 0.0,
                "lambda_round": 0.0,
                "lambda_residual": 0.0,
                "lambda_spectrum": 0.0,
            },
        )
    ]
    records.append(
        _record(
            "full",
            "arm",
            0,
            weights={
                "lambda_factor": 1.0,
                "lambda_round": 1.0,
                "lambda_residual": 0.0,
                "lambda_spectrum": 1.0,
            },
            tail_rank=4,
        )
    )
    for lam in (0.0, 1.0):
        records.append(
            _record(
                f"round={lam:g}",
                "lambda_round",
                0,
                weights={
                    "lambda_factor": 1.0,
                    "lambda_round": lam,
                    "lambda_residual": 0.0,
                    "lambda_spectrum": 0.0,
                },
            )
        )
    records.append(
        _record(
            "spectrum=0.1",
            "lambda_spectrum",
            0,
            weights={
                "lambda_factor": 1.0,
                "lambda_round": 0.0,
                "lambda_residual": 0.0,
                "lambda_spectrum": 0.1,
            },
            tail_rank=4,
        )
    )
    document = build_document(
        records, {}, {"config_path": "c", "compression": {}}, plan, wall_time_s=1.0
    )
    assert document["format"] == SWEEP_FORMAT
    report = render_report(document)
    assert H3_FALSIFIER in report
    assert "NOT RUN" in report
    assert "LOCAL-FIXTURE" in report
    assert "EXPLORATORY" in report
    assert "class 1" in report and "class 2" in report
    assert "cannot yet speak to it at confirmatory scope" in report
    assert "Direction checks" in report
    assert "lambda_residual" in report


def test_write_artifacts_round_trips_through_json(tmp_path) -> None:
    plan = SweepPlan(seeds=(0,))
    records = [_record("none", "arm", 0), _record("full", "arm", 0)]
    document = build_document(
        records, {}, {"config_path": "c", "compression": {}}, plan, wall_time_s=1.0
    )
    artifact, report = write_artifacts(
        document,
        artifact_path=tmp_path / "sweep.json",
        report_path=tmp_path / "report.md",
    )
    reloaded = json.loads(artifact.read_text(encoding="utf-8"))
    assert reloaded["format"] == SWEEP_FORMAT
    assert reloaded["cells"]["none"]["accounted_bytes"] == 1234
    assert report.read_text(encoding="utf-8").startswith("# Regularizer report")
