"""Transfer-gate arithmetic and cross-run comparability, using synthetic metrics."""

import copy
import hashlib
import math

import pytest


def fixture(key, reductions):
    report = {
        "model_key": key,
        "status": "success",
        "git": {"commit": "a" * 40},
        "data": {
            name: {"sha256": name + "-fingerprint"}
            for name in ("calibration", "c4_reserved", "pg19")
        },
        "config": {
            "selected_model_key": key,
            "transfer_gate": {
                "positive_cells_required": 3,
                "geometric_mean_reduction_percent_minimum": 0.5,
            },
        },
        "arms": {"rtn_axis0": {"tests": {}}, "mse_axis0": {"tests": {}}},
    }
    for corpus, reduction in zip(("c4_reserved", "pg19"), reductions, strict=True):
        report["arms"]["rtn_axis0"]["tests"][corpus] = {"perplexity": 20, "nll": math.log(20)}
        value = 20 * (1 - reduction / 100)
        report["arms"]["mse_axis0"]["tests"][corpus] = {"perplexity": value, "nll": math.log(value)}
    return report


def test_transfer_gate_uses_four_matched_log_ratios_and_preserves_regression():
    from spectraquant.evaluation.phase4_report import aggregate_transfer

    result = aggregate_transfer([fixture("smol360", (2, 3)), fixture("pythia410", (-1, 2))])
    assert result["positive_primary_cells"] == 3
    expected = 100 * (1 - (0.98 * 0.97 * 1.01 * 0.98) ** 0.25)
    assert result["geometric_mean_reduction_percent"] == pytest.approx(expected)
    assert result["transfer_gate_passed"] is True
    assert result["primary_cells"][2]["reduction_percent"] == pytest.approx(-1)


def test_cross_model_data_drift_and_incomplete_coverage_are_rejected():
    from spectraquant.evaluation.phase4_report import aggregate_transfer

    first, second = fixture("smol360", (2, 3)), fixture("pythia410", (1, 2))
    for name in first["data"]:
        wrong = copy.deepcopy(second)
        wrong["data"][name]["sha256"] = "changed"
        with pytest.raises(ValueError, match="fingerprint"):
            aggregate_transfer([first, wrong])
    with pytest.raises(ValueError, match="both models"):
        aggregate_transfer([first])


def test_same_configuration_cannot_hide_different_scientific_implementation():
    from spectraquant.evaluation.phase4_report import aggregate_transfer

    first, second = fixture("smol360", (2, 3)), fixture("pythia410", (1, 2))
    second["git"]["commit"] = "b" * 40
    with pytest.raises(ValueError, match="commit"):
        aggregate_transfer([first, second])


def test_transfer_rejects_changed_layer_scores_before_using_their_diagnostics(
    tmp_path, monkeypatch
):
    from spectraquant.evaluation import phase4_report

    original = b'[{"weight_mse": 0.001}]'
    (tmp_path / "scores.json").write_bytes(b'[{"weight_mse": 0.0}]')
    manifest = {
        "metrics": {
            "cloud.artifact_checksums": [
                "artifacts/runs/phase4-transfer-smol360/scores.json=sha256:"
                + hashlib.sha256(original).hexdigest()
            ]
        }
    }
    # Isolate this boundary: report authentication has already succeeded, but
    # a locally modified score file must fail before any scientific claims.
    monkeypatch.setattr(phase4_report, "verify_study_manifest", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match=r"layer-score.*checksum"):
        phase4_report.verify_transfer(
            {"model_key": "smol360", "status": "failed", "git": {"dirty": False}},
            tmp_path,
            manifest,
        )


def test_layer_score_authentication_requires_unique_remote_provenance(tmp_path):
    from spectraquant.evaluation.phase4_report import authenticate_layer_scores

    (tmp_path / "scores.json").write_bytes(b"[]")
    digest = "sha256:" + hashlib.sha256(b"[]").hexdigest()
    entry = f"artifacts/runs/phase4-transfer-smol360/scores.json={digest}"
    assert (
        authenticate_layer_scores(
            tmp_path, {"metrics": {"cloud.artifact_checksums": [entry]}}, "smol360"
        )
        == digest
    )
    for entries in ([], [entry, entry], [entry.replace("smol360", "pythia410")]):
        with pytest.raises(ValueError, match="provenance"):
            authenticate_layer_scores(
                tmp_path, {"metrics": {"cloud.artifact_checksums": entries}}, "smol360"
            )
