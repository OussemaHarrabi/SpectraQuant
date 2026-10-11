"""Artifact integrity and honest interpretation of Phase-2 results."""

import hashlib

import pytest

from spectraquant.evaluation.phase2_report import (
    summarize,
    verify_checkpoints,
    verify_study_manifest,
)


def test_summary_distinguishes_dev_selection_from_test_and_scope():
    report = {
        "status": "success",
        "target_payload_budget_bytes": 100,
        "config": {"seeds": [0, 1, 2]},
        "uniform4_development": {"nll": 3.0},
        "selection_frozen_before_test": {"family": "bits"},
        "arms": {
            "uniform4": {
                "test": {"perplexity": 20.0, "tokens": 900},
                "storage": {"target_payload_bytes": 100, "full_checkpoint_bytes": 150},
            },
            "bits_seed0": {
                "test": {"perplexity": 18.0, "tokens": 900},
                "storage": {"target_payload_bytes": 99, "full_checkpoint_bytes": 149},
                "development": {"nll": 2.9},
            },
        },
    }
    summary = summarize(report)
    assert summary["rows"][1]["perplexity_reduction_percent"] == pytest.approx(10.0)
    assert summary["selected_family"] == "bits"
    assert (
        summary["scope"]
        == "one checkpoint; calibration-subset variation; class-2 quality and class-3 stored bytes"
    )


def test_mismatched_token_protocol_is_rejected():
    report = {
        "status": "success",
        "config": {"seeds": [0]},
        "target_payload_budget_bytes": 100,
        "selection_frozen_before_test": {"family": "uniform4"},
        "arms": {
            "uniform4": {
                "test": {"perplexity": 20, "tokens": 900},
                "storage": {"target_payload_bytes": 100, "full_checkpoint_bytes": 100},
            },
            "bits_seed0": {
                "test": {"perplexity": 19, "tokens": 800},
                "storage": {"target_payload_bytes": 100, "full_checkpoint_bytes": 100},
            },
        },
    }
    with pytest.raises(ValueError, match="token"):
        summarize(report)


def test_checkpoint_corruption_is_detected(tmp_path):
    directory = tmp_path / "checkpoints" / "uniform4"
    directory.mkdir(parents=True)
    raw = b"actual packed bytes"
    (directory / "weights.sqpack").write_bytes(raw)
    expected = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    report = {
        "arms": {
            "uniform4": {
                "storage": {
                    "files": {"weights.sqpack": expected},
                    "target_payload_bytes": len(raw),
                    "full_checkpoint_bytes": len(raw),
                }
            }
        }
    }
    assert verify_checkpoints(report, tmp_path)["verified_files"] == 1
    (directory / "weights.sqpack").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match=r"checksum|size"):
        verify_checkpoints(report, tmp_path)


def test_metrics_are_anchored_to_remote_manifest_digest(tmp_path):
    path = tmp_path / "metrics.json"
    raw = b'{"status": "success"}'
    path.write_bytes(raw)
    manifest = {
        "metrics": {"phase2.full_report_sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
    }
    verify_study_manifest(path, manifest)
    path.write_bytes(b'{"status": "fake"}')
    with pytest.raises(ValueError, match="manifest"):
        verify_study_manifest(path, manifest)
