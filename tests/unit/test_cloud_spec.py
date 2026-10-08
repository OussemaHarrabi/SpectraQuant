"""``RunSpec`` validation: the frozen contract of ``design-cloud-adapter.md`` §2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _cloud_fixtures import COMMIT, make_spec
from pydantic import ValidationError

from spectraquant.cloud.spec import (
    PLATFORM_MAX_TIMEOUT_MINUTES,
    PLATFORMS,
    DatasetRef,
    ExpectedArtifact,
    checksum_hex,
    load_spec,
    save_spec,
    spec_from_config,
    spec_sha256,
)


def test_spec_accepts_a_valid_local_cpu_spec() -> None:
    spec = make_spec()

    assert spec.platform == "local_cpu"
    assert spec.gpu_required is False
    assert spec.sha256().startswith("sha256:")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_every_documented_platform_is_accepted(platform: str) -> None:
    extra = {"max_cost_authorized_usd": 5.0} if platform == "colab_enterprise" else {}
    spec = make_spec(platform=platform, gpu_required=platform != "local_cpu", **extra)

    assert spec.platform == platform


def test_paid_platform_without_authorized_cost_is_an_error() -> None:
    with pytest.raises(ValidationError, match="max_cost_authorized_usd is required"):
        make_spec(platform="colab_enterprise", gpu_required=True)


def test_missing_git_commit_is_an_error() -> None:
    with pytest.raises(ValidationError, match="git_commit is required unless allow_dirty"):
        make_spec(git_commit="")


def test_short_git_commit_is_an_error() -> None:
    with pytest.raises(ValidationError, match="full 40-character lowercase SHA"):
        make_spec(git_commit="deadbeef")


def test_dirty_spec_may_omit_the_commit_when_explicitly_allowed() -> None:
    spec = make_spec(git_commit="", allow_dirty=True)

    assert spec.allow_dirty is True
    assert spec.git_commit == ""


def test_local_cpu_refuses_gpu_required() -> None:
    with pytest.raises(ValidationError, match="cannot satisfy gpu_required=True"):
        make_spec(platform="local_cpu", gpu_required=True)


@pytest.mark.parametrize("value", [0, 5])
def test_measurement_class_must_be_in_1_to_4(value: int) -> None:
    with pytest.raises(ValidationError, match="measurement_class_expected"):
        make_spec(measurement_class_expected=value)


def test_unknown_platform_is_an_error() -> None:
    with pytest.raises(ValidationError, match="platform"):
        make_spec(platform="lambda_labs")


def test_unknown_field_is_an_error() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        make_spec(secret_token="hunter2")


def test_timeout_above_the_platform_ceiling_is_an_error() -> None:
    with pytest.raises(ValidationError, match="exceeds the documented colab ceiling"):
        make_spec(platform="colab", gpu_required=True, timeout_minutes=721)


def test_seeds_must_be_non_empty_and_non_negative() -> None:
    with pytest.raises(ValidationError, match="at least one master seed"):
        make_spec(seeds=[])
    with pytest.raises(ValidationError, match="non-negative"):
        make_spec(seeds=[-1])


def test_run_id_must_be_a_slug() -> None:
    with pytest.raises(ValidationError, match="filesystem-safe slug"):
        make_spec(run_id="not a slug/with slashes")


def test_expected_artifact_checksum_is_validated() -> None:
    with pytest.raises(ValidationError, match="not a sha256 digest"):
        ExpectedArtifact(name="x.bin", sha256="deadbeef")
    with pytest.raises(ValidationError, match="not a sha256 digest"):
        DatasetRef(name="d", revision="r", split="train", checksum="sha1:abc")


def test_checksum_hex_accepts_both_spellings() -> None:
    bare = "a" * 64

    assert checksum_hex(f"sha256:{bare}") == bare
    assert checksum_hex(bare) == bare
    with pytest.raises(ValueError):
        checksum_hex("sha256:zz")


def test_spec_is_immutable() -> None:
    spec = make_spec()

    with pytest.raises(ValidationError):
        spec.run_id = "other"


def test_spec_round_trips_through_disk(tmp_path: Path) -> None:
    spec = make_spec()
    target = save_spec(spec, tmp_path / "run_spec.json")

    loaded = load_spec(target)

    assert loaded == spec
    assert loaded.sha256() == spec.sha256()
    assert json.loads(target.read_text())["run_id"] == spec.run_id


def test_spec_digest_changes_with_the_commit() -> None:
    first = make_spec(git_commit=COMMIT)
    second = make_spec(git_commit="1" * 40)

    assert spec_sha256(first) != spec_sha256(second)


def test_spec_from_config_derives_the_smoke_spec() -> None:
    spec = spec_from_config("configs/experiment/smoke.yaml", platform="colab", allow_dirty=True)

    assert spec.experiment_config == "configs/experiment/smoke.yaml"
    assert spec.platform == "colab"
    assert spec.gpu_required is True
    assert spec.seeds == [1234]
    assert spec.measurement_class_expected == 1  # method=none records no class in the manifest
    assert spec.timeout_minutes == PLATFORM_MAX_TIMEOUT_MINUTES["colab"]
    assert spec.dataset_refs[0].name == "synthetic-lcg-v1"
    assert spec.dataset_refs[0].checksum.startswith("sha256:")
    assert spec.dataset_refs[0].revision.startswith("config-sha256:")
    assert spec.git_commit and len(spec.git_commit) == 40


def test_spec_from_config_is_deterministic_for_a_fixed_run_id() -> None:
    kwargs = {"platform": "kaggle", "run_id": "fixed-run", "allow_dirty": True}

    first = spec_from_config("configs/experiment/smoke.yaml", **kwargs)
    second = spec_from_config("configs/experiment/smoke.yaml", **kwargs)

    assert first == second


def test_spec_from_config_rejects_an_unknown_platform() -> None:
    with pytest.raises(ValueError, match="unknown platform"):
        spec_from_config("configs/experiment/smoke.yaml", platform="on_prem", allow_dirty=True)


def test_spec_from_config_records_a_relative_config_path() -> None:
    spec = spec_from_config("configs/experiment/smoke.yaml", platform="local_cpu", allow_dirty=True)

    assert not Path(spec.experiment_config).is_absolute()


def test_spec_from_config_requires_a_clean_tree_by_default() -> None:
    """The shared working tree is dirty during wave-2 work, so the default must refuse."""
    from spectraquant.reporting.gitinfo import git_info

    info = git_info()
    if info.dirty is not True:  # pragma: no cover - a clean checkout cannot exercise this branch
        pytest.skip("working tree is clean; the dirty-tree guard has nothing to reject")

    with pytest.raises(ValueError, match="working tree is dirty"):
        spec_from_config("configs/experiment/smoke.yaml", platform="colab")


def test_run_spec_repr_does_not_leak_environment_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_supersecrettokenvalue")
    spec = make_spec(repo_url="https://hf_supersecrettokenvalue@huggingface.co/datasets/x")

    assert "hf_supersecrettokenvalue" not in json.dumps(spec.to_json())
    assert "[REDACTED:HF_TOKEN]" in json.dumps(spec.to_json())
