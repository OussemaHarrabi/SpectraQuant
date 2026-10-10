"""The three allocated arms: calibration capture, proxy scoring, CP-SAT allocation, application.

Every arm here runs offline on the deterministic ``TinyLM`` stand-in (no network, no GPU) and the
configs are the committed frozen ones. The budget ladder is patched on the *loaded* plan — the plan
file itself is frozen and not this slice's to edit — so an arm only ever accepts a budget the plan
declares, and the declared values under test are fixture-sized.

What each test pins (``docs/decisions/ADR-0004-spectraquant-method.md`` is the contract):

* all three arms run end to end and write schema-valid manifests;
* the allocation respects the byte budget (the budget is the *constrained* quantity) and an
  infeasible budget fails loudly, naming the budget and the minimum achievable bytes;
* the calibration capture reads the plan's ``calibration`` role only — never the test split;
* a recorded proxy score **is** the frozen ``score_model`` output for the same inputs;
* the regularizer changes training, not the allocation: the two allocated arms agree per layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from _plan_fixtures import PIN, TinyLM, TinyTokenizer, plan_runner_env
from typer.testing import CliRunner

from spectraquant.allocation import QuantCostModel
from spectraquant.cli.main import app
from spectraquant.cloud import method_arms, plan_runner
from spectraquant.cloud.method_arms import (
    DEFAULT_ALLOC_AXIS,
    InfeasibleAllocationBudget,
    RegularizerCoefficients,
    capture_activations,
    minimum_achievable_bytes,
    module_name_of,
    proxy_score_grid,
    proxy_score_key,
    resolve_proxy_variant,
    resolve_regularizer_coefficients,
    solve_allocation,
)
from spectraquant.cloud.plan_runner import (
    compressible_linear_names,
    resolve_arm_compression,
    run_plan,
)
from spectraquant.experiment_plan import load_plan, plan_path
from spectraquant.proxies.variants import PROXY_VARIANTS
from spectraquant.quantization import accounted_bytes, fake_quantize, measure_serialized_bytes
from spectraquant.reporting.manifests import validate_manifest_file

TIER1_PLAN = "configs/tier1/smollm2_135m.yaml"
ALLOCATED_ARMS = ("proxy_allocated", "spectraquant_regularized")
runner = CliRunner()


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run offline and deterministically (stand-in model, tokenizer, corpora)."""
    plan_runner_env(monkeypatch)


def _fixture_shapes() -> dict[str, tuple[int, int]]:
    model = TinyLM()
    names, _ = compressible_linear_names(model)
    state = model.state_dict()
    return {name: (int(state[name].shape[0]), int(state[name].shape[1])) for name in names}


def _cost_model(group_size: int = 32) -> QuantCostModel:
    """The allocator's cost model as the runner builds it for ``--granularity per_group``."""
    return QuantCostModel(
        granularity="per_group", group_size=group_size, symmetric=True, axis=DEFAULT_ALLOC_AXIS
    )


def _fixture_minimum(group_size: int = 32) -> int:
    """The cheapest achievable allocation for the fixture model under the plan's grid."""
    plan = load_plan(TIER1_PLAN)
    return minimum_achievable_bytes(
        _fixture_shapes(),
        ranks=plan.grid.ranks,
        bits=plan.grid.bits,
        cost_model=_cost_model(group_size),
    )


def _patch_ladder(monkeypatch: pytest.MonkeyPatch, ladder: Sequence[int]) -> None:
    """Patch the budget ladder on the plan :func:`run_plan` loads (the plan file is frozen)."""
    real = load_plan(plan_path(TIER1_PLAN))
    patched = real.model_copy(
        update={"grid": real.grid.model_copy(update={"budget_ladder_bytes": list(ladder)})}
    )
    monkeypatch.setattr(plan_runner, "load_plan", lambda _path: patched)


def _metrics_of(result: object, arm: str) -> dict:
    runs = [run for run in result.runs if run.arm == arm]  # type: ignore[attr-defined]
    assert runs, f"arm {arm!r} did not run"
    return validate_manifest_file(runs[0].manifest_path)["metrics"]


def _allocation_of(metrics: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """``{layer: (rank, bits)}`` from the manifest's flattened allocation block."""
    prefix = "allocation.layer."
    out: dict[str, tuple[int, int]] = {}
    for key, value in metrics.items():
        if not key.startswith(prefix) or not key.endswith(".bits"):
            continue
        layer = key[len(prefix) : -len(".bits")]
        out[layer] = (int(metrics[f"{prefix}{layer}.requested_rank"]), int(value))
    return out


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------
def test_all_three_arms_run_end_to_end_and_write_schema_valid_manifests(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every allocated arm executes on the tiny fixture and writes a validated manifest.

    Two invocations, because the arms are bound differently: ``quant_then_residual`` takes a single
    ``(rank, bits)`` grid point, while the two proxy-allocated arms take a ladder rung and derive the
    per-layer point themselves (they refuse a single point, see
    ``test_the_allocated_arms_refuse_a_single_grid_point``).
    """
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum + 40])

    residual = run_plan(
        TIER1_PLAN,
        arms=["quant_then_residual"],
        rank=4,
        bits=4,
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path / "residual",
        progress=lambda _: None,
    )
    allocated = run_plan(
        TIER1_PLAN,
        arms=list(ALLOCATED_ARMS),
        budget_bytes=minimum + 40,
        regularizer={"lambda_round": 1e-3, "lambda_factor": 1.0},
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path / "allocated",
        progress=lambda _: None,
    )

    assert [run.arm for run in residual.runs] == ["quant_then_residual"]
    assert [run.arm for run in allocated.runs] == list(ALLOCATED_ARMS)

    for arm, result in (
        ("quant_then_residual", residual),
        *[(a, allocated) for a in ALLOCATED_ARMS],
    ):
        manifest = validate_manifest_file(
            next(run.manifest_path for run in result.runs if run.arm == arm)
        )
        metrics = manifest["metrics"]
        assert manifest["measurement_class"] == 2
        assert manifest["model_revision"] == PIN
        assert manifest["compression"]["bytes_source"] == "measured"
        assert manifest["compression"]["serializer"] == "spectraquant-sqpack-v1"
        # Class 1 is the constrained figure, class 3 the serialized one, and they agree by
        # construction (the container is headerless).
        assert metrics["allocation.bytes_class"] == 1
        assert metrics["allocation.measured_bytes_class"] == 3
        assert metrics["allocation.measured_bytes"] == manifest["compression"]["accounted_bytes"]
        assert (
            manifest["compression"]["measured_bytes"] == manifest["compression"]["accounted_bytes"]
        )
        assert metrics["quality.measurement_class"] == 2

    for arm in ALLOCATED_ARMS:
        metrics = _metrics_of(allocated, arm)
        assert metrics["allocation.proxy_value_class"] == 2
        assert metrics["allocation.proxy_context"] == "activations_only"
        assert metrics["allocation.n_candidates"] == (
            len(_fixture_shapes())
            * len(load_plan(TIER1_PLAN).grid.ranks)
            * len(load_plan(TIER1_PLAN).grid.bits)
        )
        # The calibrator's identity is recorded on the arm that used it.
        assert metrics["allocation.calibration.role"] == "calibration"
        assert metrics["allocation.frozen_parameter_bytes"] > 0

    residual_metrics = _metrics_of(residual, "quant_then_residual")
    assert residual_metrics["compression.requested_ranks"] == [4]
    assert residual_metrics["allocation.uniform_bits"] == 4
    # The ADR-0004 section 7 fp16-factor counterpart is recorded beside the allocator's figure.
    assert residual_metrics["allocation.bytes_formula"].startswith("sum over layers")

    regularized = _metrics_of(allocated, "spectraquant_regularized")
    assert regularized["training.steps_completed"] == regularized["training.steps_requested"]
    assert regularized["training.perplexity_is_stored_artifact"] is True
    assert regularized["training.factors_quantized_for_eval"] is True
    assert regularized["training.optimised_loss_includes_penalty"] is True
    assert regularized["regularizer.coefficients_source"] == "cli"
    assert regularized["regularizer.lambda_round"] == pytest.approx(1e-3)
    assert regularized["regularizer.lambda_factor"] == pytest.approx(1.0)
    assert (
        regularized["training.regularizer_steps_recorded"]
        == regularized["training.steps_requested"]
    )
    assert regularized["training.regularizer_terms_mean.rounding"] is not None
    assert regularized["regularizer.references_supplied"] is True

    proxy_allocated = _metrics_of(allocated, "proxy_allocated")
    assert "training.steps_completed" not in proxy_allocated  # eval only
    assert proxy_allocated["allocation.proxy_variant"] == method_arms.DEFAULT_PROXY_VARIANT


def test_eval_only_allocated_arm_quotes_its_own_artifact(
    tmp_path: Path, offline: None, monkeypatch
) -> None:
    """The evaluated weight is the stored artifact (base + quantized factors), not a float surrogate."""
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum + 40])
    result = run_plan(
        TIER1_PLAN,
        arms=["proxy_allocated"],
        budget_bytes=minimum + 40,
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path,
        progress=lambda _: None,
    )
    metrics = _metrics_of(result, "proxy_allocated")

    # Rebuild the artifact independently from the recorded allocation and check that the manifest's
    # error figure is the one the deployed ``Q + Q(B) @ Q(A)`` weights actually have.
    state = TinyLM().state_dict()
    allocation = _allocation_of(metrics)
    assert set(allocation) == set(_fixture_shapes())
    errors: list[float] = []
    for layer, (rank, bits) in allocation.items():
        weight = state[layer]
        spec = _cost_model().spec_for_bits(bits)
        base = method_arms.fake_quantize(weight, spec)
        factors = method_arms.truncated_svd(weight - base, rank)
        effective = base + method_arms.fake_quantize(factors.B, spec) @ method_arms.fake_quantize(
            factors.A, spec
        )
        errors.append(
            float(torch.linalg.matrix_norm(weight - effective) / torch.linalg.matrix_norm(weight))
        )
    assert metrics["compression.relative_fro_mean"] == pytest.approx(sum(errors) / len(errors))
    assert metrics["compression.relative_fro_max"] == pytest.approx(max(errors))
    assert int(metrics["compression.n_tensors"]) == len(allocation)


# --------------------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------------------
def test_the_allocation_respects_the_budget_and_the_tight_rung_binds(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The constrained quantity is the reported quantity, and a tight rung really constrains.

    The cheapest option for the fixture costs exactly the minimum, so a budget set to the minimum
    must produce that allocation (not the error-minimising one) — otherwise the budget would not be
    doing anything and the "respects the budget" claim would be vacuous.
    """
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum, minimum + 40])
    tight = run_plan(
        TIER1_PLAN,
        arms=["proxy_allocated"],
        budget_bytes=minimum,
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path / "tight",
        progress=lambda _: None,
    )
    loose = run_plan(
        TIER1_PLAN,
        arms=["proxy_allocated"],
        budget_bytes=minimum + 40,
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path / "loose",
        progress=lambda _: None,
    )

    for result in (tight, loose):
        metrics = _metrics_of(result, "proxy_allocated")
        achieved = int(metrics["allocation.achieved_bytes"])
        assert (
            achieved
            <= int(metrics["allocation.budget_bytes"])
            == int(metrics["allocation.budget_bytes"])
        )
        # The reported class-1 bytes are the constrained total, and the container matches it.
        assert achieved == int(metrics["allocation.measured_bytes"])
        assert int(metrics["allocation.budget_headroom_bytes"]) >= 0

    tight_metrics = _metrics_of(tight, "proxy_allocated")
    # The achieved total equals the minimum, so *every* layer sits at a cheapest option: the budget
    # is binding. (2 bits and 4 bits cost the same on these 8-wide fixture rows, because the payload
    # is padded to a 32-bit word either way, so the solver's tie-break picks the lower-error one.)
    assert int(tight_metrics["allocation.achieved_bytes"]) == minimum
    assert {bits for _, bits in _allocation_of(tight_metrics).values()} <= {2, 4}
    assert int(tight_metrics["allocation.budget_ladder_rung_bytes"]) == minimum
    # A looser rung spends strictly more bytes for a strictly lower predicted error: the budget is
    # the lever, not a formality.
    loose_metrics = _metrics_of(loose, "proxy_allocated")
    assert int(loose_metrics["allocation.achieved_bytes"]) > minimum
    assert int(loose_metrics["allocation.predicted_error"]) < int(
        tight_metrics["allocation.predicted_error"]
    )


def test_an_infeasible_budget_fails_loudly_naming_both_numbers(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget below the minimum achievable allocation is a hard failure, never a shrink.

    The plan's own ladder is patched to an infeasible rung, so the run fails for the *arithmetic*
    reason (nothing fits) rather than because the value was undeclared.
    """
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum - 1])

    with pytest.raises(InfeasibleAllocationBudget) as excinfo:
        run_plan(
            TIER1_PLAN,
            arms=["proxy_allocated"],
            budget_bytes=minimum - 1,
            granularity="per_group",
            group_size=32,
            seq_len=8,
            out_dir=tmp_path,
            progress=lambda _: None,
        )

    message = str(excinfo.value)
    assert "infeasible" in message
    assert str(minimum - 1) in message and str(minimum) in message
    assert excinfo.value.budget_bytes == minimum - 1
    assert excinfo.value.minimum_bytes == minimum
    assert not list(tmp_path.glob("**/run_manifest.json")), "a failed allocation writes no manifest"


def test_a_budget_outside_the_plans_ladder_is_refused(tmp_path: Path, offline: None) -> None:
    """A budget the plan never declared cannot be used: the ladder is the plan's binding."""
    with pytest.raises(ValueError, match="predeclared ladder"):
        run_plan(
            TIER1_PLAN,
            arms=["proxy_allocated"],
            budget_bytes=7,
            progress=lambda _: None,
            out_dir=tmp_path,
        )


# --------------------------------------------------------------------------------------
# The calibration capture never touches the test split
# --------------------------------------------------------------------------------------
def test_the_capture_reads_the_calibration_role_only(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof that the test split never reaches the capture, on three independent levels.

    1. The *loads*: ``load_plan_texts`` is wrapped and every ``(dataset, split)`` it is asked for is
       recorded. The calibration read is the plan's ``calibration`` role (``allenai/c4``, split
       ``train``); the perplexity read is the ``test_perplexity`` role (``EleutherAI/wikitext_*``,
       split ``test``).
    2. The *arguments*: ``capture_activations`` is spied on and the documents it receives carry the
       calibration marker only — no test-split document is ever handed to it.
    3. The *record*: the manifest's calibration block names the calibration dataset, role and split
       and states ``test_split_read: false``, while the perplexity block names the test dataset.
    """
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum + 40])

    loads: list[tuple[str, str]] = []
    real_load = plan_runner.load_plan_texts

    def _corpus(marker: str) -> list[str]:
        return [" ".join(f"{marker} w{index}" for index in range(12)) for _ in range(3)]

    def spy_load(ref, **kwargs):
        loads.append((str(ref.name), str(kwargs.get("split"))))
        return _corpus("CALIB" if "c4" in ref.name else "TEST-SPLIT")

    seen_documents: list[list[str]] = []
    real_capture = plan_runner.capture_activations

    def spy_capture(model, tokenizer, documents, **kwargs):
        seen_documents.append(list(documents))
        return real_capture(model, tokenizer, documents, **kwargs)

    monkeypatch.setattr(plan_runner, "load_plan_texts", spy_load)
    monkeypatch.setattr(plan_runner, "capture_activations", spy_capture)
    assert real_load is not None

    result = run_plan(
        TIER1_PLAN,
        arms=["proxy_allocated"],
        budget_bytes=minimum + 40,
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path,
        progress=lambda _: None,
    )

    plans = load_plan(TIER1_PLAN)
    calibration_dataset = next(
        dataset.name for dataset in plans.datasets if "calibration" in dataset.roles
    )
    test_dataset = next(
        dataset.name for dataset in plans.datasets if "test_perplexity" in dataset.roles
    )

    assert (calibration_dataset, "train") in loads
    assert (test_dataset, "test") in loads
    assert seen_documents, "the capture must have run"
    for corpus in seen_documents:
        assert corpus and all(document.startswith("CALIB") for document in corpus)
        assert not any("TEST-SPLIT" in document for document in corpus)

    metrics = _metrics_of(result, "proxy_allocated")
    assert metrics["allocation.calibration.role"] == "calibration"
    assert metrics["allocation.calibration.dataset"] == calibration_dataset
    assert metrics["allocation.calibration.split"] == "train"
    assert metrics["allocation.calibration.test_split_read"] is False
    assert metrics["allocation.calibration.batch_checksum"].startswith("sha256:")
    assert metrics["allocation.calibration.documents_sha256"].startswith("sha256:")
    assert metrics["allocation.calibration.documents"] == 3
    assert metrics["dataset.name"] == test_dataset
    assert metrics["dataset.split"] == "test"


def test_the_capture_is_bounded_by_its_documented_caps() -> None:
    """The cap is enforced, not decorative: retention stops at ``max_samples_per_layer``."""
    model = TinyLM()
    names, _ = compressible_linear_names(model)
    weights = {name: model.state_dict()[name] for name in names}
    tokenizer = TinyTokenizer()
    documents = [" ".join(f"token{index}" for index in range(12)) for _ in range(4)]

    captured = capture_activations(
        model,
        tokenizer,
        documents,
        weight_names=sorted(weights),
        seq_len=8,
        max_samples_per_layer=8,
        max_documents=4,
    )

    assert captured.n_samples == 8
    provenance = captured.metrics()
    assert provenance["allocation.calibration.max_samples_per_layer"] == 8
    assert provenance["allocation.calibration.samples_per_layer"] == 8
    assert provenance["allocation.calibration.windows"] >= 1
    assert provenance["allocation.calibration.forward_batches"] >= 1
    for inputs in captured.layers.values():
        assert int(inputs.activations.shape[0]) == 8
    # The retention is the cap times the layers' input widths, and it is recorded.
    assert provenance["allocation.calibration.activation_bytes"] == sum(
        int(inputs.activations.numel()) * 4 for inputs in captured.layers.values()
    )


# --------------------------------------------------------------------------------------
# The recorded proxy score is the proxy's own output
# --------------------------------------------------------------------------------------
def test_recorded_proxy_scores_equal_a_direct_score_model_call(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest's score table is the frozen ``score_model`` output, not a look-alike.

    The capture is rebuilt from the same model, tokenizer, corpus and caps, and every recorded
    ``(layer, rank, bits)`` score is compared with a direct single-layer ``score_model`` call.
    """
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum + 40])
    monkeypatch.setattr(
        plan_runner,
        "load_plan_texts",
        lambda ref, **kwargs: [" ".join(f"t{i}" for i in range(12))] * 2,
    )

    result = run_plan(
        TIER1_PLAN,
        arms=["proxy_allocated"],
        budget_bytes=minimum + 40,
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path,
        progress=lambda _: None,
    )
    metrics = _metrics_of(result, "proxy_allocated")

    model = TinyLM(commit=PIN)
    names, _ = compressible_linear_names(model)
    weights = {name: model.state_dict()[name].detach().clone() for name in names}
    capture = capture_activations(
        model,
        TinyTokenizer(),
        [" ".join(f"t{i}" for i in range(12))] * 2,
        weight_names=sorted(weights),
        seq_len=8,
    )
    plan = load_plan(TIER1_PLAN)
    cost_model = _cost_model()
    proxy = resolve_proxy_variant(None)

    checked = 0
    for layer in sorted(weights):
        for rank in plan.grid.ranks:
            for bits in plan.grid.bits:
                direct = proxy.score_model(
                    {layer: capture.layers[layer]},
                    {layer: int(rank)},
                    {layer: cost_model.spec_for_bits(int(bits))},
                ).value
                assert metrics[proxy_score_key(layer, rank, bits)] == pytest.approx(direct)
                checked += 1
    assert checked == len(weights) * len(plan.grid.ranks) * len(plan.grid.bits)
    # And the grid the manifest reports is the one the solver optimised (same proxy, same scores).
    assert metrics["allocation.n_candidates"] == checked
    assert metrics["allocation.proxy_variant"] == "gain_aware_composed"


def test_the_proxy_can_be_overridden_and_the_choice_is_recorded(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit variant from ``PROXY_VARIANTS`` is honoured and recorded; an unknown one refused."""
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum + 40])
    result = run_plan(
        TIER1_PLAN,
        arms=["proxy_allocated"],
        budget_bytes=minimum + 40,
        proxy="weight_frobenius",
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path,
        progress=lambda _: None,
    )
    metrics = _metrics_of(result, "proxy_allocated")
    assert metrics["allocation.proxy_variant"] == "weight_frobenius"
    assert metrics["allocation.proxy_value_class"] == 1  # that variant is analytical

    with pytest.raises(ValueError, match="unknown proxy"):
        run_plan(
            TIER1_PLAN,
            arms=["proxy_allocated"],
            budget_bytes=minimum + 40,
            proxy="not-a-variant",
            progress=lambda _: None,
            out_dir=tmp_path / "bad",
        )
    assert PROXY_VARIANTS["gain_aware_composed"] is not None


# --------------------------------------------------------------------------------------
# Allocation invariance and the regularizer
# --------------------------------------------------------------------------------------
def test_both_allocated_arms_agree_on_the_allocation_for_one_seed(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regularizer changes training, not the allocation.

    Both arms run in one invocation (one seed, one calibration capture, one budget), so any
    difference in the recorded per-layer allocation would be a defect; the quality numbers must
    nevertheless differ, because the second arm also trains.
    """
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum + 40])
    result = run_plan(
        TIER1_PLAN,
        arms=list(ALLOCATED_ARMS),
        budget_bytes=minimum + 40,
        regularizer={"lambda_round": 5e-3},
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path,
        progress=lambda _: None,
    )

    untrained = _metrics_of(result, "proxy_allocated")
    trained = _metrics_of(result, "spectraquant_regularized")
    assert _allocation_of(untrained) == _allocation_of(trained)
    assert (
        untrained["allocation.proxy_score.head.weight.r2b2"]
        == trained["allocation.proxy_score.head.weight.r2b2"]
    )
    assert untrained["allocation.achieved_bytes"] == trained["allocation.achieved_bytes"]
    assert untrained["perplexity"] != trained["perplexity"], "the method arm did train"
    assert float(trained["training.regularizer_terms_mean.total"]) > 0.0

    # A second invocation at the same seed reproduces the allocation, so the claim is not an artefact
    # of the two arms sharing one capture inside one process.
    with_seed = run_plan(
        TIER1_PLAN,
        arms=["proxy_allocated"],
        budget_bytes=minimum + 40,
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path / "again",
        progress=lambda _: None,
    )
    assert _allocation_of(_metrics_of(with_seed, "proxy_allocated")) == _allocation_of(untrained)


def test_the_regularizer_coefficients_come_from_the_plan_when_it_declares_them() -> None:
    """A plan-level block is read when present; an explicit override wins; the default is all-zero."""
    plan = SimpleNamespace(
        regularizer={"lambda_round": 0.25, "lambda_spectrum": 0.5, "tail_rank": 4}
    )
    coefficients, source = resolve_regularizer_coefficients(plan)
    assert source == "plan"
    assert coefficients == RegularizerCoefficients(
        lambda_round=0.25, lambda_spectrum=0.5, tail_rank=4
    )

    overridden, source = resolve_regularizer_coefficients(plan, {"lambda_round": 1.0})
    assert source == "plan+cli"
    assert overridden.lambda_round == 1.0 and overridden.lambda_spectrum == 0.5

    default, source = resolve_regularizer_coefficients(SimpleNamespace(), None)
    assert source == "defaults"
    assert default.is_noop() and default.weights() == {
        "lambda_factor": 0.0,
        "lambda_round": 0.0,
        "lambda_residual": 0.0,
        "lambda_spectrum": 0.0,
    }

    with pytest.raises(ValueError, match="unknown regularizer coefficient"):
        resolve_regularizer_coefficients(SimpleNamespace(), {"lambda_nope": 1.0})
    with pytest.raises(ValueError, match="tail_rank"):
        resolve_regularizer_coefficients(SimpleNamespace(), {"lambda_spectrum": 1.0})
    with pytest.raises(ValueError, match="tail_rank must be None"):
        resolve_regularizer_coefficients(SimpleNamespace(), {"tail_rank": 4})


def test_a_no_op_regularizer_is_recorded_as_such(
    tmp_path: Path, offline: None, monkeypatch
) -> None:
    """Zero coefficients mean no penalty was evaluated — and the manifest says so, with zeros."""
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum + 40])
    result = run_plan(
        TIER1_PLAN,
        arms=["spectraquant_regularized"],
        budget_bytes=minimum + 40,
        granularity="per_group",
        group_size=32,
        seq_len=8,
        out_dir=tmp_path,
        progress=lambda _: None,
    )
    metrics = _metrics_of(result, "spectraquant_regularized")
    assert metrics["regularizer.is_noop"] is True
    assert metrics["regularizer.coefficients_source"] == "defaults"
    assert metrics["training.regularizer_applied"] is False
    assert metrics["training.regularizer_steps_recorded"] == 0
    assert metrics["training.regularizer_terms_mean.rounding"] is None
    assert metrics["training.steps_completed"] == metrics["training.steps_requested"]


# --------------------------------------------------------------------------------------
# Resolution, solver plumbing and the CLI surface
# --------------------------------------------------------------------------------------
def test_the_allocated_arms_refuse_a_single_grid_point() -> None:
    """A ``--rank``/``--bits`` for an allocated arm would be ignored, so it is refused."""
    plan = load_plan(TIER1_PLAN)
    arm = next(arm for arm in plan.arms if arm.name == "proxy_allocated")
    resolved = resolve_arm_compression(plan, arm, granularity="per_group", group_size=32)
    assert resolved.bits is None and resolved.rank is None
    assert resolved.group_size == 32 and resolved.axis == DEFAULT_ALLOC_AXIS

    with pytest.raises(ValueError, match="per layer"):
        resolve_arm_compression(plan, arm, bits=4, granularity="per_group", group_size=32)
    with pytest.raises(ValueError, match="per layer"):
        resolve_arm_compression(plan, arm, rank=4, granularity="per_group", group_size=32)


def test_the_solver_is_deterministic_for_one_seed_and_reports_its_status() -> None:
    """One solve is reproducible: the same option table and seed give the same allocation."""
    shapes = _fixture_shapes()
    plan = load_plan(TIER1_PLAN)
    cost_model = _cost_model()
    model = TinyLM()
    weights = {name: model.state_dict()[name].detach().clone() for name in shapes}
    capture = capture_activations(
        model,
        TinyTokenizer(),
        [" ".join(f"t{i}" for i in range(12))] * 2,
        weight_names=sorted(weights),
        seq_len=8,
    )
    grid = proxy_score_grid(
        resolve_proxy_variant(None),
        capture.layers,
        ranks=plan.grid.ranks,
        bits=plan.grid.bits,
        cost_model=cost_model,
    )
    budget = minimum_achievable_bytes(
        shapes, ranks=plan.grid.ranks, bits=plan.grid.bits, cost_model=cost_model
    )
    first = solve_allocation(
        arm="proxy_allocated",
        layer_shapes=shapes,
        ranks=plan.grid.ranks,
        bits=plan.grid.bits,
        budget_bytes=budget + 60,
        cost_model=cost_model,
        error_fn=grid.error_fn,
        seed=3,
    )
    second = solve_allocation(
        arm="proxy_allocated",
        layer_shapes=shapes,
        ranks=plan.grid.ranks,
        bits=plan.grid.bits,
        budget_bytes=budget + 60,
        cost_model=cost_model,
        error_fn=grid.error_fn,
        seed=3,
    )
    assert first.as_assignment() == second.as_assignment()
    assert first.accounted_bytes == second.accounted_bytes
    assert first.diagnostics["ortools_status_code"] == 4.0  # CP-SAT OPTIMAL
    assert first.predicted_error == pytest.approx(
        grid.error_fn("head.weight", *first.per_layer["head.weight"])
        + grid.error_fn("proj.weight", *first.per_layer["proj.weight"])
    )


def test_the_cli_exposes_the_allocation_options(tmp_path: Path, offline: None, monkeypatch) -> None:
    """The explicit CLI surface: a budget rung plus the regularizer coefficients reach the runner."""
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum + 40])
    result = runner.invoke(
        app,
        [
            "run-plan",
            "--plan",
            TIER1_PLAN,
            "--arms",
            "spectraquant_regularized",
            "--budget-bytes",
            str(minimum + 40),
            "--granularity",
            "per_group",
            "--group-size",
            "32",
            "--lambda-round",
            "0.002",
            "--seq-len",
            "8",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = validate_manifest_file(
        tmp_path / "spectraquant_regularized" / "seed-0" / "run_manifest.json"
    )
    metrics = manifest["metrics"]
    assert metrics["regularizer.lambda_round"] == pytest.approx(0.002)
    assert metrics["regularizer.coefficients_source"] == "cli"
    assert metrics["allocation.budget_bytes"] == minimum + 40
    assert metrics["allocation.achieved_bytes"] <= metrics["allocation.budget_bytes"]


def test_an_infeasible_budget_exits_non_zero_through_the_cli(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loud failure, not a traceback-only crash: the CLI reports it and exits non-zero."""
    minimum = _fixture_minimum()
    _patch_ladder(monkeypatch, [minimum - 1])
    result = runner.invoke(
        app,
        [
            "run-plan",
            "--plan",
            TIER1_PLAN,
            "--arms",
            "proxy_allocated",
            "--budget-bytes",
            str(minimum - 1),
            "--granularity",
            "per_group",
            "--group-size",
            "32",
            "--seq-len",
            "8",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "infeasible" in result.output
    assert str(minimum) in result.output


def test_the_artifact_bytes_are_the_accounting_sum_and_the_container_matches() -> None:
    """Class 1 is the accounting sum over the stored tensors; class 3 is the container we wrote.

    The two are the same number by construction (the SpectraQuant container is headerless), so a
    divergence here would mean the reported byte figure is not the artifact's.
    """
    torch.manual_seed(0)
    weights = {"a.weight": torch.randn(8, 8), "b.weight": torch.randn(11, 8)}
    per_layer = {"a.weight": (4, 4), "b.weight": (2, 8)}
    cost_model = _cost_model()
    artifact = method_arms.build_residual_artifact(weights, per_layer, cost_model)

    expected = sum(
        accounted_bytes(tuple(tensor.shape), artifact.storage_specs[key])
        for key, tensor in artifact.storage.items()
    )
    assert artifact.accounted == expected
    assert measure_serialized_bytes(artifact.storage, artifact.storage_specs) == expected
    assert artifact.measured == expected
    # The deployed weight is the quantized base plus the *quantized* factors, and the factors are on
    # their grid (the objective's rounding term is defined on exactly that grid).
    stored_a, stored_b = artifact.stored_factors["a.weight"]
    spec_a = cost_model.spec_for_bits(4)
    assert torch.equal(stored_a, fake_quantize(stored_a, spec_a))
    assert torch.equal(stored_b, fake_quantize(stored_b, spec_a))
    assert torch.allclose(
        artifact.state["a.weight"], artifact.bases["a.weight"] + stored_b @ stored_a
    )
    # Byte-comparability of the three arms rests on every one of them using this same formula.
    assert artifact.metrics["allocation.bytes_formula"].startswith("sum over layers")
    assert module_name_of("model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj"
    )


def test_a_preparation_penalty_is_differentiable_and_zero_when_ablated() -> None:
    """The four terms reach A and B, and an all-zero objective contributes an exact zero."""
    torch.manual_seed(0)
    a = torch.randn(2, 6, requires_grad=True)
    b = torch.randn(5, 2, requires_grad=True)
    reference = torch.randn(5, 6)
    spec = _cost_model().spec_for_bits(4)

    active = method_arms.preparation_penalty(
        {"head.weight": (a, b)},
        {"head.weight": spec},
        RegularizerCoefficients(lambda_factor=1.0, lambda_round=1e-3),
        references={"head.weight": reference},
        name="test",
    )
    assert active.total.requires_grad
    active.total.backward()
    assert a.grad is not None and b.grad is not None and float(a.grad.abs().sum()) > 0.0

    off = method_arms.preparation_penalty(
        {"head.weight": (a, b)},
        {"head.weight": spec},
        RegularizerCoefficients(),
    )
    assert off.is_noop and float(off.total) == 0.0 and off.groups == {}
    assert all(value == 0.0 for value in off.terms.values())
