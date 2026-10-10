"""The eight frozen M3 arms end to end: shared ``Q``, the lr search, the step size, isolation.

Everything here runs **offline** on a tiny synthetic fixture (no network, no GPU): the runner's
model/dataset loaders are patched to a small causal LM, and the plan is the *real*
``configs/m3/tier1_smollm2_135m.yaml`` with only its optimisation schedule shrunk (the arms, their
points, the grids and the quantizers are the frozen ones). The point of the module is the wiring
``docs/decisions/design-m3-arms.md`` requires and the frozen invariants of
``docs/research/reproduction-plan.md`` §5.4:

* every arm writes a schema-valid manifest with its own compression block and training provenance;
* the three R1 2-bit arms share a bit-identical quantized base ``Q``;
* the learning-rate search records every candidate's **dev** perplexity and reads no test split;
* ``step_size_lr = 0`` leaves ``s`` bit-identical, ``step_size_lr > 0`` moves it;
* no state leaks between arms.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from torch import Tensor, nn

from spectraquant.cloud import plan_data, plan_runner
from spectraquant.cloud.plan_runner import (
    LR_SEARCH_MAX_STEPS,
    M3_ARM_KINDS,
    _codebook_loftq_pair,
    _divergence_trigger,
    _search_prefix_steps,
    apply_arm,
    resolve_arm_compression,
    run_plan,
)
from spectraquant.experiment_plan import load_plan
from spectraquant.factorization import truncated_svd
from spectraquant.quantization import CodebookSpec, QuantSpec, fake_quantize, fake_quantize_codebook
from spectraquant.reporting.manifests import validate_manifest_file
from spectraquant.training.low_rank import (
    QuantizedPlusLowRankLinear,
    StraightThroughQuantizedLinear,
)

M3_PLAN = "configs/m3/tier1_smollm2_135m.yaml"

#: The frozen M3 plan's model revision (the fixture model reports it as its ``_commit_hash``).
PIN = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

#: Fixture vocabulary / width. Every linear layer is ``width x width`` (or ``width x vocab`` with
#: ``vocab == width``), so the plan's rank grid ``{16, 32}`` fits without clamping: the M3 arms are
#: *model-level* arms whose rank must be admissible on every target layer.
VOCAB = 64
WIDTH = 64
CONTEXT = 32

#: Linears the fixture model owns (``proj`` and ``head``), i.e. the compressible target set.
TARGETS = 2


class MiniLM(nn.Module):
    """A deterministic two-linear causal-LM stand-in big enough for rank 32 at group size 128."""

    def __init__(self, *, commit: str | None = PIN, seed: int = 7) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.head = nn.Linear(WIDTH, VOCAB)
        config = SimpleNamespace(max_position_embeddings=CONTEXT)
        if commit is not None:
            config._commit_hash = commit
        self.config = config

    def forward(self, ids: Tensor) -> Any:
        hidden = torch.tanh(self.proj(self.embed(ids)))
        return SimpleNamespace(logits=self.head(hidden))


class MiniTokenizer:
    """A byte-ish tokenizer whose ids stay inside the fixture vocabulary."""

    def __init__(self, *, tokens_per_document: int = CONTEXT) -> None:
        self.tokens_per_document = tokens_per_document
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text: str, return_tensors: str = "pt") -> Any:
        assert return_tensors == "pt"
        length = max(2, min(self.tokens_per_document, CONTEXT))
        ids = torch.arange(length, dtype=torch.long).unsqueeze(0) % VOCAB
        return SimpleNamespace(input_ids=ids)


def _documents(count: int = 3) -> list[str]:
    return [" ".join(f"token{index}" for index in range(CONTEXT)) for _ in range(count)]


def _fixture_corpus(
    config: Any,
    tokenizer: Any,
    *,
    seq_len: int,
    max_documents: Any,
    context: Any = None,
    device: Any = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """A fixture-sized training corpus (the plan's 512-token windows exceed the fixture's context)."""
    from spectraquant.training.loop import SequenceData

    generator = torch.Generator().manual_seed(int(config.seeds.master))
    train = torch.randint(0, VOCAB, (8, CONTEXT), generator=generator, dtype=torch.int64)
    val = torch.randint(0, VOCAB, (2, CONTEXT), generator=generator, dtype=torch.int64)
    provenance = {
        "training.train_dataset": "fixture",
        "training.val_dataset": "fixture",
        "training.train_split": "train",
        "training.val_split": "validation",
        "training.train_windows": int(train.shape[0]),
        "training.val_windows": int(val.shape[0]),
        "training.seq_len": CONTEXT - 1,
        "training.corpus_checksum": "sha256:fixture",
        "training.split_rule": (
            "fixture stand-in: synthetic train/dev tensors; the test split is never read"
        ),
    }
    return SequenceData(train=train, val=val, checksum="sha256:fixture"), provenance


def _quant_spec() -> QuantSpec:
    """The frozen R2 quantizer: uniform 4-bit symmetric, per-group g128."""
    return QuantSpec(bits=4, granularity="per_group", group_size=128, symmetric=True)


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the loaders so the runner runs offline and deterministically."""

    class _Loader:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs: Any) -> MiniLM:
            kwargs.pop("dtype", None)
            kwargs.pop("torch_dtype", None)
            return MiniLM()

    class _TokenizerLoader:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs: Any) -> MiniTokenizer:
            return MiniTokenizer()

    fake = SimpleNamespace(AutoModelForCausalLM=_Loader(), AutoTokenizer=_TokenizerLoader())
    monkeypatch.setattr(plan_data, "require_transformers", lambda: fake)
    monkeypatch.setattr(plan_runner, "require_transformers", lambda: fake)
    monkeypatch.setattr(
        plan_runner, "verify_pinned_revision", lambda repo_id, revision, *, repo_type: revision
    )
    monkeypatch.setattr(plan_runner, "load_plan_texts", lambda ref, **kwargs: _documents())
    monkeypatch.setattr(plan_runner, "_training_corpus", _fixture_corpus)


def _small_plan(tmp_path: Path, *, steps: int = 4, grid: list[float] | None = None) -> Path:
    """Write the frozen M3 plan with only its *schedule* shrunk, for a cheap fixture run."""
    config = load_plan(M3_PLAN)
    schedule = config.training
    assert schedule is not None
    update: dict[str, Any] = {
        "steps": steps,
        "warmup_steps": 1,
        "batch_size": 2,
        "eval_every": 0,
        "checkpoint_every": 0,
    }
    if grid is not None:
        update["learning_rate_grid"] = grid
    shrunk = config.model_copy(update={"training": schedule.model_copy(update=update)})
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "m3_fixture_plan.yaml"
    path.write_text(
        yaml.safe_dump(shrunk.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------------------
# Arm coverage: every frozen arm runs end to end and writes its own schema-valid manifest
# --------------------------------------------------------------------------------------
def test_every_frozen_m3_arm_runs_and_writes_a_schema_valid_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _offline(monkeypatch)
    plan_file = _small_plan(tmp_path)

    result = run_plan(
        plan_file,
        out_dir=tmp_path / "runs",
        seq_len=8,
        max_tokens=8,
        progress=lambda _: None,
    )

    assert result.skipped == ()
    assert {run.arm for run in result.runs} == {
        "r1_fp16_lora",
        "r1_std_2bit",
        "r1_loftq_2bit",
        "r1_loftq_2bit_t1",
        "r2_fp16",
        "r2_rtn_4bit",
        "r2_lrqat_4bit",
        "r2_fullqat_4bit",
    }
    assert {run.kind for run in result.runs} == M3_ARM_KINDS
    assert len({run.run_id for run in result.runs}) == len(result.runs)

    by_kind = {run.kind: run for run in result.runs}
    # Each arm's own compression block: method from the schema's enum, and the arm's own point.
    expected_method = {
        "r1_fp16_lora": "none",
        "r1_std_2bit": "rtn",
        "r1_loftq_2bit": "loftq",
        "r1_loftq_2bit_t1": "loftq",
        "r2_fp16": "none",
        "r2_rtn_4bit": "rtn",
        "r2_lrqat_4bit": "lr-qat",
        "r2_fullqat_4bit": "lr-qat",
    }
    expected_bits = {
        "r1_fp16_lora": None,
        "r1_std_2bit": 2,
        "r1_loftq_2bit": 2,
        "r1_loftq_2bit_t1": 2,
        "r2_fp16": None,
        "r2_rtn_4bit": 4,
        "r2_lrqat_4bit": 4,
        "r2_fullqat_4bit": 4,
    }
    for kind, run in by_kind.items():
        manifest = validate_manifest_file(run.manifest_path)
        compression = manifest["compression"]
        assert compression["method"] == expected_method[kind], kind
        assert compression["bits"] == expected_bits[kind], kind
        assert compression["accounted_bytes"] is not None and compression["accounted_bytes"] > 0
        metrics = manifest["metrics"]
        assert metrics["plan.arm_kind"] == kind
        assert metrics["plan.arm"] == run.arm
        # Training provenance: the schedule used and the steps completed are recorded for the
        # trained arms; the eval-only arms record the zero-step block.
        if kind in {"r2_fp16", "r2_rtn_4bit"}:
            assert manifest["training"]["steps"] == 0
        else:
            assert manifest["training"]["steps"] == 4
            assert metrics["training.steps_completed"] == metrics["training.steps_requested"]
            assert metrics["training.corpus_checksum"] == "sha256:fixture"

    # The two eval-only references and the trained fp16 LoRA reference carry no compression class.
    assert by_kind["r2_fp16"].measurement_class is None
    assert by_kind["r1_fp16_lora"].measurement_class is None
    assert by_kind["r2_rtn_4bit"].measurement_class == 2


def test_the_r2_rtn_reference_measures_its_container_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eval-only PTQ baseline is a real packed container (class 3), like ``ptq_uniform``."""
    _offline(monkeypatch)
    plan_file = _small_plan(tmp_path)
    result = run_plan(
        plan_file,
        arms=["r2_rtn_4bit"],
        out_dir=tmp_path / "runs",
        seq_len=8,
        progress=lambda _: None,
    )
    manifest = validate_manifest_file(result.runs[0].manifest_path)
    compression = manifest["compression"]
    assert compression["bytes_source"] == "measured"
    assert compression["measured_bytes"] == compression["accounted_bytes"]
    assert compression["serializer"] == "spectraquant-sqpack-v1"
    assert compression["group_size"] == 128
    assert manifest["packed_bytes"] == compression["measured_bytes"]


def test_the_lrqat_arm_carries_a_trainable_step_size_and_a_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``r2_lrqat_4bit``'s layer has the LR-QAT parameter set and can fold into the merged weight."""
    _offline(monkeypatch)
    plan = load_plan(_small_plan(tmp_path, grid=[1.0e-4]))
    arm = next(arm for arm in plan.arms if arm.kind == "r2_lrqat_4bit")
    compression = resolve_arm_compression(plan, arm)
    assert compression.rank == 32 and compression.bits == 4 and compression.group_size == 128
    assert compression.step_size_lr == 1.0e-5

    weights = {"proj.weight": torch.randn(WIDTH, WIDTH), "head.weight": torch.randn(VOCAB, WIDTH)}
    applied = apply_arm(weights, arm, compression, seed=0)
    init = applied.trainable_init
    assert init is not None and init.step_sizes and init.spec is not None

    linear = nn.Linear(WIDTH, WIDTH)
    layer = QuantizedPlusLowRankLinear.from_linear(
        linear,
        2,
        base=torch.randn(WIDTH, WIDTH),
        factors=(torch.randn(2, WIDTH), torch.randn(WIDTH, 2)),
        step_size=0.05,
    )
    assert {name for name, _ in layer.named_parameters()} == {"A", "B", "bias", "step_size"}
    assert layer.has_step_size is True
    codes, merged = layer.merged_weight()
    assert codes.dtype == torch.int8
    assert merged.shape == (WIDTH, WIDTH)


# --------------------------------------------------------------------------------------
# The shared-Q invariant across the three R1 2-bit arms
# --------------------------------------------------------------------------------------
def test_the_three_r1_two_bit_arms_share_the_quantized_base(tmp_path: Path) -> None:
    """``r1_std_2bit`` and both LoftQ arms must see a bit-identical 2-bit base ``Q``.

    ``Q`` is ``fake_quantize_codebook(W, CodebookSpec(bits=2, block_size=64, kind="nf"))`` for all
    three (``docs/decisions/design-m3-arms.md`` §2/§4.1: "the same quantized matrix ``Q``", "the base
    must be identical across them"), so the arms differ **only** in the adapter initialisation - which
    is exactly what trend T2 tests. The LoftQ schedule's internal base ``Q_t = q(W - B @ A)`` is not
    stored, for the same reason.
    """
    plan = load_plan(M3_PLAN)
    spec = CodebookSpec(bits=2, kind="nf", block_size=64)
    torch.manual_seed(0)
    weights = {"proj.weight": torch.randn(WIDTH, WIDTH), "head.weight": torch.randn(VOCAB, WIDTH)}
    reference_q = {name: fake_quantize_codebook(w, spec) for name, w in weights.items()}

    arms = {arm.kind: arm for arm in plan.arms}
    applied = {}
    for kind in ("r1_std_2bit", "r1_loftq_2bit", "r1_loftq_2bit_t1"):
        arm = arms[kind]
        compression = resolve_arm_compression(plan, arm)
        assert compression.quantizer == "codebook"
        assert compression.block_size == 64
        applied[kind] = apply_arm(weights, arm, compression, seed=0)

    # Every one of the three stores Q bit for bit - the shared-base control the design requires.
    for kind in applied:
        init = applied[kind].trainable_init
        assert init is not None
        for name in weights:
            assert torch.equal(init.bases[name], reference_q[name]), (kind, name)

    # ... and they genuinely differ in the adapter initialisation alone.
    std = applied["r1_std_2bit"].trainable_init
    t1 = applied["r1_loftq_2bit_t1"].trainable_init
    t5 = applied["r1_loftq_2bit"].trainable_init
    assert std is not None and t1 is not None and t5 is not None
    assert std.iterations is None and t1.iterations == 1 and t5.iterations == 5
    assert bool((std.factors["proj.weight"][1] == 0).all()), "the standard init starts from B = 0"
    assert not torch.equal(std.factors["proj.weight"][0], t1.factors["proj.weight"][0])
    assert not torch.equal(t1.factors["proj.weight"][0], t5.factors["proj.weight"][0])

    # The std arm's adapter is the Kaiming A / zero B pair; the LoftQ arms' is the residual's SVD.
    first_base = fake_quantize_codebook(weights["proj.weight"], spec)
    closed_form = truncated_svd(weights["proj.weight"] - first_base, 16)
    assert torch.equal(t1.factors["proj.weight"][0], closed_form.A)

    # The LoftQ initialisation must beat the base alone (the T2-a mechanism, in the arm).
    for kind in ("r1_std_2bit", "r1_loftq_2bit", "r1_loftq_2bit_t1"):
        metrics = applied[kind].metrics
        base_error = metrics["compression.initial_base_relative_fro_mean"]
        residual_error = metrics["compression.initial_residual_relative_fro_mean"]
        if kind != "r1_std_2bit":
            assert residual_error < base_error, kind
    assert (
        applied["r1_std_2bit"].metrics["compression.shared_base_rule"]
        == "fake_quantize_codebook(W, spec)"
    )


def test_the_codebook_loftq_pair_matches_the_frozen_algorithm() -> None:
    """The runner's NF2 schedule is the paper's Algorithm 1 from ``A_0 = B_0 = 0``.

    Checked against computations that do **not** share its code path: ``T = 1`` must equal
    ``SVD_rank(W - q_NF2(W))`` (the closed form the frozen design states for that arm), and the
    alternating schedule must strictly reduce the reconstruction error as ``T`` grows.
    """
    torch.manual_seed(3)
    weight = torch.randn(128, 96)
    spec = CodebookSpec(bits=2, kind="nf", block_size=64)

    first_base = fake_quantize_codebook(weight, spec)
    closed_form = truncated_svd(weight - first_base, 16)
    a1, b1 = _codebook_loftq_pair(weight, rank=16, spec=spec, iterations=1)
    assert torch.equal(a1, closed_form.A)
    assert torch.equal(b1, closed_form.B)

    def error(iterations: int) -> float:
        a, b = _codebook_loftq_pair(weight, rank=16, spec=spec, iterations=iterations)
        base = fake_quantize_codebook(weight - b @ a, spec)
        reference = torch.linalg.matrix_norm(weight)
        return float(torch.linalg.matrix_norm(weight - base - b @ a) / reference)

    assert error(1) < float(
        torch.linalg.matrix_norm(weight - first_base) / torch.linalg.matrix_norm(weight)
    )
    assert error(3) < error(1)
    with pytest.raises(ValueError, match="iterations"):
        _codebook_loftq_pair(weight, rank=16, spec=spec, iterations=0)


# --------------------------------------------------------------------------------------
# R2 initialisation: the straight-through full-QAT layer
# --------------------------------------------------------------------------------------
def test_straight_through_quantized_linear_forward_and_gradient() -> None:
    """``StraightThroughQuantizedLinear``: forward = fake-quantized dense, gradient = dequantizer's.

    The gradient reaching the weight is the gradient of the dequantized weight - finite, non-zero,
    and equal to ``dL/d(dequant)`` (the quantizer's own Jacobian is replaced by identity, which is
    the standard QAT estimator and the documented limitation of the layer).
    """
    torch.manual_seed(5)
    spec = _quant_spec()
    dense = nn.Linear(WIDTH, WIDTH)
    layer = StraightThroughQuantizedLinear.from_linear(dense, spec)
    assert {name for name, _ in layer.named_parameters()} == {"weight", "bias"}

    inputs = torch.randn(4, WIDTH)
    with torch.no_grad():
        quantized_weight = fake_quantize(layer.weight, spec)
    expected = torch.nn.functional.linear(inputs, quantized_weight, layer.bias)
    actual = layer(inputs)
    # Same *dequantized* weight on both sides, so the difference is float32 reduction order only.
    assert torch.allclose(actual, expected, atol=1e-6)

    loss = (actual**2).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert float(layer.weight.grad.abs().sum()) > 0.0

    # The straight-through claim, exactly: the weight's gradient is the dequantized weight's.
    reference = quantized_weight.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(inputs, reference, layer.bias).mul(1.0).pow(2).sum().backward()
    assert torch.allclose(layer.weight.grad, reference.grad, atol=0.0)
    # A separate class from the low-rank layer, with a different trainable parameter set.
    assert not isinstance(layer, QuantizedPlusLowRankLinear)


def test_full_qat_replaces_every_target_weight_with_a_quantized_trainable_layer() -> None:
    plan = load_plan(M3_PLAN)
    arm = next(a for a in plan.arms if a.kind == "r2_fullqat_4bit")
    compression = resolve_arm_compression(plan, arm)
    weights = {"proj.weight": torch.randn(WIDTH, WIDTH), "head.weight": torch.randn(VOCAB, WIDTH)}

    applied = apply_arm(weights, arm, compression, seed=0)
    init = applied.trainable_init
    assert init is not None and init.style == "straight_through"
    assert set(init.dense_weights) == set(weights)
    assert init.spec is not None and init.spec.bits == 4
    # Its stored object is the quantized weight of every target layer (class 1, no adapter bytes).
    assert applied.accounted > 0 and applied.compression["ranks"] is None
    assert applied.compression["bytes_source"] == "accounted"


# --------------------------------------------------------------------------------------
# The learning-rate search
# --------------------------------------------------------------------------------------
def test_the_lr_search_records_every_candidate_and_selects_on_dev_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _offline(monkeypatch)
    # The search must not measure the test split: a spy on the perplexity protocol proves it.
    calls: list[int] = []
    real_perplexity = plan_runner.token_level_perplexity

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real_perplexity(*args, **kwargs)

    monkeypatch.setattr(plan_runner, "token_level_perplexity", spy)

    grid = [1.0e-4, 1.0e-2]
    plan_file = _small_plan(tmp_path, steps=4, grid=grid)
    result = run_plan(
        plan_file,
        arms=["r1_std_2bit"],
        out_dir=tmp_path / "runs",
        seq_len=8,
        max_tokens=8,
        progress=lambda _: None,
    )
    metrics = validate_manifest_file(result.runs[0].manifest_path)["metrics"]

    search = metrics["training.lr_search"]
    assert [record["candidate"] for record in search] == grid
    assert all("dev_perplexity" in record for record in search)
    assert all(record["steps"] == _search_prefix_steps(4) for record in search)
    assert all(record["steps"] <= LR_SEARCH_MAX_STEPS for record in search)
    # One measurement at the initialisation and one after training; a search that peeked at the test
    # split would add one per candidate on top of them.
    assert len(calls) == 2
    # The selection is on dev, and the test split was not read by the search.
    assert metrics["training.lr_search_split"] == "development"
    assert metrics["training.lr_search_test_split_read"] is False
    finite = [record for record in search if record["dev_perplexity"] is not None]
    assert finite, "at least one candidate must produce a finite dev perplexity"
    best = min(finite, key=lambda record: record["dev_perplexity"])
    assert metrics["training.learning_rate"] == best["candidate"]
    assert metrics["training.learning_rate_declared"] == 1.0e-4


def test_the_search_prefix_is_a_small_documented_fraction_of_the_schedule() -> None:
    assert _search_prefix_steps(1000) == LR_SEARCH_MAX_STEPS
    assert _search_prefix_steps(1000) < 1000
    assert _search_prefix_steps(1) == 1


def test_a_non_finite_dev_number_cannot_be_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate whose prefix diverged is excluded, not sorted as the best.

    Forced by construction: the plan's grid is replaced with a single enormous rate, which drives the
    fixture model's dev loss non-finite, so the search has no usable candidate and must fall back to
    the plan's declared rate *recording* why - rather than selecting a ``None``.
    """
    _offline(monkeypatch)
    plan_file = _small_plan(tmp_path, steps=4, grid=[1.0e12])
    result = run_plan(
        plan_file,
        arms=["r1_std_2bit"],
        out_dir=tmp_path / "runs",
        seq_len=8,
        max_tokens=8,
        progress=lambda _: None,
    )
    metrics = validate_manifest_file(result.runs[0].manifest_path)["metrics"]
    record = metrics["training.lr_search"][0]
    assert record["candidate"] == 1.0e12
    if record["dev_perplexity"] is None:
        assert metrics["training.learning_rate"] == metrics["training.learning_rate_declared"]
    else:  # pragma: no cover - a finite dev loss means this fixture cannot force the exclusion
        assert metrics["training.learning_rate"] == record["candidate"]


# --------------------------------------------------------------------------------------
# The learned step size
# --------------------------------------------------------------------------------------
def _run_step_size_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, lr_s: float) -> dict:
    _offline(monkeypatch)
    plan_file = _small_plan(tmp_path, steps=8, grid=[1.0e-3])
    config = load_plan(plan_file)
    arms = [
        arm.model_copy(update={"point": {**arm.point, "step_size_lr": lr_s}})
        if arm.kind == "r2_lrqat_4bit"
        else arm
        for arm in config.arms
    ]
    edited = config.model_copy(update={"arms": arms})
    plan_file.write_text(
        yaml.safe_dump(edited.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    result = run_plan(
        plan_file,
        arms=["r2_lrqat_4bit"],
        out_dir=tmp_path / "runs",
        seq_len=8,
        max_tokens=8,
        progress=lambda _: None,
    )
    return validate_manifest_file(result.runs[0].manifest_path)["metrics"]


def test_step_size_lr_zero_freezes_s_and_a_positive_rate_moves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _run_step_size_arm(tmp_path / "frozen", monkeypatch, lr_s=0.0)
    trained = _run_step_size_arm(tmp_path / "trained", monkeypatch, lr_s=1.0e-2)

    # `0` is the *frozen* grid point; the manifest keeps it apart from "not applicable".
    assert frozen["training.step_size_lr"] == 0.0
    assert frozen["training.step_size_frozen"] is True
    assert frozen["training.step_sizes_initial"] == frozen["training.step_sizes_final"]
    assert frozen["training.step_sizes_initial"], "the LR-QAT arm must expose its step sizes"

    assert trained["training.step_size_lr"] == 1.0e-2
    assert trained["training.step_size_frozen"] is False
    assert trained["training.step_sizes_final"] != trained["training.step_sizes_initial"]


def test_an_arm_without_a_step_size_records_not_applicable(tmp_path: Path) -> None:
    """``None`` (not applicable) and ``0.0`` (frozen) must not be conflated."""
    plan = load_plan(M3_PLAN)
    arm = next(arm for arm in plan.arms if arm.kind == "r1_std_2bit")
    compression = resolve_arm_compression(plan, arm)
    assert compression.step_size_lr is None
    weights = {"proj.weight": torch.randn(WIDTH, WIDTH)}
    metrics = apply_arm(weights, arm, compression, seed=0).metrics
    assert metrics["compression.step_size_lr"] is None
    assert metrics["compression.step_size_frozen"] is None


def test_lrqat_without_a_declared_step_size_rate_is_refused() -> None:
    plan = load_plan(M3_PLAN)
    arm = next(arm for arm in plan.arms if arm.kind == "r2_lrqat_4bit")
    stripped = arm.model_copy(update={"point": {"rank": 32, "bits": 4, "group_size": 128}})
    with pytest.raises(ValueError, match="step_size_lr"):
        resolve_arm_compression(plan, stripped)


# --------------------------------------------------------------------------------------
# Resolution refusals
# --------------------------------------------------------------------------------------
def test_a_missing_required_point_names_the_arm() -> None:
    plan = load_plan(M3_PLAN)
    arm = next(arm for arm in plan.arms if arm.kind == "r1_fp16_lora")
    stripped = arm.model_copy(update={"point": {}})
    with pytest.raises(ValueError, match="r1_fp16_lora"):
        resolve_arm_compression(plan, stripped)


def test_the_r2_arms_pin_their_own_quantizer() -> None:
    plan = load_plan(M3_PLAN)
    arm = next(a for a in plan.arms if a.kind == "r2_rtn_4bit")
    with pytest.raises(ValueError, match="granularity"):
        resolve_arm_compression(plan, arm, granularity="per_tensor")
    codebook_arm = next(a for a in plan.arms if a.kind == "r1_std_2bit")
    with pytest.raises(ValueError, match="codebook"):
        resolve_arm_compression(plan, codebook_arm, granularity="per_group")


def test_an_r1_codebook_arm_must_stay_at_two_bits() -> None:
    plan = load_plan(M3_PLAN)
    arm = next(a for a in plan.arms if a.kind == "r1_std_2bit")
    four_bit = arm.model_copy(update={"point": {"rank": 16, "bits": 4, "block_size": 64}})
    with pytest.raises(ValueError, match="2 bits only"):
        resolve_arm_compression(plan, four_bit)


# --------------------------------------------------------------------------------------
# Divergence rule (§5.4)
# --------------------------------------------------------------------------------------
def test_a_divergent_arm_is_recorded_not_reported_as_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken run is a *recorded* outcome: the number never leaves as a quality measurement.

    Forced by construction: the plan's own rate is set to an absurd value, so the arm's loss explodes
    and its perplexity cannot be measured. The frozen rule (reproduction-plan.md §5.4) then requires
    ``training.divergent`` with the trigger, and no run file may carry a non-standard ``Infinity``
    literal.
    """
    import json

    _offline(monkeypatch)
    out = tmp_path / "runs"
    config = load_plan(M3_PLAN)
    schedule = config.training
    assert schedule is not None
    broken = config.model_copy(
        update={
            "training": schedule.model_copy(
                update={
                    "steps": 4,
                    "warmup_steps": 0,
                    "batch_size": 2,
                    "learning_rate": 1.0e12,
                    "learning_rate_grid": [],
                }
            )
        }
    )
    path = tmp_path / "divergent_plan.yaml"
    path.write_text(
        yaml.safe_dump(broken.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    result = run_plan(
        path,
        arms=["r1_std_2bit"],
        out_dir=out,
        seq_len=8,
        max_tokens=8,
        progress=lambda _: None,
    )
    run = result.runs[0]
    assert run.divergent is True and run.perplexity is None
    manifest = validate_manifest_file(run.manifest_path)
    metrics = manifest["metrics"]
    assert metrics["training.divergent"] is True
    assert metrics["training.divergence_trigger"]
    assert metrics["perplexity"] is None
    assert metrics["perplexity_non_finite"] is True
    assert metrics["perplexity.is_quality_measurement"] is False
    assert metrics["quality.is_measurement"] is False
    # The per-arm metrics file is strict JSON: no `Infinity`/`NaN` literal sneaks in.
    for file in (run.metrics_path, result.aggregate_metrics_path):
        assert file is not None
        json.dumps(json.loads(Path(file).read_text()), allow_nan=False)

    # A healthy arm is the opposite: the flag is present and false, and the number is a measurement.
    healthy = _small_plan(tmp_path / "healthy", steps=2, grid=[1.0e-4])
    ok = run_plan(
        healthy,
        arms=["r1_std_2bit"],
        out_dir=tmp_path / "healthy-runs",
        seq_len=8,
        max_tokens=8,
        progress=lambda _: None,
    )
    healthy_metrics = validate_manifest_file(ok.runs[0].manifest_path)["metrics"]
    assert healthy_metrics["training.divergent"] is False
    assert healthy_metrics["perplexity.is_quality_measurement"] is True
    assert healthy_metrics["quality.is_measurement"] is True


def test_the_divergence_rule_fires_on_perplexity_and_on_a_flat_loss() -> None:
    improving = [10.0 - index * 0.01 for index in range(200)]
    assert _divergence_trigger(improving, 10.0) is None
    assert "threshold" in str(_divergence_trigger(improving, 1000.5))
    assert "finite" in str(_divergence_trigger(improving, None))
    flat = [5.0] * 200
    trigger = _divergence_trigger(flat, 10.0)
    assert trigger is not None and "did not decrease" in trigger
    # A short run (fewer than the 100-step window) is never judged on the loss trajectory alone.
    assert _divergence_trigger([5.0] * 100, 10.0) is None
    # A non-finite value inside the trajectory is divergent whatever the endpoints say.
    assert "non-finite" in str(_divergence_trigger([10.0, float("nan"), *improving], 10.0))


# --------------------------------------------------------------------------------------
# No state leaks between arms
# --------------------------------------------------------------------------------------
def test_a_non_trainable_arm_is_unchanged_whether_it_runs_alone_or_after_a_trainable_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (extended to the M3 set): a trained arm's representation must not leak.

    ``r2_rtn_4bit`` is deterministic and non-trainable, so its perplexity must be identical whether
    it runs on its own or after the trained ``r1_loftq_2bit`` and ``r2_lrqat_4bit`` arms - which
    replace layers and are exactly the arms that used to leak through a shared model instance.
    """
    _offline(monkeypatch)
    plan_file = _small_plan(tmp_path, steps=4, grid=[1.0e-3])

    alone = run_plan(
        plan_file,
        arms=["r2_rtn_4bit"],
        out_dir=tmp_path / "alone",
        seq_len=8,
        max_tokens=8,
        progress=lambda _: None,
    )
    after = run_plan(
        plan_file,
        arms=["r1_loftq_2bit", "r2_lrqat_4bit", "r2_rtn_4bit"],
        out_dir=tmp_path / "after",
        seq_len=8,
        max_tokens=8,
        progress=lambda _: None,
    )

    alone_metrics = validate_manifest_file(alone.runs[0].manifest_path)["metrics"]
    after_runs = {run.arm: run for run in after.runs}
    after_metrics = validate_manifest_file(after_runs["r2_rtn_4bit"].manifest_path)["metrics"]
    assert alone_metrics["perplexity"] == after_metrics["perplexity"]
    assert (
        alone_metrics["compression.relative_fro_mean"]
        == after_metrics["compression.relative_fro_mean"]
    )


def test_the_standard_adapter_initialisation_is_seed_controlled_but_q_is_not() -> None:
    """The seed moves the standard adapter's ``A`` (never ``B``) and never the shared base ``Q``."""
    plan = load_plan(M3_PLAN)
    arm = next(a for a in plan.arms if a.kind == "r1_std_2bit")
    compression = resolve_arm_compression(plan, arm)
    torch.manual_seed(11)
    weights = {"proj.weight": torch.randn(WIDTH, WIDTH)}

    first = apply_arm(weights, arm, compression, seed=0).trainable_init
    second = apply_arm(weights, arm, compression, seed=1).trainable_init
    assert first is not None and second is not None
    assert torch.equal(first.bases["proj.weight"], second.bases["proj.weight"])
    assert not torch.equal(first.factors["proj.weight"][0], second.factors["proj.weight"][0])
    assert bool((first.factors["proj.weight"][1] == 0).all())


def test_the_plan_is_the_only_source_of_the_frozen_arm_points() -> None:
    """Every frozen arm is bound by the plan, and the runner resolves exactly those points."""
    plan = load_plan(M3_PLAN)
    resolved = {arm.kind: resolve_arm_compression(plan, arm) for arm in plan.arms}
    assert set(resolved) == M3_ARM_KINDS
    assert resolved["r1_fp16_lora"].bits is None
    assert resolved["r1_fp16_lora"].rank == 16
    assert resolved["r1_std_2bit"].block_size == 64
    assert resolved["r1_loftq_2bit"].loftq_iterations == 5
    assert resolved["r1_loftq_2bit_t1"].loftq_iterations == 1
    assert resolved["r2_rtn_4bit"].group_size == 128
    assert resolved["r2_fullqat_4bit"].rank is None


def test_only_the_arms_own_parameters_are_trainable(tmp_path: Path) -> None:
    """The frozen recipe trains the adapter only: embeddings and untouched modules stay frozen."""
    from spectraquant.cloud.plan_runner import _install_trainable_layers

    plan = load_plan(M3_PLAN)
    weights = {"proj.weight": torch.randn(WIDTH, WIDTH), "head.weight": torch.randn(VOCAB, WIDTH)}
    for kind in ("r1_std_2bit", "r2_lrqat_4bit", "r2_fullqat_4bit"):
        arm = next(a for a in plan.arms if a.kind == kind)
        applied = apply_arm(weights, arm, resolve_arm_compression(plan, arm), seed=0)
        assert applied.trainable_init is not None
        model = MiniLM()
        _install_trainable_layers(model, applied.trainable_init, device="cpu")
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        assert trainable, kind
        assert not any(name.startswith("embed.") for name in trainable), kind
        assert all(name.split(".")[0] in {"proj", "head"} for name in trainable), kind
        # The base buffers are frozen by construction, so only the adapter can move.
        for module in model.modules():
            if isinstance(module, QuantizedPlusLowRankLinear):
                assert module.base.requires_grad is False
