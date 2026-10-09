"""End-to-end tests of the Milestone-5 preparation loop on the Tier-0 fixture.

Covers the loop contract that the H3 sweep depends on: deterministic seeding, bit-for-bit
checkpoint/resume, per-term diagnostics, and a schema-valid manifest that declares the class-1
analytical byte figure of the factorized artifact. All of it is CPU-only and runs in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from spectraquant.config import load_experiment_config
from spectraquant.data.synthetic import build_corpus
from spectraquant.quantization import QuantSpec
from spectraquant.regularizers.spectral import PreparationObjective
from spectraquant.reporting.manifests import validate_manifest_file
from spectraquant.training.loop import (
    LoopConfig,
    SequenceData,
    evaluate_sequences,
    load_checkpoint,
    save_checkpoint,
    train_language_model,
)
from spectraquant.training.low_rank import factorize_linears, gather_factor_pairs
from spectraquant.training.seeding import seed_everything
from spectraquant.training.tiny_lm import TinyCharTransformer

pytestmark = pytest.mark.integration

SEED = 4242
STEPS = 12
CONFIG = "configs/experiment/regularizer_tier0.yaml"


@pytest.fixture(scope="module")
def fixture():
    cfg = load_experiment_config(CONFIG, ["method=regularizer_rounding"])
    corpus = build_corpus(cfg.data, seed=cfg.experiment.seed)
    data = SequenceData.from_corpus(corpus)
    spec = QuantSpec(
        bits=int(cfg.method.compression.bits),
        granularity=cfg.method.compression.granularity,
        group_size=cfg.method.compression.group_size,
        symmetric=cfg.method.compression.symmetric,
    )
    return cfg, data, spec


def _build(cfg, *, rank: int = 8):
    """Deterministically build the fixture and factorize it (the caller's reproducibility duty)."""
    seed_everything(SEED)
    model = TinyCharTransformer(cfg.model)
    references = factorize_linears(model, rank=rank)
    return model, references


def _loop(**overrides) -> LoopConfig:
    payload = {"batch_size": 8, "lr": 3e-3, "seed": SEED, "log_every": 4}
    payload.update(overrides)
    return LoopConfig(**payload)


def _train(cfg, data, spec, tmp_path: Path, *, name: str, **overrides):
    model, references = _build(cfg)
    objective = overrides.pop("objective", None)
    if objective is None:
        objective = PreparationObjective.from_config(cfg.method.regularizer, spec=spec)
    return train_language_model(
        model,
        method=cfg.method.compression.method,
        steps=overrides.pop("steps", STEPS),
        output_dir=tmp_path / name,
        data=data,
        objective=objective,
        references=references,
        loop=overrides.pop("loop", _loop()),
        spec=spec,
        measurement_class=cfg.method.measurement_class,
        config={"config_path": str(cfg.config_path), "data": {"name": cfg.data.name}},
        **overrides,
    )


def test_same_seed_reproduces_the_loss_sequence(fixture, tmp_path: Path) -> None:
    cfg, data, spec = fixture
    first = _train(cfg, data, spec, tmp_path, name="a")
    second = _train(cfg, data, spec, tmp_path, name="b")
    assert first.losses == second.losses
    assert first.loss_sequence_sha256() == second.loss_sequence_sha256()
    assert len(first.losses) == STEPS


def test_checkpoint_resume_reproduces_the_remaining_loss_sequence(fixture, tmp_path: Path) -> None:
    cfg, data, spec = fixture
    loop = _loop(checkpoint_every=6)
    full = _train(cfg, data, spec, tmp_path, name="full", loop=loop, steps=STEPS)
    checkpoint = tmp_path / "full" / "checkpoints" / "step-0000006.pt"
    assert checkpoint.is_file()

    model, references = _build(cfg)
    resumed = train_language_model(
        model,
        method=cfg.method.compression.method,
        steps=STEPS,
        output_dir=tmp_path / "resumed",
        data=data,
        objective=PreparationObjective.from_config(cfg.method.regularizer, spec=spec),
        references=references,
        loop=loop,
        spec=spec,
        resume_from=checkpoint,
        write=False,
    )
    assert resumed.manifest.metrics["resumed_from_step"] == 6
    assert resumed.losses == full.losses
    assert resumed.loss_sequence_sha256() == full.loss_sequence_sha256()
    assert [step for step, _ in resumed.val_losses] == [step for step, _ in full.val_losses]


def test_resume_rejects_a_mismatched_seed_or_configuration(fixture, tmp_path: Path) -> None:
    cfg, data, spec = fixture
    loop = _loop(checkpoint_every=6)
    _train(cfg, data, spec, tmp_path, name="full", loop=loop)
    checkpoint = tmp_path / "full" / "checkpoints" / "step-0000006.pt"

    model, references = _build(cfg)
    with pytest.raises(ValueError):
        train_language_model(
            model,
            method=cfg.method.compression.method,
            steps=STEPS,
            output_dir=tmp_path / "bad-seed",
            data=data,
            objective=PreparationObjective.from_config(cfg.method.regularizer, spec=spec),
            references=references,
            loop=_loop(seed=SEED + 1, checkpoint_every=6),
            spec=spec,
            resume_from=checkpoint,
            write=False,
        )

    model, references = _build(cfg)
    with pytest.raises(ValueError):
        train_language_model(
            model,
            method=cfg.method.compression.method,
            steps=STEPS,
            output_dir=tmp_path / "bad-lr",
            data=data,
            references=references,
            loop=_loop(lr=1e-2, checkpoint_every=6),
            spec=spec,
            resume_from=checkpoint,
            write=False,
        )


def test_load_checkpoint_rejects_a_foreign_payload(tmp_path: Path) -> None:
    model, _ = _build(load_experiment_config(CONFIG))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)
    bogus = tmp_path / "bogus.pt"
    torch.save({"format": "something-else"}, bogus)
    with pytest.raises(ValueError):
        load_checkpoint(bogus, model=model, optimizer=optimizer, generator=generator, loop=_loop())


def test_manifest_is_schema_valid_and_labels_the_bytes(fixture, tmp_path: Path) -> None:
    cfg, data, spec = fixture
    result = _train(cfg, data, spec, tmp_path, name="manifest")
    assert result.manifest_path is not None
    validate_manifest_file(result.manifest_path)
    document = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert document["measurement_class"] == 2
    assert document["compression"]["method"] == "spectraquant"
    assert document["compression"]["ranks"] == [8]
    assert document["compression"]["bits"] == 4
    assert document["compression"]["bytes_source"] == "accounted"
    assert document["compression"]["measured_bytes"] is None
    assert document["compression"]["accounted_bytes"] > 0
    assert document["training"]["steps"] == STEPS
    metrics = document["metrics"]
    assert metrics["loss_sequence_sha256"].startswith("sha256:")
    assert metrics["val_loss_final"] > 0.0
    # Per-term diagnostics: the task loss plus every objective component and its coefficient.
    for key in (
        "term_mean_task",
        "term_mean_factorization",
        "term_mean_rounding",
        "term_mean_residual",
        "term_mean_spectrum",
        "term_mean_total",
        "lambda_round",
    ):
        assert key in metrics, key
    assert metrics["regularizer_name"] == "rounding"


def test_regularizer_disabled_arm_reports_a_noop_objective(fixture, tmp_path: Path) -> None:
    _cfg, data, spec = fixture
    none_cfg = load_experiment_config(CONFIG, ["method=regularizer_none"])
    result = _train(none_cfg, data, spec, tmp_path, name="none")
    assert result.manifest.metrics["regularizer_is_noop"] is True
    # The component *values* are still monitored (they are diagnostics, not contributions); what is
    # zero is their weighted sum, so the optimised loss is the task loss alone.
    assert result.manifest.metrics["lambda_round"] == 0.0
    assert result.manifest.metrics["term_mean_rounding"] > 0.0
    assert result.manifest.metrics["term_mean_total"] == pytest.approx(
        result.manifest.metrics["term_mean_task"], rel=1e-9
    )


def test_regularizer_terms_change_the_optimised_loss(fixture, tmp_path: Path) -> None:
    """A non-noop arm must optimise task + penalty, not the task alone."""
    cfg, data, spec = fixture
    penalized = _train(cfg, data, spec, tmp_path, name="rounding")
    disabled = _train(
        cfg,
        data,
        spec,
        tmp_path,
        name="disabled",
        objective=PreparationObjective(spec=spec),
    )
    assert penalized.losses != disabled.losses
    assert penalized.terms[0]["total"] > penalized.terms[0]["task"]


def test_objective_penalties_reach_the_factors(fixture, tmp_path: Path) -> None:
    """The trained factors must differ from their SVD initialisation under a non-noop objective."""
    cfg, data, spec = fixture
    model, references = _build(cfg)
    initial = {
        name: (a.detach().clone(), b.detach().clone())
        for name, (a, b) in gather_factor_pairs(model).items()
    }
    train_language_model(
        model,
        method=cfg.method.compression.method,
        steps=4,
        output_dir=tmp_path / "moved",
        data=data,
        objective=PreparationObjective.from_config(cfg.method.regularizer, spec=spec),
        references=references,
        loop=_loop(),
        spec=spec,
        write=False,
    )
    moved = {
        name: not torch.allclose(a.detach(), initial[name][0])
        or not torch.allclose(b.detach(), initial[name][1])
        for name, (a, b) in gather_factor_pairs(model).items()
    }
    assert all(moved.values())


def test_evaluate_sequences_is_deterministic_and_bounded(fixture) -> None:
    cfg, data, _spec = fixture
    model, _ = _build(cfg)
    first = evaluate_sequences(model, data.val, batch_size=8)
    assert first == evaluate_sequences(model, data.val, batch_size=8)
    assert 0.0 < first < 10.0


def test_save_checkpoint_round_trip(tmp_path: Path, fixture) -> None:
    cfg, _data, _spec = fixture
    model, _ = _build(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(SEED + 1)
    torch.randint(0, 10, (3,), generator=generator)
    state_before = generator.get_state().clone()
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        optimizer=optimizer,
        generator=generator,
        step=3,
        loop=_loop(),
        losses=[1.0, 0.5, 0.25],
        terms=[{"task": 1.0}],
        val_losses=[(3, 0.9)],
        metadata={"note": "unit"},
    )
    torch.randint(0, 10, (3,), generator=generator)
    fresh = torch.Generator().manual_seed(0)
    payload = load_checkpoint(path, model=model, optimizer=optimizer, generator=fresh, loop=_loop())
    assert payload["step"] == 3
    assert payload["metadata"] == {"note": "unit"}
    assert torch.equal(fresh.get_state(), state_before)


def test_evaluate_language_model_still_fails_loudly() -> None:
    from spectraquant.training.loop import evaluate_language_model

    with pytest.raises(NotImplementedError):
        evaluate_language_model(torch.nn.Linear(2, 2), dataset_id="wikitext-2")


def test_warmup_ramps_the_learning_rate_and_then_holds_it(fixture, tmp_path: Path) -> None:
    """The schedule's warmup must actually reach the optimiser, not be recorded and ignored.

    Warmup is a pure function of the step index, so the *first* step must use the smallest rate
    (`lr / (warmup + 1)`) and every step after the ramp must use `lr` exactly.
    """
    cfg, data, spec = fixture
    rates: list[float] = []
    model, references = _build(cfg)

    original_step = torch.optim.AdamW.step

    def recording_step(self, *args, **kwargs):
        rates.append(float(self.param_groups[0]["lr"]))
        return original_step(self, *args, **kwargs)

    torch.optim.AdamW.step = recording_step
    try:
        train_language_model(
            model,
            method="none",
            steps=5,
            output_dir=tmp_path / "warmup",
            data=data,
            references=references,
            loop=_loop(warmup_steps=3),
            spec=spec,
            write=False,
        )
    finally:
        torch.optim.AdamW.step = original_step

    base = 3e-3
    assert rates[0] == pytest.approx(base / 4)  # step 0 of a 3-step ramp
    assert rates[1] == pytest.approx(base / 2)
    assert rates[2] == pytest.approx(base * 3 / 4)
    assert rates[3] == pytest.approx(base)
    assert rates[4] == pytest.approx(base)


def test_a_resumed_run_continues_the_warmup_schedule(fixture, tmp_path: Path) -> None:
    """A resume must not restart the ramp: the schedule is positioned by the resumed step."""
    cfg, data, spec = fixture
    first = _train(
        cfg,
        data,
        spec,
        tmp_path,
        name="warm-first",
        steps=2,
        loop=_loop(warmup_steps=4, checkpoint_every=1),
    )
    assert first.checkpoint_paths, "the resume test needs a checkpoint to resume from"

    rates: list[float] = []
    original_step = torch.optim.AdamW.step

    def recording_step(self, *args, **kwargs):
        rates.append(float(self.param_groups[0]["lr"]))
        return original_step(self, *args, **kwargs)

    torch.optim.AdamW.step = recording_step
    try:
        resumed = _train(
            cfg,
            data,
            spec,
            tmp_path,
            name="warm-resume",
            steps=5,
            loop=_loop(warmup_steps=4),
            resume_from=Path(first.checkpoint_paths[-1]),
        )
    finally:
        torch.optim.AdamW.step = original_step

    assert resumed.steps_completed == 5
    # Steps 2, 3 and 4 of a 4-step ramp: 3/5, 4/5, then the full rate.
    assert rates[0] == pytest.approx(3e-3 * 3 / 5)
    assert rates[1] == pytest.approx(3e-3 * 4 / 5)
    assert rates[2] == pytest.approx(3e-3)
