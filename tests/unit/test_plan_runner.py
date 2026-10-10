"""``spectraquant run-plan``: honest arm coverage, schema-valid manifests, loud pin failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from _plan_fixtures import OTHER_PIN, PIN, documents, plan_runner_env
from typer.testing import CliRunner

from spectraquant.cli.main import app
from spectraquant.cloud import plan_data
from spectraquant.cloud.plan_data import PinnedRevisionError, is_plan_document
from spectraquant.cloud.plan_runner import (
    IMPLEMENTED_ARM_KINDS,
    TRAINABLE_ARM_KINDS,
    ModelRevisionMismatch,
    apply_arm,
    arm_not_implemented_error,
    compressible_linear_names,
    resolve_arm_compression,
    revision_of,
    run_plan,
)
from spectraquant.cloud.remote import RESULT_LINE_PREFIX, materialise_datasets, parse_result_line
from spectraquant.cloud.spec import dataset_declaration_digest, plan_to_run_spec
from spectraquant.experiment_plan import load_plan
from spectraquant.factorization import truncated_svd
from spectraquant.quantization import QuantSpec, measure_serialized_bytes
from spectraquant.reporting.manifests import validate_manifest_file

TIER1_PLAN = "configs/tier1/smollm2_135m.yaml"
TIER2_PLAN = "configs/tier2/tinyllama_1_1b.yaml"
REPRO_PLAN = "configs/repro/lr_qat_smollm2_135m.yaml"

runner = CliRunner()


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every arm offline and deterministically (fake model, tokenizer and corpus)."""
    plan_runner_env(monkeypatch)


def _manifest_of(result: object, arm: str) -> dict:
    runs = [run for run in result.runs if run.arm == arm]  # type: ignore[attr-defined]
    assert runs, f"arm {arm!r} did not run"
    return validate_manifest_file(runs[0].manifest_path)


# --------------------------------------------------------------------------------------
# Arm coverage
# --------------------------------------------------------------------------------------
def test_the_implemented_kinds_cover_the_frozen_arms_and_the_allocated_arms() -> None:
    """The declared arm set: the four originals, the eight frozen M3 arms and the three allocated ones.

    Updated 2026-10-10 (second time): ``quant_then_residual``, ``proxy_allocated`` and
    ``proxy_allocated_regularized`` are implemented now (``ADR-0004``), so the *negative* half of the
    assertion is what is left: ``qlora`` and ``spectraquant`` still have no implementation.
    """
    assert set(IMPLEMENTED_ARM_KINDS) == {
        "fp16_reference",
        "ptq_uniform",
        "low_rank_only",
        "rank_then_quant",
        "quant_then_residual",
        "proxy_allocated",
        "proxy_allocated_regularized",
        "loftq",
        "lr_qat",
        "r1_fp16_lora",
        "r1_std_2bit",
        "r1_loftq_2bit",
        "r1_loftq_2bit_t1",
        "r2_fp16",
        "r2_rtn_4bit",
        "r2_lrqat_4bit",
        "r2_fullqat_4bit",
    }
    assert (
        frozenset(
            {
                "loftq",
                "lr_qat",
                "r1_fp16_lora",
                "r1_std_2bit",
                "r1_loftq_2bit",
                "r1_loftq_2bit_t1",
                "r2_lrqat_4bit",
                "r2_fullqat_4bit",
                # The method arm trains the allocated factors with the preparation objective.
                "proxy_allocated_regularized",
            }
        )
        == TRAINABLE_ARM_KINDS
    )
    assert not (set(IMPLEMENTED_ARM_KINDS) & {"qlora", "spectraquant"})


def test_the_reproduction_plan_runs_all_four_of_its_arms(tmp_path: Path, offline: None) -> None:
    """Every arm of the M3 plan is implemented and bound, so the default run executes all of them.

    Before the trainable arms were wired, `lr_qat` and `loftq` were skipped as unimplemented. They now
    train, so the default run must include them and report nothing as skipped.
    """
    result = run_plan(REPRO_PLAN, out_dir=tmp_path, progress=lambda _: None)

    assert [run.arm for run in result.runs] == [
        "fp16_reference",
        "lr_qat",
        "loftq",
        "rank_then_quant",
    ]
    assert result.skipped == ()


def test_the_tier1_plan_skips_its_unbound_arms_with_a_reason(tmp_path: Path, offline: None) -> None:
    """An arm the invocation cannot bind must be reported with why, never silently omitted.

    The three allocated arms are implemented but unbound: ``quant_then_residual`` needs an explicit
    ``--rank``/``--bits`` (the plan declares grids but binds no point), and the two proxy-allocated
    arms need an explicit ``--budget-bytes`` rung (the plan declares the ladder, not a rung). A
    default run therefore *runs* the five bound arms and records these three as skipped.
    """
    result = run_plan(TIER1_PLAN, out_dir=tmp_path, progress=lambda _: None)

    skipped = {entry["arm"]: entry["reason"] for entry in result.skipped}
    assert set(skipped) == {"quant_then_residual", "proxy_allocated", "spectraquant_regularized"}
    assert "pass --rank" in skipped["quant_then_residual"]
    assert "--budget-bytes" in skipped["proxy_allocated"]
    assert "--budget-bytes" in skipped["spectraquant_regularized"]
    for reason in skipped.values():
        assert reason.strip()
    assert {run.arm for run in result.runs} == {
        "fp16_reference",
        "ptq_uniform_8",
        "ptq_uniform_4",
        "low_rank_only",
        "rank_then_quant",
    }


def test_explicit_unimplemented_trainable_arm_raises_naming_the_milestone(tmp_path: Path) -> None:
    """`qlora` and `spectraquant` are trainable and still unimplemented: they must refuse, not guess."""
    with pytest.raises(NotImplementedError, match="M5"):
        run_plan(TIER2_PLAN, arms=["qlora"], out_dir=tmp_path, progress=lambda _: None)


def test_quant_then_residual_without_a_grid_point_is_refused(tmp_path: Path) -> None:
    """The M4 baseline is implemented but unbound in the plan: a run must state its point.

    Exercised without the offline fixture on purpose: the refusal happens during grid resolution, so
    no model is loaded (and nothing touches the network).
    """
    with pytest.raises(ValueError, match="pass --rank"):
        run_plan(
            TIER1_PLAN, arms=["quant_then_residual"], out_dir=tmp_path, progress=lambda _: None
        )


def test_an_allocated_arm_without_a_budget_is_refused(tmp_path: Path) -> None:
    """The allocated arms are bound by the byte budget: an explicit run must name a ladder rung."""
    with pytest.raises(ValueError, match=r"--budget-bytes"):
        run_plan(TIER1_PLAN, arms=["proxy_allocated"], out_dir=tmp_path, progress=lambda _: None)


def test_rank_arm_without_a_grid_point_is_refused() -> None:
    """An unbound arm is refused at resolution time, before any model is loaded.

    Exercised through the resolver rather than through ``run_plan``: the committed plan now binds its
    arms, so a plan-level call would proceed to download the model and this test would be measuring
    the network instead of the rule.
    """
    from spectraquant.cloud.plan_data import plan_path
    from spectraquant.cloud.plan_runner import resolve_arm_compression
    from spectraquant.experiment_plan import load_plan

    plan = load_plan(plan_path(TIER1_PLAN))
    unbound = plan.model_copy(
        update={"arms": [a.model_copy(update={"point": {}}) for a in plan.arms]}
    )
    arm = next(a for a in unbound.arms if a.name == "low_rank_only")
    with pytest.raises(ValueError, match="pass --rank"):
        resolve_arm_compression(
            unbound,
            arm,
            bits=None,
            rank=None,
            granularity="per_group",
            group_size=32,
            symmetric=True,
            axis=0,
        )


def test_the_plan_binds_every_arm_so_no_flag_is_needed() -> None:
    """The plan is the single source of truth for each arm's grid point.

    Before this, the point lived only in the invocation: the notebook's runner command was the sole
    record of what ran, and `--bits=4` was able to bind two different arms to the same width.
    """
    from spectraquant.cloud.plan_data import plan_path
    from spectraquant.cloud.plan_runner import resolve_arm_compression
    from spectraquant.experiment_plan import load_plan

    plan = load_plan(plan_path(TIER1_PLAN))
    expected = {"ptq_uniform_8": 8, "ptq_uniform_4": 4}
    for name, bits in expected.items():
        arm = next(a for a in plan.arms if a.name == name)
        resolved = resolve_arm_compression(
            plan,
            arm,
            bits=None,
            rank=None,
            granularity="per_group",
            group_size=32,
            symmetric=True,
            axis=0,
        )
        assert resolved.bits == bits, name
    for name, rank in {"low_rank_only": 8, "rank_then_quant": 8}.items():
        arm = next(a for a in plan.arms if a.name == name)
        resolved = resolve_arm_compression(
            plan,
            arm,
            bits=None,
            rank=None,
            granularity="per_group",
            group_size=32,
            symmetric=True,
            axis=0,
        )
        assert resolved.rank == rank, name


def test_bits_outside_the_plan_grid_are_refused(tmp_path: Path) -> None:
    # Exercised through an arm that does not name its width: an arm that *does* (ptq_uniform_4) now
    # reports the conflict with its own name first, which is a stronger error.
    with pytest.raises(ValueError, match="predeclared grid"):
        run_plan(
            TIER1_PLAN,
            arms=["rank_then_quant"],
            rank=8,
            bits=5,
            out_dir=tmp_path,
            progress=lambda _: None,
        )


def test_bits_never_silently_override_the_width_in_the_arm_name(tmp_path: Path) -> None:
    """Regression: --bits=4 turned the ptq_uniform_8 arm into a 4-bit arm without saying so.

    The cloud run that exposed this produced "ptq_uniform_8" and "ptq_uniform_4" with identical
    perplexity (18.6190) and identical accounted bytes (59,719,680), because both had been run at
    4 bits; nothing in the record revealed the substitution.
    """
    with pytest.raises(ValueError, match="conflicts with arm 'ptq_uniform_4'"):
        run_plan(
            TIER1_PLAN,
            arms=["ptq_uniform_4"],
            bits=8,
            out_dir=tmp_path,
            progress=lambda _: None,
        )


def test_the_arm_name_supplies_the_width_when_no_bits_are_given(tmp_path: Path) -> None:
    """The name-encoded width is used as-is, so --bits stays free for the arms that need it."""
    from spectraquant.cloud.plan_data import plan_path
    from spectraquant.cloud.plan_runner import resolve_arm_compression
    from spectraquant.experiment_plan import load_plan

    plan = load_plan(plan_path(TIER1_PLAN))
    arm = next(a for a in plan.arms if a.name == "ptq_uniform_8")
    resolved = resolve_arm_compression(
        plan,
        arm,
        bits=None,
        rank=None,
        granularity="per_group",
        group_size=32,
        symmetric=True,
        axis=0,
    )
    assert resolved.bits == 8


def test_rank_outside_the_plan_grid_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="predeclared grid"):
        run_plan(
            TIER1_PLAN,
            arms=["low_rank_only"],
            rank=3,
            out_dir=tmp_path,
            progress=lambda _: None,
        )


def test_seed_outside_the_reported_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reported seed list"):
        run_plan(
            TIER1_PLAN,
            arms=["ptq_uniform_4"],
            seeds=[9],
            out_dir=tmp_path,
            progress=lambda _: None,
        )


def test_per_group_requires_a_grid_group_size() -> None:
    plan = load_plan(TIER1_PLAN)
    arm = next(arm for arm in plan.arms if arm.name == "ptq_uniform_4")

    with pytest.raises(ValueError, match="requires group_size"):
        resolve_arm_compression(plan, arm, granularity="per_group")
    with pytest.raises(ValueError, match="not in plan"):
        resolve_arm_compression(plan, arm, granularity="per_group", group_size=48)


# --------------------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------------------
def test_ptq_arm_writes_a_validated_class2_manifest_with_measured_bytes(
    tmp_path: Path, offline: None
) -> None:
    result = run_plan(
        TIER1_PLAN, arms=["ptq_uniform_4"], out_dir=tmp_path, seq_len=8, progress=lambda _: None
    )

    manifest = _manifest_of(result, "ptq_uniform_4")

    assert manifest["measurement_class"] == 2
    assert manifest["model_id"] == load_plan(TIER1_PLAN).models[0].id
    assert manifest["model_revision"] == PIN
    compression = manifest["compression"]
    assert compression["method"] == "rtn"
    assert compression["bits"] == 4
    assert compression["bytes_source"] == "measured"
    assert compression["measured_bytes"] == compression["accounted_bytes"] > 0
    assert compression["serializer"] == "spectraquant-sqpack-v1"
    assert compression["exclusions"] == []
    metrics = manifest["metrics"]
    assert metrics["perplexity.method"] == "non-overlapping-window-token-ce-v1"
    assert metrics["quality.measurement_class"] == 2
    assert metrics["bytes.measured_class"] == 3
    assert metrics["model.revision_loaded"] == PIN
    assert metrics["plan.arm_kind"] == "ptq_uniform"
    assert manifest["training"]["steps"] == 0
    assert manifest["training"]["tokens"] > 0


def test_rank_then_quant_manifest_records_the_factors(tmp_path: Path, offline: None) -> None:
    result = run_plan(
        TIER1_PLAN,
        arms=["rank_then_quant"],
        rank=4,
        bits=4,
        out_dir=tmp_path,
        seq_len=8,
        progress=lambda _: None,
    )

    manifest = _manifest_of(result, "rank_then_quant")

    assert manifest["measurement_class"] == 2
    assert manifest["compression"]["ranks"] == [4]
    assert manifest["compression"]["bits"] == 4
    assert manifest["compression"]["bytes_source"] == "measured"
    assert manifest["metrics"]["compression.stored_n_tensors"] > 0


def test_low_rank_only_uses_class1_factor_bytes(tmp_path: Path, offline: None) -> None:
    result = run_plan(
        TIER1_PLAN,
        arms=["low_rank_only"],
        rank=4,
        out_dir=tmp_path,
        seq_len=8,
        progress=lambda _: None,
    )

    manifest = _manifest_of(result, "low_rank_only")

    assert manifest["measurement_class"] == 2
    assert manifest["compression"]["method"] == "svd"
    assert manifest["compression"]["bytes_source"] == "accounted"
    assert manifest["compression"]["measured_bytes"] is None
    assert manifest["compression"]["accounted_bytes"] > 0


def test_fp16_reference_claims_no_compression_class(tmp_path: Path, offline: None) -> None:
    result = run_plan(
        TIER1_PLAN,
        arms=["fp16_reference"],
        out_dir=tmp_path,
        seq_len=8,
        progress=lambda _: None,
    )

    manifest = _manifest_of(result, "fp16_reference")

    assert manifest["measurement_class"] is None
    assert manifest["compression"]["method"] == "none"
    assert manifest["compression"]["bytes_source"] == "accounted"
    assert (
        manifest["compression"]["accounted_bytes"]
        == 2 * manifest["metrics"]["compression.n_parameters"]
    )


def test_arms_shrink_the_same_tensor_set_monotonically(offline: None) -> None:
    """The arms are compared at equal memory: same tensors, decreasing stored bytes."""
    from _plan_fixtures import TinyLM

    model = TinyLM()
    state = model.state_dict()
    names, excluded = compressible_linear_names(model)
    assert excluded == []  # the stand-in has no tied embedding
    weights = {name: state[name].clone() for name in names}
    plan = load_plan(TIER1_PLAN)
    by_name = {arm.name: arm for arm in plan.arms}

    reference = apply_arm(
        weights, by_name["fp16_reference"], resolve_arm_compression(plan, by_name["fp16_reference"])
    )
    eight = apply_arm(
        weights, by_name["ptq_uniform_8"], resolve_arm_compression(plan, by_name["ptq_uniform_8"])
    )
    four = apply_arm(
        weights, by_name["ptq_uniform_4"], resolve_arm_compression(plan, by_name["ptq_uniform_4"])
    )
    low_rank = apply_arm(
        weights,
        by_name["low_rank_only"],
        resolve_arm_compression(plan, by_name["low_rank_only"], rank=4),
    )

    assert reference.measured is None and reference.accounted > eight.accounted > four.accounted
    assert four.measured == four.accounted
    assert low_rank.accounted < reference.accounted
    # Reuse, not re-derivation: the container really is what measure_serialized_bytes reports.
    expected = measure_serialized_bytes(four.storage, four.storage_specs)
    assert four.measured == expected
    assert all(isinstance(spec, QuantSpec) for spec in four.storage_specs.values())


def test_gpt2_style_tied_head_is_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    from _plan_fixtures import TinyLM

    model = TinyLM()
    model.head.weight = model.embed.weight  # tie, as SmolLM2/GPT-2 do

    names, excluded = compressible_linear_names(model)

    assert "head.weight (tied embedding weight)" in excluded
    assert "head.weight" not in names


# --------------------------------------------------------------------------------------
# Revision gate
# --------------------------------------------------------------------------------------
def test_revision_of_records_the_pin_when_it_matches() -> None:
    from _plan_fixtures import TinyLM

    assert revision_of(TinyLM(commit=PIN), requested=PIN, source="hub") == PIN


def test_revision_of_refuses_a_different_commit() -> None:
    from _plan_fixtures import TinyLM

    with pytest.raises(ModelRevisionMismatch, match="!= plan pin"):
        revision_of(TinyLM(commit=OTHER_PIN), requested=PIN, source="hub")


def test_revision_of_refuses_an_unprovable_commit() -> None:
    from _plan_fixtures import TinyLM

    with pytest.raises(ModelRevisionMismatch, match="recorded no commit hash"):
        revision_of(TinyLM(commit=None), requested=PIN, source="hub")


def test_run_plan_fails_loudly_on_a_revision_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_runner_env(monkeypatch, commit=OTHER_PIN)

    with pytest.raises(ModelRevisionMismatch, match="!="):
        run_plan(
            TIER1_PLAN,
            arms=["fp16_reference"],
            out_dir=tmp_path,
            seq_len=8,
            progress=lambda _: None,
        )


def test_local_model_substitution_must_be_acknowledged(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow-local-substitution"):
        run_plan(
            TIER1_PLAN,
            arms=["fp16_reference"],
            model_dir=tmp_path,
            out_dir=tmp_path / "out",
            progress=lambda _: None,
        )


def test_substituted_run_is_recorded_as_not_a_plan_measurement(
    tmp_path: Path, offline: None
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    text = tmp_path / "corpus.txt"
    text.write_text("\n\n".join(documents()), encoding="utf-8")

    result = run_plan(
        TIER1_PLAN,
        arms=["fp16_reference"],
        model_dir=model_dir,
        perplexity_text=text,
        allow_local_substitution=True,
        out_dir=tmp_path / "out",
        seq_len=8,
        progress=lambda _: None,
    )

    manifest = _manifest_of(result, "fp16_reference")
    metrics = manifest["metrics"]
    assert manifest["model_revision"].startswith("local-dir:sha256:")
    assert metrics["substitution.is_plan_measurement"] is False
    assert metrics["substitution.model_dir"] == str(model_dir)
    assert metrics["dataset.source"] == str(text)
    assert result.metrics["local_substitution"]["is_plan_measurement"] is False


# --------------------------------------------------------------------------------------
# Result line and aggregate metrics
# --------------------------------------------------------------------------------------
def test_result_line_and_aggregate_metrics(
    tmp_path: Path, offline: None, capsys: pytest.CaptureFixture[str]
) -> None:
    result = run_plan(
        TIER1_PLAN, arms=["ptq_uniform_4"], out_dir=tmp_path, seq_len=8, progress=lambda _: None
    )

    captured = capsys.readouterr().out
    payload = parse_result_line(captured)
    assert payload is not None
    assert RESULT_LINE_PREFIX in captured
    assert payload["plan"] == "tier1_smollm2_135m"
    assert payload["is_plan_measurement"] is True
    assert payload["arms_run"][0]["arm"] == "ptq_uniform_4"
    assert payload["model_revision"] == PIN

    aggregate = json.loads(result.aggregate_metrics_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert aggregate["perplexity.method"] == "non-overlapping-window-token-ce-v1"
    assert aggregate["dataset"]["revision_resolved"] == load_plan(TIER1_PLAN).datasets[0].revision


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def test_cli_runs_one_arm(tmp_path: Path, offline: None) -> None:
    result = runner.invoke(
        app,
        [
            "run-plan",
            "--plan",
            TIER1_PLAN,
            "--arms",
            "ptq_uniform_4",
            "--out",
            str(tmp_path),
            "--seq-len",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "ptq_uniform_4" / "seed-0" / "run_manifest.json").is_file()
    payload = parse_result_line(result.output)
    assert payload is not None and payload["arms_run"][0]["measurement_class"] == 2


def test_cli_exits_non_zero_for_an_unimplemented_trainable_arm(tmp_path: Path) -> None:
    """An unimplemented arm must exit non-zero naming its owner, never run and produce a number.

    Exercised through the Tier-2 plan's `qlora`: the Tier-1 reproduction arms are implemented now, so
    invoking one of those here would start a real training run (and a model download) in a unit test.
    """
    result = runner.invoke(
        app, ["run-plan", "--plan", TIER2_PLAN, "--arms", "qlora", "--out", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "not implemented" in result.output
    assert "M5" in result.output


# --------------------------------------------------------------------------------------
# Plan datasets (the notebook's data cell)
# --------------------------------------------------------------------------------------
def test_is_plan_document_distinguishes_plans_from_experiment_configs() -> None:
    assert is_plan_document(TIER1_PLAN) is True
    assert is_plan_document(REPRO_PLAN) is True
    assert is_plan_document("configs/experiment/smoke.yaml") is False
    assert is_plan_document("configs/does-not-exist.yaml") is False


def test_materialise_datasets_verifies_plan_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_plan(TIER1_PLAN)
    refs = [
        {
            "name": dataset.name,
            "revision": dataset.revision,
            "split": "test",
            "checksum": dataset_declaration_digest(dataset.name, dataset.revision, "test"),
        }
        for dataset in [plan.datasets[0]]
    ]
    monkeypatch.setattr(
        plan_data,
        "verify_pinned_revision",
        lambda repo_id, revision, *, repo_type: revision,
    )

    records = materialise_datasets(
        config_path=TIER1_PLAN,
        overrides=[],
        dataset_refs=refs,
        dest_dir=tmp_path,
        seed=0,
    )

    assert records[0]["verified"] is True
    assert records[0]["revision_resolved"] == plan.datasets[0].revision
    assert records[0]["checksum_semantics"] == "revision-declaration-digest"
    assert (tmp_path / "datasets.json").is_file()


def test_materialise_datasets_rejects_a_tampered_plan_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = load_plan(TIER1_PLAN)
    refs = [
        {
            "name": plan.datasets[0].name,
            "revision": plan.datasets[0].revision,
            "split": "test",
            "checksum": "sha256:" + "0" * 64,
        }
    ]
    monkeypatch.setattr(
        plan_data,
        "verify_pinned_revision",
        lambda repo_id, revision, *, repo_type: revision,
    )

    with pytest.raises(PinnedRevisionError, match="declaration digest"):
        materialise_datasets(
            config_path=TIER1_PLAN,
            overrides=[],
            dataset_refs=refs,
            dest_dir=tmp_path,
            seed=0,
        )


def test_materialise_datasets_rejects_a_floating_revision(tmp_path: Path) -> None:
    plan = load_plan(TIER1_PLAN)
    refs = [
        {
            "name": plan.datasets[0].name,
            "revision": "main",
            "split": "test",
            "checksum": dataset_declaration_digest(plan.datasets[0].name, "main", "test"),
        }
    ]

    with pytest.raises(PinnedRevisionError, match=r"not declared by plan|immutable"):
        materialise_datasets(
            config_path=TIER1_PLAN,
            overrides=[],
            dataset_refs=refs,
            dest_dir=tmp_path,
            seed=0,
        )


def test_wrapper_manifest_resolves_a_plan_spec(tmp_path: Path) -> None:
    """The notebook's manifest cell must build a valid wrapper manifest from a *plan* spec."""
    from spectraquant.cloud.remote import build_run_manifest
    from spectraquant.paths import repo_root
    from spectraquant.reporting.manifests import validate_manifest_dict

    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)

    document = build_run_manifest(
        spec=spec.to_json(),
        workdir=repo_root(),
        artifacts_dir=tmp_path,
        status="success",
        started_utc="2026-10-09T00:00:00Z",
        finished_utc="2026-10-09T00:01:00Z",
    )

    validate_manifest_dict(document, source="wrapper")
    assert document["compression"]["method"] == "none"
    assert document["measurement_class"] is None
    assert document["config_path"] == TIER1_PLAN
    assert document["resolved_config"]["name"] == "tier1_smollm2_135m"
    assert document["metrics"]["cloud.plan.name"] == "tier1_smollm2_135m"
    assert document["metrics"]["cloud.plan.dataset_checksum_semantics"] == (
        "revision-declaration-digest"
    )
    assert document["model_id"] is None


def test_wrapper_manifest_carries_the_runs_model_identity(tmp_path: Path) -> None:
    from spectraquant.cloud.remote import build_run_manifest
    from spectraquant.paths import repo_root
    from spectraquant.reporting.manifests import validate_manifest_dict

    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)

    document = build_run_manifest(
        spec=spec.to_json(),
        workdir=repo_root(),
        artifacts_dir=tmp_path,
        status="success",
        started_utc="2026-10-09T00:00:00Z",
        finished_utc="2026-10-09T00:01:00Z",
        run_manifest={
            "model_id": "HuggingFaceTB/SmolLM2-135M",
            "model_revision": PIN,
            "theoretical_bits": 4.0,
            "packed_bytes": 152,
            "metrics": {"perplexity": 11.2},
        },
    )

    validate_manifest_dict(document, source="wrapper")
    assert document["model_id"] == "HuggingFaceTB/SmolLM2-135M"
    assert document["model_revision"] == PIN
    assert document["packed_bytes"] == 152


def test_plan_bundle_layout_is_collectable(tmp_path: Path) -> None:
    """The spec's expected artifacts must be exactly what the notebook's export cell produces."""
    from _cloud_fixtures import executed_notebook

    from spectraquant.cloud.collect import collect
    from spectraquant.cloud.registry import REGISTRY_FILENAME, Registry
    from spectraquant.cloud.remote import build_run_manifest
    from spectraquant.paths import repo_root

    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)
    bundle = tmp_path / "bundle"
    (bundle / "notebook").mkdir(parents=True)
    document = build_run_manifest(
        spec=spec.to_json(),
        workdir=repo_root(),
        artifacts_dir=bundle,
        status="success",
        started_utc="2026-10-09T00:00:00Z",
        finished_utc="2026-10-09T00:01:00Z",
    )
    (bundle / "run_manifest.json").write_text(json.dumps(document), encoding="utf-8")
    plan_dir = bundle / "artifacts" / "runs" / "tier1_smollm2_135m-plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "metrics.json").write_text(
        '{"plan.name": "tier1_smollm2_135m"}\n', encoding="utf-8"
    )
    (bundle / "notebook" / f"{spec.run_id}.ipynb").write_text(
        json.dumps(executed_notebook(spec.run_id)), encoding="utf-8"
    )
    registry = Registry(tmp_path / "runs" / REGISTRY_FILENAME)

    report = collect(spec.run_id, bundle, spec=spec, registry=registry, record=False)

    assert report.ok is True, report.reasons
    assert report.checksum_verified is True
    assert report.executed_notebook is not None
    assert report.metrics()["cloud.plan.name"] == "tier1_smollm2_135m"


def test_plan_bundle_without_the_metrics_artifact_is_rejected(tmp_path: Path) -> None:
    from _cloud_fixtures import executed_notebook

    from spectraquant.cloud.collect import collect
    from spectraquant.cloud.registry import REGISTRY_FILENAME, Registry
    from spectraquant.cloud.remote import build_run_manifest
    from spectraquant.paths import repo_root

    spec = plan_to_run_spec(TIER1_PLAN, platform="colab", allow_dirty=True)
    bundle = tmp_path / "bundle"
    (bundle / "notebook").mkdir(parents=True)
    document = build_run_manifest(
        spec=spec.to_json(),
        workdir=repo_root(),
        artifacts_dir=bundle,
        status="success",
        started_utc="2026-10-09T00:00:00Z",
        finished_utc="2026-10-09T00:01:00Z",
    )
    (bundle / "run_manifest.json").write_text(json.dumps(document), encoding="utf-8")
    (bundle / "notebook" / f"{spec.run_id}.ipynb").write_text(
        json.dumps(executed_notebook(spec.run_id)), encoding="utf-8"
    )

    report = collect(
        spec.run_id,
        bundle,
        spec=spec,
        registry=Registry(tmp_path / "runs" / REGISTRY_FILENAME),
        record=False,
    )

    assert report.ok is False
    assert any("metrics.json" in reason for reason in report.reasons)


def test_truncated_svd_is_the_factorization_source(offline: None) -> None:
    """A rank arm stores exactly ``B @ A`` from the frozen factorization slice."""
    from _plan_fixtures import TinyLM

    model = TinyLM()
    state = model.state_dict()
    names, _ = compressible_linear_names(model)
    weights = {name: state[name].clone() for name in names}
    plan = load_plan(TIER1_PLAN)
    arm = next(arm for arm in plan.arms if arm.name == "low_rank_only")

    applied = apply_arm(weights, arm, resolve_arm_compression(plan, arm, rank=4))
    name = sorted(weights)[0]
    factors = truncated_svd(weights[name], 4)

    assert applied.state[name].shape == weights[name].shape
    assert not applied.state[name].equal(weights[name])
    assert bool((applied.state[name] - factors.B @ factors.A).abs().max() == 0.0)


def test_arm_error_messages_name_the_owner() -> None:
    plan = load_plan(TIER2_PLAN)
    trainable = next(arm for arm in plan.arms if arm.name == "qlora")

    message = str(arm_not_implemented_error(trainable))

    assert "M5" in message and "trainable" in message


class _FlatLogits(torch.nn.Module):
    """A stub LM: uniform logits, optionally suppressing the classes the tokens actually use.

    Uniform logits over ``vocab`` classes give exactly ``perplexity == vocab``. Suppressing the target
    classes by ``-penalty`` makes the model assign them almost no probability, which is what a broken
    artifact looks like.
    """

    def __init__(self, vocab: int, *, suppress: tuple[int, ...] = (), penalty: float = 0.0) -> None:
        super().__init__()
        self.config = type("C", (), {"max_position_embeddings": 64})()
        self._vocab = vocab
        self._suppress = suppress
        self._penalty = penalty

    def forward(self, ids: torch.Tensor) -> object:
        logits = torch.zeros((1, int(ids.shape[1]), self._vocab))
        if self._suppress and self._penalty:
            for token_id in self._suppress:
                logits[:, :, token_id] = -float(self._penalty)
        return type("Out", (), {"logits": logits})()


class _OneToken:
    """A tokenizer that maps every character to one id (two ids per document)."""

    def __call__(self, text: str, return_tensors: str = "pt") -> object:
        ids = torch.tensor([[1, 2]], dtype=torch.long)
        return type("Enc", (), {"input_ids": ids})()


def test_a_degenerate_perplexity_is_flagged_not_reported_as_quality() -> None:
    """Regression: the rank-8 truncation arm scored 1.4e16 and was recorded like any other number."""
    from spectraquant.cloud.plan_runner import DEGENERATE_PERPLEXITY, token_level_perplexity

    # Constant logits over 100 classes: uniform, so perplexity is exactly 100.
    sane = token_level_perplexity(
        _FlatLogits(100), _OneToken(), ["ab", "cd"], seq_len=8, device="cpu"
    )
    assert sane["perplexity"] == pytest.approx(100.0, rel=1e-6)
    assert sane["perplexity_degenerate"] is False

    # A single token class assigned all the mass elsewhere: perplexity explodes past the threshold.
    broken = token_level_perplexity(
        _FlatLogits(100, suppress=(1, 2), penalty=100.0),
        _OneToken(),
        ["ab"],
        seq_len=8,
        device="cpu",
    )
    assert broken["perplexity"] > DEGENERATE_PERPLEXITY
    assert broken["perplexity_degenerate"] is True


def test_loftq_pairs_its_factors_with_their_own_companion_base() -> None:
    """Regression: the LoftQ factors must be paired with `quantize(w - B@A)`, not `quantize(w)`.

    The primitive returns the low-rank half only; its companion base is the *next* quantization
    half-step of the same alternation. Quantizing the full weight instead paired a base with factors
    that do not belong to it, and the reconstruction was far worse than the naive baseline: 75 %
    relative error against 7.3 % for a plain quantize-then-SVD.
    """
    from spectraquant.cloud.plan_runner import LOFTQ_ITERATIONS
    from spectraquant.factorization.decomposition import truncated_svd
    from spectraquant.quantization import QuantSpec as _Spec
    from spectraquant.quantization import fake_quantize

    torch.manual_seed(0)
    weight = torch.randn(64, 48)
    spec = _Spec(bits=4, granularity="per_group", group_size=32, symmetric=True)
    rank = 8

    def relative(reference: torch.Tensor, other: torch.Tensor) -> float:
        return float(
            torch.linalg.matrix_norm(reference - other) / torch.linalg.matrix_norm(reference)
        )

    arm = next(a for a in load_plan(TIER1_PLAN).arms if a.name == "rank_then_quant")
    compression = resolve_arm_compression(
        load_plan(TIER1_PLAN),
        arm,
        bits=4,
        rank=rank,
        granularity="per_group",
        group_size=32,
        symmetric=True,
        axis=0,
    )
    assert compression.bits is not None and compression.rank is not None

    # The naive baseline: quantize, then take the residual's SVD.
    quantized = fake_quantize(weight, spec)
    baseline = truncated_svd(weight - quantized, rank)
    naive_error = relative(weight, quantized + baseline.B @ baseline.A)

    # The corrected pairing, exactly as the runner does it.
    from spectraquant.factorization.loftq import loftq_initialise

    factors = loftq_initialise(weight, rank=rank, spec=spec, iterations=LOFTQ_ITERATIONS)
    companion = fake_quantize(weight - factors[1] @ factors[0], spec)
    loftq_error = relative(weight, companion + factors[1] @ factors[0])

    assert LOFTQ_ITERATIONS >= 3, "the paper's quality claims are made for T >= 3"
    assert loftq_error < naive_error, (
        f"the LoftQ pairing must beat the naive baseline: {loftq_error:.6f} vs {naive_error:.6f}"
    )


def test_the_loftq_arm_records_its_iteration_count(tmp_path: Path, offline: None) -> None:
    """The schedule is part of the result: a reader must see how many alternating steps ran."""
    from spectraquant.cloud.plan_runner import LOFTQ_ITERATIONS

    result = run_plan(REPRO_PLAN, arms=["loftq"], out_dir=tmp_path, progress=lambda _: None)

    metrics = validate_manifest_file(result.runs[0].manifest_path)["metrics"]
    assert metrics["compression.loftq_iterations"] == LOFTQ_ITERATIONS
    assert metrics["training.steps_completed"] == metrics["training.steps_requested"]


def test_a_cuda_request_without_a_cuda_build_is_refused() -> None:
    """A GPU plan must not compute on CPU while the manifest claims a GPU.

    The first GPU attempt did exactly that in effect: the model stayed on CPU while the evaluation
    moved the inputs to CUDA, and the run died inside the first embedding lookup. The refusal is
    explicit, so the failure names the cause instead of the symptom.
    """
    from spectraquant.cloud.plan_runner import load_model

    plan = load_plan(TIER1_PLAN)
    if torch.cuda.is_available():  # pragma: no cover - this workstation has no CUDA device
        pytest.skip("this machine has a CUDA build; the refusal cannot be exercised here")

    with pytest.raises(RuntimeError, match="no CUDA support"):
        load_model(plan, device="cuda")


def test_a_streaming_read_without_a_cap_is_refused() -> None:
    """An unbounded streaming read has no end, so the cap is required, not optional.

    Regression for the sixth GPU attempt: the training corpus read C4's `en` train split (hundreds of
    gigabytes) without streaming, and the run sat downloading for over an hour without reaching a
    single optimiser step.
    """
    from spectraquant.cloud.plan_data import load_plan_texts
    from spectraquant.experiment_plan import DatasetRole

    ref = DatasetRole(
        name="allenai/c4",
        revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        license="odc-by",
        config="en",
        roles=["train"],
    )
    with pytest.raises(ValueError, match="requires max_documents"):
        load_plan_texts(ref, split="train", dataset_config="en", stream=True)


def test_the_training_corpus_streams_the_train_role_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the huge train split streams; the bounded dev split is read normally."""
    from spectraquant.cloud import plan_runner

    seen: list[dict] = []

    def _record(ref, **kwargs):
        seen.append({"role_dataset": ref.name, **kwargs})
        return [" ".join(f"t{index}" for index in range(40)) for _ in range(2)]

    monkeypatch.setattr(plan_runner, "load_plan_texts", _record)
    plan = load_plan(REPRO_PLAN)

    class _Tokenizer:
        def __call__(self, text: str, return_tensors: str = "pt"):
            import torch as _torch

            return type("Enc", (), {"input_ids": _torch.zeros((1, 40), dtype=_torch.int64)})()

    plan_runner._training_corpus(plan, _Tokenizer(), seq_len=8, max_documents=4)

    by_dataset = {entry["role_dataset"]: entry for entry in seen}
    train = by_dataset["allenai/c4"]
    dev = by_dataset["EleutherAI/wikitext_document_level"]
    assert train["stream"] is True
    assert train["split"] == "train"
    assert dev["stream"] is False
    assert dev["split"] == "validation"


def test_a_non_trainable_arm_is_unaffected_by_a_trainable_arm_before_it(
    tmp_path: Path, offline: None
) -> None:
    """Regression: a trained arm's representation leaked into the next arm.

    A trainable arm *replaces* layers, which no `load_state_dict` can undo, and the loop reused one
    model instance. On the first GPU run `rank_then_quant` reported exactly the trained LoftQ arm's
    perplexity (102.3583 to four decimals) instead of its own.

    The check is exact: a non-trainable arm's number must be identical whether it runs alone or after
    a trained arm, because the fixture is deterministic.
    """
    alone = run_plan(
        REPRO_PLAN, arms=["rank_then_quant"], out_dir=tmp_path / "alone", progress=lambda _: None
    )
    after = run_plan(
        REPRO_PLAN,
        arms=["loftq", "rank_then_quant"],
        out_dir=tmp_path / "after",
        progress=lambda _: None,
    )

    alone_metrics = validate_manifest_file(alone.runs[0].manifest_path)["metrics"]
    after_runs = {run.arm: run for run in after.runs}
    after_metrics = validate_manifest_file(after_runs["rank_then_quant"].manifest_path)["metrics"]

    assert alone_metrics["perplexity"] == after_metrics["perplexity"]
    # And the two arms must not be reporting the same model as each other.
    loftq_metrics = validate_manifest_file(after_runs["loftq"].manifest_path)["metrics"]
    assert after_metrics["perplexity"] != loftq_metrics["perplexity"]
