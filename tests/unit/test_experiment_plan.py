"""Tests for the frozen cloud experiment plans (``spectraquant.experiment_plan``).

These tests are the machine-checkable half of the preregistration freeze: they assert that every
committed plan validates, that its grid matches the predeclared grid, that the harness pin matches
the evaluation protocol, and that the guard rails (no local training, no unpaid-but-paid plan) hold.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from spectraquant.experiment_plan import PlanConfig, list_plans, load_plan, plan_path

HARNESS_COMMIT = "ddd67220430a2470529f25fd5c05a576ca1057a0"
PREDECLARED_BITS = [2, 3, 4, 8]
PREDECLARED_RANKS = [2, 4, 8, 16, 32]
PREDECLARED_GROUP_SIZES = [32, 64, 128]


def test_committed_plans_exist_and_validate() -> None:
    plans = list_plans()
    assert plans, "no plans committed: the preregistration freeze checklist requires them"
    for path in plans:
        plan = load_plan(path)
        assert isinstance(plan, PlanConfig)
        assert plan.name


def test_expected_plans_are_present() -> None:
    names = {load_plan(p).name for p in list_plans()}
    assert {
        "tier1_smollm2_135m",
        "tier2_tinyllama_1_1b",
        "repro_lr_qat_loftq_smollm2_135m",
    } <= names


def test_grid_matches_the_predeclared_preregistration_grid() -> None:
    for path in list_plans():
        plan = load_plan(path)
        assert set(plan.grid.bits) <= set(PREDECLARED_BITS), path
        assert set(plan.grid.ranks) <= set(PREDECLARED_RANKS), path
        assert set(plan.grid.group_sizes) <= set(PREDECLARED_GROUP_SIZES), path
        assert plan.grid.budget_ladder_bytes == sorted(plan.grid.budget_ladder_bytes)


def test_every_plan_pins_the_evaluation_harness_and_task_suite() -> None:
    for path in list_plans():
        plan = load_plan(path)
        assert plan.harness["commit"] == HARNESS_COMMIT, path
        assert plan.harness["tasks"] == [
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "piqa",
            "winogrande",
            "boolq",
        ], path


def test_models_and_datasets_are_pinned_with_licences() -> None:
    for path in list_plans():
        plan = load_plan(path)
        for model in plan.models:
            assert len(model.revision) >= 7, path
            assert model.license, path
            assert model.parameters > 0, path
        for dataset in plan.datasets:
            assert len(dataset.revision) >= 7, path
            assert dataset.license, path


def test_no_plan_hosts_trainable_work_locally() -> None:
    for path in list_plans():
        plan = load_plan(path)
        if any(arm.trainable for arm in plan.arms):
            assert plan.substrate != "local_cpu", path


def test_free_tier_plans_declare_zero_authorization() -> None:
    for path in list_plans():
        plan = load_plan(path)
        if plan.cost.free_tier_only:
            assert plan.cost.max_cost_authorized_usd == 0.0, path


def test_seed_floor_is_respected_and_seeds_unique() -> None:
    for path in list_plans():
        plan = load_plan(path)
        assert len(plan.seeds.reported) >= plan.seeds.floor, path
        assert len(set(plan.seeds.reported)) == len(plan.seeds.reported), path
        assert plan.seeds.master >= 0


def test_measurement_classes_are_declared() -> None:
    for path in list_plans():
        plan = load_plan(path)
        assert plan.measurement_classes
        assert all(1 <= cls <= 5 for cls in plan.measurement_classes), path


# ---------------------------------------------------------------- schema guard rails


def _minimal_document() -> dict:
    return {
        "name": "unit-test-plan",
        "substrate": "colab",
        "tier": 1,
        "models": [
            {
                "id": "org/model",
                "revision": "0123456789",
                "license": "apache-2.0",
                "parameters": 1000,
                "role": "test",
            }
        ],
        "datasets": [
            {"name": "org/data", "revision": "0123456789", "license": "mit", "roles": ["train"]}
        ],
        "seeds": {"master": 0, "reported": [0, 1, 2], "floor": 3},
        "grid": {
            "bits": [4, 8],
            "ranks": [2, 4],
            "group_sizes": [32],
            "budget_ladder_bytes": [100, 200],
        },
        "arms": [{"name": "a", "kind": "ptq_uniform", "trainable": False}],
        "harness": {"commit": HARNESS_COMMIT, "tasks": ["piqa"]},
        "cost": {"platform_hours_max": 1.0, "max_cost_authorized_usd": 0.0, "free_tier_only": True},
        "measurement_classes": [2],
    }


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    document = _minimal_document()
    document["typo_field"] = 1
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="typo_field"):
        load_plan(path)


def test_local_cpu_with_a_trainable_arm_is_rejected(tmp_path: Path) -> None:
    document = _minimal_document()
    document["substrate"] = "local_cpu"
    document["arms"] = [{"name": "a", "kind": "lr_qat", "trainable": True}]
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="local_cpu"):
        load_plan(path)


def test_paid_plan_without_authorization_is_rejected(tmp_path: Path) -> None:
    document = _minimal_document()
    document["cost"] = {
        "platform_hours_max": 10.0,
        "max_cost_authorized_usd": 0.0,
        "free_tier_only": False,
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="max_cost_authorized_usd"):
        load_plan(path)


def test_seed_count_below_floor_is_rejected(tmp_path: Path) -> None:
    document = _minimal_document()
    document["seeds"] = {"master": 0, "reported": [0, 1, 2], "floor": 5}
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="floor"):
        load_plan(path)


def test_unsorted_grid_is_rejected(tmp_path: Path) -> None:
    document = _minimal_document()
    document["grid"]["ranks"] = [4, 2]
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="ascending"):
        load_plan(path)


def test_plan_path_reports_what_it_tried() -> None:
    with pytest.raises(FileNotFoundError, match="tried"):
        plan_path("does-not-exist-plan")


def test_a_perplexity_dataset_must_declare_its_config() -> None:
    """A multi-config repository cannot be loaded without one - fail locally, not on the platform.

    Regression for the first real cloud run: the runner reached the datasets library without a config
    name and died with "Config name is missing", minutes after the environment had been built.
    """
    document = _minimal_document()
    document["datasets"] = [
        {
            "name": "org/data",
            "revision": "0123456789",
            "license": "mit",
            "roles": ["test_perplexity"],
        }
    ]
    path = Path(tempfile.mkdtemp()) / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="declares no config"):
        load_plan(path)

    document["datasets"][0]["config"] = "wikitext-2-raw-v1"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert load_plan(path).datasets[0].config == "wikitext-2-raw-v1"


def test_committed_plans_pin_their_dataset_configs() -> None:
    for path in list_plans():
        plan = load_plan(path)
        for dataset in plan.datasets:
            if "test_perplexity" in dataset.roles:
                assert dataset.config, (plan.name, dataset.name)


# ---------------------------------------------------------------- training schedule


def _training_document() -> dict:
    return {
        "steps": 200,
        "learning_rate": 2.0e-4,
        "batch_size": 8,
        "seq_len": 512,
        "warmup_steps": 20,
        "grad_clip": 1.0,
        "optimizer": "adamw",
        "eval_every": 50,
        "checkpoint_every": 100,
    }


def test_a_trainable_arm_requires_a_training_schedule(tmp_path: Path) -> None:
    document = _minimal_document()
    document["arms"] = [
        {
            "name": "spectraquant_regularized",
            "kind": "proxy_allocated_regularized",
            "trainable": True,
        }
    ]
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="no `training` block") as excinfo:
        load_plan(path)
    message = str(excinfo.value)
    assert "unit-test-plan" in message
    assert "spectraquant_regularized" in message


def test_a_training_schedule_without_a_trainable_arm_is_refused(tmp_path: Path) -> None:
    document = _minimal_document()
    document["training"] = _training_document()
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="unused"):
        load_plan(path)


def test_warmup_longer_than_the_schedule_is_refused(tmp_path: Path) -> None:
    document = _minimal_document()
    document["arms"] = [{"name": "a", "kind": "lr_qat", "trainable": True}]
    document["training"] = {**_training_document(), "warmup_steps": 201}
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="warmup_steps"):
        load_plan(path)


def test_unknown_field_inside_training_is_refused(tmp_path: Path) -> None:
    document = _minimal_document()
    document["arms"] = [{"name": "a", "kind": "lr_qat", "trainable": True}]
    document["training"] = {**_training_document(), "typo_field": 1}
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(Exception, match="typo_field"):
        load_plan(path)


def test_committed_plans_declare_the_documented_training_schedule() -> None:
    expected = {
        # Tier-1's schedule is pinned from the measured step cost too (A-0016 moved it to cuda).
        "tier1_smollm2_135m": (300, 1.0e-4, 8, 512, 50, 1.0, "adamw", 100, 250, 5000),
        "tier2_tinyllama_1_1b": (200, 2.0e-4, 4, 1024, 20, 1.0, "adamw", 50, 100, 5000),
        # The reproduction plan's schedule is pinned from a measurement (2.17 s/step on a T4) and its
        # corpus size is pinned too: 300 steps at batch 8 over 5 000 streamed documents is about one
        # epoch. The first cell left the corpus size on the command line (200 documents -> 99
        # windows) against 500 steps, so the factors memorised the corpus.
        "repro_lr_qat_loftq_smollm2_135m": (300, 1.0e-4, 8, 512, 50, 1.0, "adamw", 100, 250, 5000),
        # The frozen M3 protocol's own budget (reproduction-plan.md §3): 1000 steps x 512 tokens,
        # 10 % warmup, the pilot's five seeds.
        "m3_tier1_smollm2_135m": (1000, 1.0e-4, 8, 512, 100, 1.0, "adamw", 100, 250, 5000),
    }
    for path in list_plans():
        plan = load_plan(path)
        schedule = plan.training
        assert schedule is not None, path
        assert (
            schedule.steps,
            schedule.learning_rate,
            schedule.batch_size,
            schedule.seq_len,
            schedule.warmup_steps,
            schedule.grad_clip,
            schedule.optimizer,
            schedule.eval_every,
            schedule.checkpoint_every,
            schedule.corpus_documents,
        ) == expected[plan.name], path


def test_the_m3_plan_matches_the_frozen_reproduction_protocol() -> None:
    """The M3 plan is a transcription of `docs/research/reproduction-plan.md` §3, so pin it.

    The plan exists because the protocol's eight arms were never implemented. Its numbers are copied
    from the frozen text, not chosen here; a drift between the two would silently change what the
    reproduction tests, so the transcription is asserted rather than trusted.
    """
    plan = load_plan("configs/m3/tier1_smollm2_135m.yaml")

    # §3 "Token budget per arm": 1000 optimizer steps x 512 tokens; seeds {0..4} for the pilot.
    assert plan.training is not None
    assert plan.training.steps == 1000
    assert plan.training.seq_len == 512
    assert plan.training.batch_size == 8
    assert list(plan.seeds.reported) == [0, 1, 2, 3, 4]
    assert plan.seeds.floor == 5

    # §3.1: R1 at 2-bit NF-style codebook, block 64, rank 16 for the model-level arms.
    for name in ("r1_std_2bit", "r1_loftq_2bit", "r1_loftq_2bit_t1"):
        arm = next(a for a in plan.arms if a.name == name)
        assert arm.point["bits"] == 2, name
        assert arm.point["block_size"] == 64, name
        assert arm.point["rank"] == 16, name
        assert arm.trainable is True, name

    # §3.1: the two LoftQ arms differ from the standard one only by their initialisation, so they
    # must bind the same rank/bits/block, and T = 5 vs T = 1 respectively.
    assert next(a for a in plan.arms if a.name == "r1_loftq_2bit").point["loftq_iterations"] == 5
    assert next(a for a in plan.arms if a.name == "r1_loftq_2bit_t1").point["loftq_iterations"] == 1

    # §3.2: R2 at 4-bit symmetric group 128; LR-QAT carries rank 32 and a learned step size; the two
    # reference arms are eval-only.
    for name in ("r2_rtn_4bit", "r2_lrqat_4bit", "r2_fullqat_4bit"):
        arm = next(a for a in plan.arms if a.name == name)
        assert arm.point["bits"] == 4, name
        assert arm.point["group_size"] == 128, name
    assert next(a for a in plan.arms if a.name == "r2_lrqat_4bit").point["rank"] == 32
    assert next(a for a in plan.arms if a.name == "r2_rtn_4bit").trainable is False
    assert next(a for a in plan.arms if a.name == "r2_fp16").trainable is False

    # §3.1/§3.2: both learning-rate grids, unioned, searched on the validation split only.
    assert set(plan.training.learning_rate_grid) == {
        1.0e-5,
        5.0e-5,
        1.0e-4,
        3.0e-4,
        1.0e-3,
        1.0e-2,
    }

    # The plan declares the frozen model and dataset pins, not new ones.
    assert plan.models[0].id == "HuggingFaceTB/SmolLM2-135M"
    assert plan.models[0].revision == "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    roles = {role for dataset in plan.datasets for role in dataset.roles}
    assert {"train", "development", "test_perplexity"} <= roles
    # The test split is never a training or calibration source.
    for dataset in plan.datasets:
        if "test_perplexity" in dataset.roles:
            assert "train" not in dataset.roles and "calibration" not in dataset.roles
