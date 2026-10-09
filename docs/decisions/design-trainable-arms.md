# Design note — trainable arms in the cloud plan runner

Status: **proposed** (implementation not started). Required by `AGENTS.md` §3: the interface below is
shared between the plan schema, the cloud runner and the training loop, so it is specified here
before it is implemented.

## 1. Problem

`configs/{tier1,tier2,repro}/*.yaml` declare trainable arms (`lr_qat`, `loftq`,
`spectraquant_regularized`, and the Tier-2 `spectraquant`/`qlora`), and `cloud run-plan` skips every
one of them with a "needs the M5 training loop" reason. The loop itself **already exists**
(`spectraquant.training.loop.train_language_model`: optimiser steps, checkpoint/resume,
`PreparationObjective`, `TaskLoss`, `QuantSpec`, deterministic seeding, manifest emission). What is
missing is the wiring, and three pieces of information the plans do not carry:

1. a **training schedule** (steps, learning rate, batch/sequence shape, warmup, grad clip,
   optimiser) — the frozen plans pin grids for compression but nothing for training;
2. the **corpus** the arm trains on (the plan pins `allenai/c4` with roles `train, calibration`);
3. the **method-specific initialisation** that distinguishes the arms (LR-QAT's low-rank auxiliary
   weight, LoftQ's alternating quantize-SVD initialisation).

## 2. Interface

### 2.1 Plan schema additions (frozen-plan compatible)

```yaml
training:                      # new top-level block, optional
  steps: 400
  learning_rate: 2.0e-4
  batch_size: 8
  seq_len: 512
  warmup_steps: 20
  grad_clip: 1.0
  optimizer: adamw
  eval_every: 50               # dev-NLL checkpoints; never the test split
```

* `PlanConfig.training: TrainingSchedule | None`. A plan with **no** `training` block may not declare
  a trainable arm — validated at load time, so the failure is local and immediate rather than a
  skip reason at run time.
* Hyperparameters are **per plan**, not per arm: the arms must differ only in method, or the
  comparison is not matched (`AGENTS.md` §4.5). If an arm needs a different schedule, that is a
  second plan, and the report must say so.
* Every value is validated positive; `seq_len ≤ model.max_position_embeddings` is checked when the
  model is loaded (a plan cannot know it).

### 2.2 Runner wiring

`plan_runner.run_plan` gains the trainable kinds in `IMPLEMENTED_ARM_KINDS`, and dispatches each to a
new `_train_arm(model, arm, schedule, data, plan, ...)` that:

1. builds `SequenceData` from the plan's `train`-role corpus (tokenised with the pinned tokenizer,
   document boundaries preserved, deterministic order, `checksum` recorded);
2. applies the arm's initialisation **before** training: `low_rank_only`-style SVD for the plain
   `spectraquant` arm, the alternating quantize-SVD for `loftq`, the quantized-main/low-rank-auxiliary
   pair for `lr_qat`, the regularized objective for `spectraquant_regularized`;
3. calls `train_language_model(...)` with the schedule, the arm's `PreparationObjective` (or `None`),
   `measurement_class=2`, and `resume_from` when the registry has a checkpoint for the arm;
4. evaluates with the existing `token_level_perplexity` on the **test** split, exactly as the
   non-trainable arms do, so the numbers are comparable;
5. writes the same manifest shape, adding `training.*` (schedule actually used, steps completed,
   dev-NLL trajectory, wall time, peak RSS) and `training.initialisation` (what the arm applied
   before step 0).

Invariants:

* **The test split is never used for a decision.** Dev NLL drives any early stopping; the test
  perplexity is computed once, after training, and recorded. `eval_every` evaluates dev only.
* **Resume is honest**: a resumed run records `training.resumed_from` and the total steps reached;
  a run that reaches fewer steps than declared is `status: failed` with the step count, never a
  silently shorter run presented as complete.
* **Determinism**: the plan's `seeds.master` derives the torch/numpy/python seeds
  (`spectraquant.training.seeding`); the same seed, schedule and commit reproduce the same dev-NLL
  trajectory on the same hardware class. Cross-hardware bit-for-bit equality is **not** claimed
  (cuDNN kernels differ); the claim is same-substrate reproducibility, recorded in the manifest.
* **Memory honesty**: training-time memory, stored checkpoint size and deployed representation stay
  three separate numbers (`AGENTS.md` §4.2). `training.peak_rss_mb` is the first; the checkpoint
  bytes are the second; the accounted compression bytes remain the third.

### 2.3 Substrate

Trainable arms require the GPU substrate. `PlanConfig.substrate == "local_cpu"` is already refused
for a trainable plan (`experiment_plan.py`). Verified 2026-10-09: the Kaggle free tier provides
**2× Tesla T4, sm_75** (`docs/research/backend-capability.md` §9) — fp16 and int8 yes, **bf16 no**.
The T4 is the declared device for M3/M5; a T4 number is never presented as comparable to a published
A100/H100 number.

## 3. What this does NOT change

* No new scientific claim: M3 is a *reproduction* of published trends (LR-QAT, LoftQ) at Tier-1 scale
  under the predeclared tolerance (same sign, ≥25 % of the published effect size), and M5 is the H3
  cell. Neither becomes confirmatory before its bundle is collected and validated.
* The non-trainable arms' numbers are unaffected; the collected comparator matrix
  (`docs/results/tier1-cloud-run-2026-10-09.md`) stays as measured.
* No new dependency: the loop, the objective, the quantizer and the seeding module all exist.

## 4. Acceptance criteria

1. A plan declaring a trainable arm **without** a `training` block fails to load, with a message
   naming the file and the missing block.
2. `cloud run-plan --plan configs/repro/lr_qat_smollm2_135m.yaml` runs `lr_qat` and `loftq` end to
   end on the pinned Tier-1 corpus and writes one manifest per (arm, seed), each carrying the
   schedule used, the steps completed and the dev-NLL trajectory.
3. A run interrupted after N steps and resumed reaches the declared total steps and records
   `training.resumed_from`; the resumed run's test perplexity equals the uninterrupted run's for the
   same seed on the same substrate (verified once, reported).
4. The equal-memory gate still refuses byte-unequal pairs: a trainable arm's reported bytes are the
   *stored* bytes of the artifact it produces, not the fp32 training state.
5. Local (CPU) tests cover the wiring with a tiny synthetic model — no network, no GPU — and the
   GPU path is exercised only on the cloud substrate.

## 5. Open questions for the owner

1. **Tier-1 training budget.** The Tier-1 plan's trainable arm needs a schedule; the preregistration
   (§4) estimated ≈0.35–0.44 h per WikiText-2 epoch *locally*, and the cloud estimate has not been
   measured. Choosing `steps` before measuring a step's cost risks either a wasted GPU-hour budget or
   an under-trained arm. Proposal: run one short GPU timing cell (≈100 steps) first, record it, then
   pin the schedule from the measurement. This is a *cost* measurement, not a result.
2. **Which arm is Tier-1's trainable comparator.** `lr_qat` and `loftq` are the M3 reproduction arms;
   `spectraquant_regularized` is the M5 method arm. The plan currently has no schedule for any of
   them, so the schedule must be pinned for all three at once, or the plans split.
