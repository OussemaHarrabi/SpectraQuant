# Trainable-arm diagnostic on the cloud GPU, 2026-10-09/10

> **This is not an M3 result.** The frozen M3 protocol
> (`docs/research/reproduction-plan.md` §3) specifies eight arms that are **not implemented** in the
> runner: `R1-FP16-LoRA`, `R1-std-2bit`, `R1-loftq-2bit`, `R1-loftq-2bit-T1` (2-bit NF-style
> codebook, block 64, rank 16), `R2-FP16`, `R2-RTN-4bit-g128`, `R2-LRQAT-4bit-g128` (rank 32 plus a
> learned step size, `Φ₀` downcast to Q4.4), `R2-fullQAT-4bit-g128`, plus a validation-split
> learning-rate search. What ran here is the plans' **default grid** (rank 8, uniform 4-bit
> per-group 32), one seed, 300 steps. It tests the *plumbing* and the *direction*, not the
> predeclared bit width or the verdict rule. The M3 verdict is therefore still open
> (`docs/coordination/claim-contradictions.md`).

**Runs** `repro_lr_qat_loftq_smollm2_135m-cloud` (Kaggle, Tesla T4, CUDA 12.8) ·
**registry state** `validated` (two collected bundles: the isolation-fix run and the pinned-schedule
run) · **code commits** `363adf78…` (final) and `4fec57d…` (first validated GPU run).

Two collected bundles are reported. The first (500 steps over a 99-window corpus) was a **negative,
attributable** outcome caused by our own corpus sizing. The second (300 steps over 2 841 windows,
5 000 streamed documents) is the **corrected** run, and its direction is positive.

## 1. What was measured

Model `HuggingFaceTB/SmolLM2-135M` @ `93efa2f0…`, WikiText-2 test (document level, config
`wikitext-2-raw-v1`, 20 000-token cap), dev = WikiText-2 validation, train = a streamed head of
`allenai/c4` (`en`). Schedule pinned from a measurement: 500 steps, lr 1.0e-4, batch 8, seq 512,
50-step warmup. Both trained arms run on the GPU; the compression math runs on CPU by design.

| arm | perplexity at init | after training | accounted bytes | class | train wall | run |
|---|---|---|---|---|---|---|
| `fp16_reference` | — | 14.0183 | 212 336 640 | — | — | both |
| `lr_qat` (rank 8, 4-bit) | **17.722** | **17.0694** | 64 604 160 | 2 | 618 s | corrected (300 steps, 2 841 windows) |
| `loftq` (rank 8, 4-bit, T=3) | **17.793** | **16.6936** | 64 604 160 | 2 | 625 s | corrected |
| `rank_then_quant` (rank 8, 4-bit factors) | — | degenerate (5.0e19) | 1 598 400 | 2 | — | corrected |
| `lr_qat` (superseded) | 17.722 | 43.0962 | 64 604 160 | 2 | 1062 s | first (500 steps, 99 windows) |
| `loftq` (superseded) | 17.793 | 269.6186 | 64 604 160 | 2 | 1072 s | first |

Corrected-run dev-loss trajectory (per 100 steps) — flat, i.e. no overfitting:

```
lr_qat : 100:3.242  200:3.242  300:3.253   final task loss 3.045
loftq  : 100:3.198  200:3.223  300:3.246   final task loss 2.996
```

Dev-loss trajectory (recorded per 100 steps), which is the evidence for the diagnosis:

```
lr_qat : 100:3.3168  200:3.4786  300:3.6982  400:3.9962  500:4.3478   final task loss 0.4661
loftq  : 100:3.5124  200:4.3353  300:4.9668  400:5.6575  500:6.0777   final task loss 0.0630
```

## 2. Reading

* **The initialisations are sound.** Both arms start at ≈17.7 perplexity against a 14.0183 fp16
  reference — a 26 % degradation from rank-8 + 4-bit compression, which is the ordinary cost of that
  much compression and is *not* a broken artifact. (For contrast, untrained rank-8 truncation was
  degenerate at 1.4e16: the low-rank *preparation* is what makes rank 8 viable at all.)
* **With a corpus-sized schedule, training improves both arms in this single run per arm** (corrected
  run; one seed, one run per arm — not a distribution, and not an M3 verdict): LoftQ 17.793 → **16.6936**
  and LR-QAT 17.722 → **17.0694**, against an fp16 reference of 14.0183 and a *flat* dev-loss trajectory.
  The direction matches the published trends; the magnitude cannot be scored against the frozen rule
  because the arms are not the predeclared ones (see the note at the top).
* **The superseded run destroyed both arms, and the corpus explains it.** The streamed train corpus was only
  **99 windows** of 513 tokens (~50 k tokens) — a 200-document cap — while 500 steps at batch 8 draw
  4 000 sequences from it. The task loss collapsed to 0.47 (`lr_qat`) and 0.06 (`loftq`), i.e. the
  factors memorised those 99 windows, and dev loss rose monotonically in both arms. This is a
  corpus-sizing defect in our own invocation, not a property of LR-QAT or LoftQ.
* **The record made the attribution possible.** `training.perplexity_at_init` was added *because* the
  first GPU run could not be interpreted: without it, "43.1" and "269.6" say nothing about whether
  the initialisation or the schedule was at fault. With it, the statement is precise: the
  initialisation is good, the schedule on this corpus is harmful.
* **`rank_then_quant` is degenerate** (5.0e19, flagged). Its stored object is only the two quantized
  rank-8 factors (1.6 MB) with no fp16 base, so it is not a fair equal-memory point against the
  trained arms (64.6 MB) — it is a *lower-bound* artifact and is reported as such.

## 3. Defects this phase exposed

All were invisible to the 1034-test fake-runner suite, and each is now pinned by a regression test.

| # | Defect | Evidence |
|---|---|---|
| 1 | `uv run` re-synced the env on every stage, restoring CPU torch after the CUDA install | `Torch not compiled with CUDA enabled` at the training step |
| 2 | Nothing verified the CUDA build was the installed one | fixed by the install-cell check, which then caught #3 in seconds |
| 3 | `torch==2.14.1` matched `2.14.1+cpu` (PEP 440), so the CUDA install was a no-op | `torch 2.14.1+cpu cuda False` |
| 4 | The cu128 index has no 2.14.1 for cp311 (it tops out at 2.11.0) | `No solution found when resolving dependencies` |
| 5 | `load_model` never moved the model to the requested device | `index is on cuda:0, different from other tensors on cpu` |
| 6 | The compression primitives are CPU-only; a GPU run pushed CUDA tensors into them | `w is on device cuda:0; only CPU execution is supported` |
| 7 | The train corpus read C4's `en` train (hundreds of GB) without streaming | run sat >1 h without one optimiser step; operator-stopped |
| 8 | A trainable arm's replaced layers leaked into the next arm | `loftq` and `rank_then_quant` reported *identical* perplexity (102.3583) |
| 9 | A run could not be resubmitted from `validated`/`rejected`/`failed`/`collected` | each refused in turn, blocking the corrected run |

## 4. Next action (not a claim)

The M3 cell needs a **corpus-sized schedule**, not a different method: stream a materially larger C4
head (or pin the corpus size in the plan rather than in the invocation, which is the actual defect —
`--max-documents` currently lives only on the command line) and set steps so that
`steps × batch_size` stays a fraction of the corpus windows, with dev-based early stopping. Until
that run exists, **no reproduction claim is made** for LR-QAT or LoftQ, in either direction.
