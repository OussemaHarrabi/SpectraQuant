# The SpectraQuant research storyline

The central question is how local compression decisions translate into whole-model
quality under an actual stored-memory constraint. The project began with joint
low-rank and low-bit allocation, and keeps its negative results as part of that
question's answer.

| Stage | Question | Evidence and honest conclusion |
|---|---|---|
| Original method | Does allocated factor training beat uniform quantization at equal target bytes? | Single-seed Smol135M study is negative; LR-QAT reduced reproduction is partial and uses more storage. |
| Deployment correction, A-0023 | Does scoring the actual reloaded residual representation rescue allocation? | No; three calibration subsets underperform uniform INT4 on the previously used WikiText benchmark. |
| Joint diagnosis, A-0024 | Do summed local gains predict joint loss and transfer? | Predictions overstate joint calibration gains; all six old allocations worsen disjoint-block loss. |
| Simpler control, A-0024 | Can a fixed conventional criterion improve quality at identical bytes? | Groupwise clipping gives1.93% lower new C4 holdout perplexity on one model, with equal full-checkpoint bytes. |
| Transfer, A-0025 | Does that fixed criterion survive new scale, architecture and text domain? | Completed: 2/4 positive cells, geometric mean reduction0.575%, frozen consistency gate failed. Secondary activation-weighted controls improve C4 on both architectures, with Pythia PG19 regressions. |

The scientific contribution so far is an audited diagnosis and bounded positive
control. A conventional clipping result is not an original allocation-algorithm
gain. The transfer test measures the control in four prescribed model/corpus cells; either
outcome completes a coherent result rather than selecting an unrelated benchmark.

The report must show every primary cell, the fixed transfer gate, real file sizes,
document-level uncertainty and exact provenance. Secondary arms measure sensitivity
to grouping and calibration; they must not replace a failed primary comparison after
test exposure. Original full Tier2, independent training replications, modern
stronger PTQ algorithms and accelerated GPU kernels remain separate open gates.

The bounded study is complete, including verified public evidence and generated
figures. Read `docs/report/spectraquant-research-synthesis.md`. The positive
secondary C4 result does not rescue the failed primary endpoint. The optional
kernel phase is deferred under its predeclared transfer gate; no extra GPU job.
Additional algorithm changes require a new protocol and an unexposed holdout.
