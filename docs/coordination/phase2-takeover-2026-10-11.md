# Phase 2 takeover, 11 October 2026

## 1. Scope completed

Recovered the original checkout at a603a4c and isolated the approved continuation
in a separate checkout/branch. Original uncommitted CLI/preflight changes were
preserved. User confirmed 20 remaining Kaggle GPU-hours. Both older Kaggle kernels
were COMPLETE before submission. Implemented A-0023 rather than rerunning the
negative frozen experiment. Submitted one private, time-bounded run.

## 2. Files changed

Scientific runner `src/spectraquant/cloud/phase2.py`, JSON config under
`configs/phase2`, CLI phase2 command, generated-notebook resource accounting,
aggregate manifest handling, cloud generation/submission and safe resumption
scripts, offline result verification/reporting, synthetic correctness tests,
ADR-0023 and append-only preregistration amendment A-0023.

## 3. Commands executed

- Existing uv-managed Python environment, with this checkout's src on PYTHONPATH.
- CPU pytest, Ruff lint/format, Pyright with explicit existing interpreter.
- Process-local Git safe.directory for the owned checkout, never wildcard/global.
- `git push -u origin research/phase2-global-damage`.
- `python scripts/experiments/phase2_cloud.py --submit` through official Kaggle CLI.
- Read-only Kaggle status checks and GitHub Actions status.

## 4. Tests and results

Full CPU suite: 1,220 passed, 3 documented skips before review/provenance fixes.
Affected suite after fixes: 69 passed. Analysis integrity tests: 3 passed.
Ruff lint/format clean, Pyright zero errors. GitHub CI/docs/security all green
for scientific-run commit fe39413153c41dfc8e23ab2d88bfa9c4f217f72e.
No local research model was loaded or run. Independent read-only review found
and prompted fixes to provenance, checkpoint collection, duplicate submissions
and failure-score preservation.

## 5. Evidence created

- Private remote id: `oussemaharrabi/sq-phase2-global-20261011`, version1.
- Immutable scientific code SHA: fe39413153c41dfc8e23ab2d88bfa9c4f217f72e.
- Local spec/submission/registry: `artifacts/runs/sq-phase2-global-20261011/`.
- Runner9,500s; notebook12,000s. At default dualT4, maximum first-session
  allocated device-hours6.67. This is a conservative bound, not Kaggle quota API.
- Remote research results pending at this handoff; no new efficacy claim yet.

## 6. Assumptions

Free Kaggle quota is user-confirmed20h. Phase2 is exploratory after negative
historical results, with one checkpoint and three calibration subsets. Known
prior WikiText2 exposure prevents describing this as an untouched blind test.

## 7. Risks / unresolved questions

Additive marginal damage may fail under joint layer interactions. Tiny-model
improvements cannot establish Tier2 generalization. Quality uses dense fp16
reconstruction; packed bytes do not imply low-bit runtime or acceleration.
No trainable method or downstream benchmark in the initial attempt.

## 8. Recommended next action

Resume existing run; never resubmit its id:

```powershell
uv run python scripts/experiments/phase2_resume.py --fetch
```

This checks status, downloads artifacts/executed notebook, performs schema and
checksum collection, verifies every checkpoint and emits a scoped research
report. Evaluate the validation-selected family and all declared limits before
spending another GPU-hour. Preserve the >=3h reserve. Failure merits a specific
root-cause repair, not an unbounded repeat. A positive pilot should next receive
a frozen transfer check on a second pinned model, rather than a test-driven sweep.
