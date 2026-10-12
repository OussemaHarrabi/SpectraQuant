# SpectraQuant takeover: completed and verified controlled follow-up

## 1. Scope completed

Phase2 completed on Kaggle and was collected independently. Both proposed
allocation families failed validation; the baseline was retained. Phase3 was
implemented, tested, independently reviewed and submitted once. It completed
successfully on T4x2, version1, and all expected scientific artifacts were
collected and verified. The conventional clipping control improves heldout
perplexity by1.93% at exactly equal target and full checkpoint bytes.

Original checkout, at a603a4c, is preserved.
Its uncommitted CLI, cloud/dryrun and cloud-preflight-test edits were untouched.
Continuation uses an isolated checkout on branch `research/phase2-global-damage`.
Never overwrite the original edits. Machine-specific paths stay in the local
handoff outside this repository.

## 2. Files changed

Scientific Phase2/Phase3 runners, frozen JSON configurations and amendments,
generated Kaggle notebooks, submission/resume tooling, manifest provenance,
checkpoint verification, path-safe wrapped collection, CLI commands and tests.
Research report, README, results tables, allocation-instability analysis and
compute ledger reflect the completed negative result. The separate career
record records verified Phase3 evidence with the same measurement boundaries.

## 3. Commands executed

Existing uv-managed Python environment with this checkout's src on PYTHONPATH;
CPU pytest, Ruff, Pyright, official Kaggle CLI push/status/output/pull, Git
commits/pushes, GitHub CI checks and draft PR creation. Git ownership overrides
are process-local, never global or wildcard. No research model ran locally.

## 4. Tests and results

Full CPU suite:1231 passed,3 documented skips. After the collector path fix,
36 focused tests passed. Ruff/format/Pyright clean. All GitHub CI/docs/security
checks pass at292d5a0. Independent review findings addressed, including a
pipeline watchdog and verifying dynamic checkpoints before registry acceptance.
Final collection verified11 expected artifacts and6 checkpoint files; selection
identity, block disjointness, cap compliance, token and byte parity were checked.
The report renderer now distinguishes one clipping comparison from calibration
subset replications, covered by a failing-first regression. Final affected
report/Phase3/collection suite:37passed. Ruff/format clean; Pyright0errors with
the existing uv-managed environment explicitly selected by --venvpath.

## 5. Evidence created

- Phase2 code: `fe39413153c41dfc8e23ab2d88bfa9c4f217f72e`.
- Phase2 remote: `oussemaharrabi/sq-phase2-global-20261011`, version1.
- Source report SHA256: `1cb35a549391fb47c0c7f679267cecb287e6e29a97d2200d3944c304d5d34456`.
- Source scores SHA256: `b2e368fb7cd037d24af43f63b8cc40cc55151afe42d106c2f83ba3c8b14d92eb`.
- Phase2:25 artifacts and21 checkpoint files verified;49,951 predicted WikiText
  test tokens. Uniform4 PPL22.6213; alternatives23.8709–25.9582,5.52–14.75% worse.
- Source manifest and checkpoint files are under the git-ignored collected bundle
  `artifacts/runs/sq-phase2-global-20261011/collected/`.
- First cloud interval through runner finish:1,187.4s, approximate0.660
  visible-device-hours, excluding final export and platform teardown.
- Phase3 code: `71b2e2b0f94bd7790a9389a0a0963839669494ab`.
- Phase3 remote: `oussemaharrabi/sq-phase3-controls-20261011`, version1.
- Phase3 frozen spec/submission: `artifacts/runs/sq-phase3-controls-20261011/`.
- Phase3 metrics SHA256: `adc99ea7e4a3ca57dee3b5ce62744c35165a04a7d67b414e361c013635f42002`.
- Raw numerical report, original selection and extracted collection attestation:
  `artifacts/sample-results/phase3-controlled-followup/`.
- Validation selectedmseclip at13:05:54UTC before the runner loaded C4 validation.
  Heldout49,951 predicted tokens: uniform4 PPL28.0056, mseclip27.4656 (-1.93%).
  Both target payload59,719,680B, full checkpoint116,522,698B.
- Six prior allocations: calibration joint gains0.0525–0.1315NLL versus predicted
  gains0.1953–0.3133; disjoint training loss worsens0.0473–0.0987NLL.
  The18bounded allocations were evaluated on development only.
- Phase3 scientific218.6s, outer interval through runner finish289.184s,
  approximate visible-device-hours0.160658; export/platform overhead excluded.
- Draft PR: https://github.com/OussemaHarrabi/SpectraQuant/pull/1 . Do not merge.

## 6. Assumptions

User directly confirmed20 remaining Kaggle GPU-hours on11 October. Free only.
First cap6.67 plus second cap2.67 leaves10.66h even when the full first cap is
reserved; >=3h reserve remains. Exact account quota billing is not available.
One model and three calibration subsets are exploratory, not training replicates.

## 7. Risks / unresolved questions

No duplicate submission or further GPU job. The new holdout is C4 validation, first used for this
project after development selection freezes. Pretraining exposure is unknown.
Old WikiText test is retired from decisions. Quality is class2 dense FP16
reconstruction, storage class3 actual files; no GPU low-bit acceleration claim.
Full original Tier2 reproduction and cross-model generalization remain open.
Clipping is conventional and deterministic. Do not label its small positive
result an original SpectraQuant allocator or replicated trainable-method gain.

These scripts submit no additional GPU job during collection. Check current
chat and registry state before any further cloud action.

## 8. Recommended next action

The controlled follow-up is complete. Resume collection only if verifying the
existing bundle again; never push its id again:

```bash
uv run python scripts/experiments/phase2_resume.py --study phase3 --fetch
```

The research report, README, ledger, claim ledger and career record are updated
with the bounded result. Keep the draft PR unmerged. Any future cross-model or
stronger-baseline study needs a new frozen protocol, new holdout and separate
compute authorization. Do not tune on the exposed C4 holdout or add GPU work
through the result-collection follow-up.
