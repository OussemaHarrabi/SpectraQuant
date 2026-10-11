# SpectraQuant takeover: validated first result, queued follow-up

## 1. Scope completed

Phase2 completed on Kaggle and was collected independently. Both proposed
allocation families failed validation; the baseline was retained. Phase3 was
implemented, tested, independently reviewed and submitted once. Its results
remain pending: the last observed state is QUEUED for T4x2, version1.

Original checkout `C:/Users/oussa/oussema/SpectraQuant`, at a603a4c, is preserved.
Its uncommitted CLI, cloud/dryrun and cloud-preflight-test edits were untouched.
Continuation checkout: `C:/Users/oussa/oussema/PFE/work/spectraquant-phase2`,
branch `research/phase2-global-damage`. Never overwrite the original edits.

## 2. Files changed

Scientific Phase2/Phase3 runners, frozen JSON configurations and amendments,
generated Kaggle notebooks, submission/resume tooling, manifest provenance,
checkpoint verification, path-safe wrapped collection, CLI commands and tests.
Research report, README, results tables, allocation-instability analysis and
compute ledger reflect the completed negative result. Career record revision41
records verified evidence and keeps Phase3 pending.

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
- Draft PR: https://github.com/OussemaHarrabi/SpectraQuant/pull/1 . Do not merge.

## 6. Assumptions

User directly confirmed20 remaining Kaggle GPU-hours on11 October. Free only.
First cap6.67 plus second cap2.67 leaves10.66h even when the full first cap is
reserved; >=3h reserve remains. Exact account quota billing is not available.
One model and three calibration subsets are exploratory, not training replicates.

## 7. Risks / unresolved questions

Kaggle scheduling is the current delay; the accepted follow-up remains queued.
No duplicate submission. The new holdout is C4 validation, first used for this
project after development selection freezes. Pretraining exposure is unknown.
Old WikiText test is retired from decisions. Quality is class2 dense FP16
reconstruction, storage class3 actual files; no GPU low-bit acceleration claim.
Full original Tier2 reproduction and cross-model generalization remain open.

A persistent follow-up was requested from the app but automatic approval review
rejected it because explicit authorization for a recurring workflow was absent.
No automation was created. A specific user-approval question is pending. Do not
treat the original experiment approval as approval for persistent automation.

## 8. Recommended next action

Resume the existing job; never push its id again. Example PowerShell setup:

```powershell
$env:PYTHONPATH='C:/Users/oussa/oussema/PFE/work/spectraquant-phase2/src'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:GIT_CONFIG_COUNT='1'
$env:GIT_CONFIG_KEY_0='safe.directory'
$env:GIT_CONFIG_VALUE_0='C:/Users/oussa/oussema/PFE/work/spectraquant-phase2'
& 'C:/Users/oussa/oussema/SpectraQuant/.venv/Scripts/python.exe' scripts/experiments/phase2_resume.py --study phase3 --fetch --kaggle-cli 'C:/Users/oussa/oussema/SpectraQuant/.venv/Scripts/kaggle.exe'
```

Only finished/failed remote runs are fetched. UTF8 avoids the Windows log-download
encoding failure. Verify all scientific checksums and checkpoints before registry
validation. On success, report joint calibration gaps, development selection and
fresh heldout outcome, including any negative results. Update research report,
README, ledger, career record and PR description; commit/push accepted evidence.
On failure, preserve its artifacts and identify the cause before considering any
repair. Never use failure or heldout values to justify undisclosed method tuning.
