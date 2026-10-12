# A-0024: joint-loss diagnosis and bounded allocation follow-up

11 October 2026. Frozen before the new cloud submission; exploratory scope.

Phase 2 completed on immutable `fe39413`. Both allocation families failed the
WikiText validation rule, which retained uniform4. Test degradation is not a
reason to retune on that test. Retire WikiText test from all further decisions.

Questions:
1. Did additive marginal calibration gains survive joint installation on the
   **same** calibration blocks? Compare measured joint delta with the summed
   marginal prediction, then measure a disjoint 4096-token training diagnostic.
2. Can a trust region of at most 8, 16 or 32 changed matrices reduce interactions?
   Reuse the checksum-anchored Phase 2 scores, but reproduce every old selected
   candidate byte-for-byte before accepting score reuse. Same byte ceiling,
   same bits/ranks; three calibration subsets, not independent training seeds.
3. Does a simpler weight-MSE clipping control help at exactly the same INT4
   packed payload? Choose per-group alpha from {1,.95,.9,.85,.8,.75} by error of
   the actual serialized/reloaded weight rounded to inference dtype. Include
   alpha=1 so error cannot exceed RTN on this criterion. No novelty claim.

WikiText validation selects one family/cap by mean NLL across the three
calibration subsets; threshold >=0.005 below uniform4. Include the one
deterministic clipping control in selection without treating it as three
replicates. Freeze selection to disk **before** first reading C4 validation.
Run the selected family (three allocations if selected), uniform4 and the
predeclared clipping comparator on the first 50,000 C4 validation tokens.
That split is new to this project's method development, not necessarily to
pretraining. If no family passes validation, publish the negative result and
retain uniform4. No method or alpha changes after seeing this held-out result.

Full checkpoints saved only for these final arms. Target payload byte ceiling
and full checkpoint bytes reported separately. Quality remains dense FP16
reconstruction (class 2), storage real files (class 3); no kernel speed claim.

Hard limits: 3600s scientific runner, 4800s platform (80min), free private T4x2,
<=2.67 visible-device-hours worst case plus the previous <=6.67 reserved cap
and >=3h reserve, within the user-confirmed 20h. First run actually used an
approximate 0.660 visible-device-hours through runner finish. This is not exact
Kaggle billing. Early throughput/disk gates and finally-saved failure evidence
remain mandatory. No gradient training, no full Tier 2 or SOTA claim.

Dependencies are read-only previous-kernel artifacts, pinned by report and
checkpoint SHA256, plus the existing immutable model/C4/WikiText revisions.
Preserve all Phase 2 evidence and failures. Do not resubmit an existing run id.
