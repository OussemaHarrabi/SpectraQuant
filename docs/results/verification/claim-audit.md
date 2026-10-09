# Independent claim-to-evidence audit (ClaimAudit L)

Auditor: `ClaimAudit2` (independent verifier slice). Date: 2026-10-09.
Scope: headline claims in `README.md`, `docs/coordination/status.md` (§3, §5) and each
`docs/results/*.md` report. Each claim is quoted, its cited artifact named, and the number
re-derived by reading the artifact (never the prose). Read-only everywhere except
`docs/results/verification/**`.

**Sibling work excluded.** `docs/results/ablation-report.md`, `reports/**` and `CAREER_EVIDENCE.md`
are being written concurrently by the `Ablations` and `ReleasePackage` slices; per instruction they
are **not** treated as final evidence and not audited here.

**Concurrent edit to `README.md` (noted mid-audit).** At session start the working tree was clean
and `README.md` equalled committed `HEAD` (`2f6d7a9`). During the audit the `ReleasePackage` slice
began rewriting `README.md` (now `M` in `git status`, uncommitted). Findings **L1/L2/B1/B2 below are
against the committed `HEAD` version** — inspect `git show HEAD:README.md` to confirm — and the
in-flight working copy already fixes them (new blurb: "the compression machinery … is implemented
and has produced Tier-0 fixture evidence"; 4-CPU row now "**yes** — available and measured"). The
uncommitted rewrite is *not* audited here as final; the two README issues should be re-checked on
landing.

Method: recomputed numbers in an independent Python process loading the committed JSON artifacts;
re-ran four artifact-producing scripts from a clean clone of `origin/infra/bootstrap` (HEAD
`2f6d7a9`). All 8+ required re-derivations pass; **no headline number contradicts its artifact.**
The findings are documentation-provenance issues, the most serious being a stale `README.md` whose
front-door status contradicts the delivered evidence.

---

## A. Claim ledger

Verdict key: **OK** number matches artifact · **STALE-DOC** prose outdated vs artifact/repo ·
**SPIN** wording overstates/attributes · **NOT-RUN** correctly marked not run.

| # | Claim (quoted, abbreviated) | Source | Cited artifact | Re-derived | Verdict |
|---|---|---|---|---|---|
| L1 | "no compression method, proxy, allocator or Tier-2 model exists in this repository" | README status blurb (**HEAD**) | (whole repo) | quant/proxy/allocator/regularizer all implemented + artifacts committed | **STALE-DOC** (HEAD; in-flight sibling edit fixes it) |
| L2 | "nothing in this scaffold produces it yet, so no class 4 number is claimed anywhere in this repository" (class 4-CPU) | README (**HEAD**) | `class4cpu/class4cpu.json` exists | class4cpu artifact committed, class 4-CPU | **STALE-DOC** (HEAD; in-flight sibling edit fixes it) |
| L3 | naive proxy "ρ = −0.060 under LayerNorm vs gain-aware ρ = **+0.830**" | status §3 #11 | `proxy-fixture.json` | recomputed −0.06043956 / +0.82967033 | OK |
| L4 | "joint/Σproxy 0.069 (naive) vs 0.85 (gain-aware); joint/Σsingle = 1.375" | status §3 #11 | `proxy-fixture.json` | 0.069044 / 0.852635 / 1.375108 | OK |
| L5 | candidate "ρ = +0.830 against +0.412 / +0.357 / +0.121 / −0.044" | status §3 #16 | `proxy-fixture.json` | 0.829670 / 0.412088 / 0.357143 / 0.120879 / −0.043956 | OK |
| L6 | "CP-SAT equals the exhaustive oracle exactly at three budgets (Δ < 1e-9)" | status §3 #12 | `allocator-report.md` table | table self-consistent; no JSON artifact | OK (prose-only) |
| L7 | "uniform +7.5–62.8 % and greedy +20.6–94.1 % worse than the oracle" | status §3 #12 | `allocator-report.md` | +7.53/62.76/49.50 % and +94.13/44.45/20.62 % | OK (prose-only) |
| L8 | "15/15 aggregate contrasts" (candidate beats comparators) | status §3 #17 | `proxy-validation.json` | 5+5+5 = 15 "supported", 0 otherwise | OK |
| L9 | "10 trained models, 12 cells, 5 seeds"; "achieved MDE(z) 0.476–0.715 … inside 0.33–0.79" | status §3 #17 | `proxy-validation.json` | group MDE 0.636/0.715/0.476; candidate-contrast MDE 1.453/1.299/0.968 (see B4) | **SPIN** |
| L10 | int4 1662 B, int8 2727 B, "int4 relative error 0.0771" | status §3 #18 | `class4cpu.json` | sum(files)=1662 / 2727; rel err 0.077129 | OK |
| L11 | "18 arms, byte parity 18/18 within 0.5 % (0.0 % relative), 9 equal-memory pairs gated with 3 refused" | status §3 #20 | `frontier.json` | n_reconciliations=18, max_rel=0.0; 12 gated, 3 refused | OK |
| L12 | "mixed allocation cuts hidden-state damage 2.73x versus uniform at identical measured bytes (10.48 vs 28.57 at 16 928 B)" | status §3 #20 | `frontier.json` | 28.565754/10.477057 = 2.7265x | OK |
| L13 | "predicted-vs-measured rank agreement +0.971"; "4 equal-memory wins" | status §3 #20 | `frontier.json` | 0.9708333; 4 wins | OK |
| L14 | "λround drives the rounding residual down −90.6 % and improves dense-fidelity −36 %, but dev NLL worsens (+638 % full arm)" | status §3 #21 | `sweep.json` | −90.61 % (product residual) OK; **−36 % is the `full` arm vs `none`, not a λround effect** (λround output-err = +8.17 %) | **SPIN** |
| L15 | "the untuned `combined` variant is worse than two comparators" | status §3 #17 | `proxy-validation.json` | combined refuted vs weight_frobenius & weight_magnitude (layernorm + pooled) | OK |
| L16 | "cloud Tier-1 cells NOT RUN" (H2/H4/H3) | status §3 #17/#20/#21, §5 | artifacts' `cloud_cells`/`confirmatory_cell` | all three `"status":"NOT RUN"` | NOT-RUN ✓ |
| L17 | frozen verdicts "A: differentiated (narrow, empirical), B: differentiated (weak–moderate), C: previously-known" | status §5 M1 | `m1-novelty-review.md` | review's revised basis matches; no stronger claim | OK |
| L18 | "13 compressible linear modules per model" vs "9 compressible linear layers per model" | `proxy-fixture-report.md` §1/§4 | `proxy-fixture.json` | artifact has 13 layers; ρ is over 13 | **STALE-DOC** (internal) |
| L19 | "10 trained models and 12 reported cells" | `proxy-validation-report.md` | `proxy-validation.json` | `per_cell` has 12 entries | OK |
| L20 | "every cloud cell is marked NOT RUN"; scope line verbatim; backend/op/container/threads/CPU named | `class4cpu-report.md`, AGENTS §5 | `class4cpu.json` | scope line present verbatim; `threads.cpu_model`, `kernels`, `measurement_class:"4-CPU"` all present | OK |

---

## B. Re-derived numbers (command + output)

All recomputations were run in an independent process loading the committed JSON (no repo module
imported for the arithmetic).

**R1 — proxy-fixture Spearman ρ, LayerNorm (L3/L5).** Rank-correlate `proxies.*.per_layer` against
`damage.per_layer` over all 13 modules:

```
gain_aware_composed        recomputed rho=0.829670  artifact=0.829670
per_layer_output_error     recomputed rho=-0.060440  artifact=-0.060440
weight_frobenius           recomputed rho=0.412088  artifact=0.412088
weight_magnitude           recomputed rho=0.357143  artifact=0.357143
activation_magnitude       recomputed rho=0.120879  artifact=0.120879
hessian_diag               recomputed rho=-0.043956  artifact=-0.043956
in_situ_output_error       recomputed rho=0.472527  artifact=0.472527
noLN gain_aware rho: 0.7307692307692307 artifact=0.7307692307692307
```

**R2 — joint/super-additivity (L4).** `joint_hidden / sum_single_layer`:

```
layernorm    recomputed 1.3751080626424674  artifact 1.3751080626424674
no_layernorm recomputed 1.0828346460363278  artifact 1.0828346460363278
```

**R3 — class-4-CPU container bytes (L10).** Sum of every produced file per artifact:

```
class4cpu fp32: sum(file bytes)=8321  total_bytes=8321
class4cpu int4: sum(file bytes)=1662  total_bytes=1662   (graph 638 + sidecar 1024)
class4cpu int8: sum(file bytes)=2727  total_bytes=2727
int4 payload recompute K*N/2 = 1024 (stored 1024); scales 256B/4 = 64 = K/group_size*N
```

**R4 — class-4-CPU relative error (L10).** `max_abs_err / ref_max_abs`:

```
[4,64]  int4 max_abs=0.204059 ref=2.6457 rel=0.077129 (reported 0.077129)
[4,64]  int8 max_abs=0.020771 ref=2.6457 rel=0.007851 (reported 0.007851)
[512,64] int4 max_abs=0.277936 ref=2.9404 rel=0.094523 (reported 0.094523)
[512,64] int8 max_abs=0.037231 ref=2.9404 rel=0.012662 (reported 0.012662)
```

**R5 — frontier ratio at 16 928 B (L12).** `reference/candidate` hidden damage:

```
ratio=0.366770 inv=2.7265x cand_dmg=10.477057 ref_dmg=28.565754
budget=9496  inv=1.6273x ; greedy 16928 inv=1.0239x
```

**R6 — frontier byte reconciliation & rank agreement (L11/L13).**

```
n_reconciliations=18  max_relative_difference=0.0  all_within_tolerance=True
rank_agreement: across_all_arms_spearman=0.9708333  per_arm_layer_spearman_mean=0.865690
                per_solver: greedy 1.0, ortools 1.0, uniform 0.942857
```

**R7 — proxy-validation aggregate contrast count (L8).** Verdict tally per group/candidate:

```
layernorm    gain_aware_composed {'supported': 5}
no_layernorm gain_aware_composed {'supported': 5}
all_fixtures gain_aware_composed {'supported': 5}   -> 15/15 supported
layernorm    combined {'inconclusive': 3, 'refuted': 2}
all_fixtures combined {'refuted': 5}
```

**R8 — regularizer sweeps (L14) + arm contrast.**

```
lambda_round vs product_rounding_residual: relative_change = -0.906125  (-90.61 %)  non_increasing=True
lambda_round vs factor_rounding_residual : relative_change = -0.939845  (-93.98 %)
lambda_round vs output_error_vs_dense    : relative_change = +0.081695  (+8.17 %)  <-- degrades
lambda_spectrum vs measured_tail_energy  : relative_change = -0.605691  (-60.57 %)
lambda_residual vs factor_rounding_resid : relative_change = +0.068284  (+6.83 %)  (STE degenerate)

none -> full: dev NLL +638.3516 % ; output err vs dense -35.7278 % ;
              factor residual -45.4417 % ; product residual -35.2246 %
```

**R9 — allocator percentages (L6/L7).** From `allocator-report.md`'s own objective column:

```
budget 4000: uniform +7.53 %  greedy +94.13 %   (oracle 1.6600)
budget 8000: uniform +62.76 % greedy +44.45 %   (oracle 0.5975)
budget 16000: uniform +49.50 % greedy +20.62 %  (oracle 0.3788)
```

This matches the status range symbols `+7.5–62.8 %` / `+20.6–94.1 %`, but note the underlying
objective values live **only in the report prose** (see B7).

---

## C. Labelling discipline

- **Class 4-CPU claim (AGENTS §5):** `class4cpu.json` carries `measurement_class:"4-CPU"`, names the
  backend (`onnxruntime`), the ops (`MatMulNBits`/`MatMulInteger`), the container (SpectraQuant ONNX
  int4/int8), threads (`threads.effective=8`, `intra_op=8`/`inter_op=1`), the CPU model
  (`AMD Ryzen AI 7 350`), a same-session fp32 baseline, and `dequantized_path=false`. The mandatory
  scope line is present **verbatim**: `"not comparable to published GPU latency or throughput
  figures"` (substring check returns `True`). **OK.**
- **No class mixing:** each report prints a class table. `allocation-integration-report.md` §1
  separates class-1 accounted bytes, class-3 measured bytes, class-2 predicted/measured damage;
  `regularizer-report.md` §6 lists class 1/2 only and states none of 3/4/5 is measured.
  `frontier.json` keeps `accounted_bytes` (class 1) and `measured_bytes` (class 3) as distinct fields.
  **OK.**
- **Cloud cells marked NOT RUN:** `proxy-validation.json` → both `cloud_cells` are `"NOT RUN"`;
  `sweep.json` → `confirmatory_cell.status = "NOT RUN"`; class4cpu/frontier reports state the GPU
  halves are deferred. **OK.**
- **Fake quantization never storage/speed:** the frontier's class-2 hidden-damage win is labelled
  class 2 in §1 and the report explicitly states the class-3 number is a byte count, "never a quality
  claim". The class-4-CPU report states "No speedup is claimed or implied" and its latency table is
  labelled "protocol compliance, not a performance claim". **OK.**
- **Analytical not presented as measured:** the frontier's classification labels the equal-memory
  gate as on **measured class-3** bytes; `regularizer-report.md` §5 states its equal-memory check is on
  class-1 analytical bytes, "not class-3 measured storage". **OK.**
- **README measurement table:** row 4-CPU states "no such path exists in this scaffold yet" — false;
  see L2/B2.

---

## D. Spin check

| Question | Finding |
|---|---|
| Is the M5 mixed result presented as an improvement? | **No.** `regularizer-report.md` §7 and `status.md` §5 report "H3 not supported", "mixed and reported as such". `sweep.json` `falsifier_h3` published verbatim. Only `status.md` §3 #21 mis-attributes the −36 % fidelity figure to λround (B3). |
| Is the naive proxy's failure softened? | **No.** `proxy-fixture-report.md` "deliberately publishes the failure"; ρ = −0.060 published in the headline and repeated in `status.md`. |
| Is the allocator's hidden-state-vs-logits caveat still visible wherever the win is quoted? | **Yes.** `allocation-integration-report.md` §3 ("The logits axis is a published caveat, not a win") and `status.md` #20 ("logits damage worsens at mixed points"). `frontier.json` keeps both axes. |
| Is the achieved MDE presented as the predeclared one? | **Partly ambiguous.** The report separates `predeclared_mde` (0.33–0.79) from `mde_achieved_z`. But `status.md` #17 and the report §0 quote the **group-level** achieved MDE (0.476–0.715) as "inside the band", while the candidate's own contrast achieved MDE is **0.968/1.299/1.453 — above the conservative end**. The tables disclose both, but the one-line summary invites reading the candidate as inside the band. See B4. |
| Any novelty claim stronger than the frozen verdicts (A narrow/empirical, B weak–moderate, C previously-known)? | **No.** `status.md` §5 M1 repeats exactly those verdicts; README makes no SOTA claim. The adversarial review's own table (A "overclaimed" → narrowed, B "overclaimed (mild)" → keep, drop "moderate") is consistent with the frozen wording. |

---

## E. Reproducibility from a clean checkout

Clone of the pushed branch:

```
$ git clone --branch infra/bootstrap --depth 1 https://github.com/OussemaHarrabi/SpectraQuant.git sq-verify-tmp
$ git rev-parse HEAD   -> 2f6d7a9be6cdcc64188dcc1c0948ef1aca5bbd63   (== origin/infra/bootstrap)
$ uv sync --all-extras -> resolved; torch==2.14.1+cpu, onnxruntime, ortools, transformers 4.57.6 ...
```

Four artifact-producing scripts were re-run (≥2 required). Comparison is against the committed
artifact in the clone.

**E1 `scripts/experiments/proxy_fixture_measurement.py`** — 18 s.

```
$ git diff --stat artifacts/sample-results/proxy-fixture/proxy-fixture.json
 1 file changed, 3 insertions(+), 3 deletions(-)
diff lines: only "wall_time_s" (x2) and "git_commit" changed.
```

Recomputed ρ path prints `per_layer_output_error rho=+0.242 … weight_frobenius rho=+0.429 …`
(no-LayerNorm fixture) matching the committed artifact. **Drift = timing + commit only.**

**E2 `scripts/experiments/class4cpu_measurement.py`** — 4 s.

```
[int4] total=1662 payload=1024 scales=256 overhead=382
[int8] total=2727 payload=2048 scales=4 overhead=674
[[4,64] int4] max_abs_err=0.204059 rel=0.0771292
[[512,64] int4] max_abs_err=0.277936 rel=0.0945227
```

`git diff` shows only latency blocks (`median_ns`, `p95_ns`, …), `timestamp_utc` and `git_commit`.
**Bytes and error figures reproduce exactly; latency drifts as documented ("indicative").**

**E3 `scripts/experiments/allocation_frontier.py`** — 17 s.

```
rank agreement across arms: +0.9708
equal-memory wins (mixed beats uniform): 4
  budget 16928 B: hidden damage 1.0477e+01 vs uniform 2.8566e+01 (ratio 0.367)
git diff --stat: 19 files, +58 -58
  frontier.json: only fixture.training.wall_time_s + git commit/describe/short_commit
  18 manifests : only git commit/describe/short_commit
```

Matches the report's §8 claim (identical apart from wall time; here also the commit SHA because the
tree advanced from `5928721-dirty` to `2f6d7a9-dirty`). **Scientific content identical.**

**E4 `scripts/experiments/proxy_validation_sweep.py`** — 36 s.

```
$ git diff artifacts/sample-results/proxy-validation/proxy-validation.json | grep '^[-+]' | grep -v commit|timestamp|wall_time
   (empty)
[all_fixtures] gain_aware_composed vs weight_frobenius: SUPPORTED ... delta_z = +0.604
[all_fixtures] gain_aware_composed vs per_layer_output_error: SUPPORTED ...
```

**Only provenance/timing drift; every contrast (including 15/15 SUPPORTED) reproduces byte-for-byte.**

Conclusion: the four artifacts are reproducible; no scientific drift. Committed provenance SHAs
(`856f288`, `3fa0c1cb`, `5928721`, `1207b69`, `b2ea1ede`) all exist and are ancestors of HEAD.

**E5 — write-scope proof.** The tree **moved during the audit**: the clone and script re-runs in E1–E4
were taken at `origin/infra/bootstrap` = `2f6d7a9` (the pushed state at audit start); the
`Ablations` slice then committed `f354a17` locally, so the current HEAD is `f354a17`. `git status
--short` in the main repo at the end of the audit:

```
 M README.md
?? CAREER_EVIDENCE.md
?? docs/cards/
?? docs/results/verification/claim-audit.md
?? reports/
?? scripts/reproduce/generate_release_artifacts.py
?? tests/unit/test_release_artifacts.py
```

The auditor's **only** write is `docs/results/verification/claim-audit.md`. Every other entry is a
sibling slice's concurrent work: `M README.md` and the new `CAREER_EVIDENCE.md`, `reports/**`,
`docs/cards/**`, `scripts/reproduce/generate_release_artifacts.py`,
`tests/unit/test_release_artifacts.py` belong to the `ReleasePackage`/`Ablations` streams.
`git diff --name-only` lists only `README.md`, and it is the sibling's edit, not the auditor's
(the auditor edited no tracked file). The temp clone `C:/Users/oussa/sq-verify-tmp` was outside the
repository and has been removed.

---

## F. Freeze discipline

Freeze commit = `6281c56` (preregistration FROZEN, amendment A-0006).

- `docs/research/preregistration.md` was changed once after the freeze — in `fe7681e` — and that
  change (candidate proxy declared + §13 class-4-CPU item flipped to satisfied) **is recorded** as
  amendment **A-0007** in the same commit. `preregistration-amendments.md` contains A-0007 and A-0008
  (regularizer surrogate), both append-only. **Clean.**
- Frozen plan configs (`configs/tier1`, `configs/tier2`, `configs/repro`) and
  `src/spectraquant/experiment_plan.py` were **not** touched after the freeze. **Clean.**
- **One silent post-freeze change:** `artifacts/schemas/run-manifest.schema.json` was modified in
  `c36f15f` (adds `"svd"` to the `method` enum) — a frozen M1 checklist item (`run_manifest.json`
  schema) changed with **no amendment**. The commit message documents it, but the amendments file does
  not. **See B5.**

---

## G. Blocking issues

**B1 — `README.md` (committed `HEAD`) front-door status is false; a sibling is concurrently fixing
it.** The HEAD blurb "Status: pre-alpha research scaffold … The science has not started yet: **no
compression method, proxy, allocator or Tier-2 model exists in this repository**" and the "What is
*not* done yet" rows for Quantization/Proxies/Allocation all contradict the committed evidence
(`proxy-fixture.json`, `proxy-validation.json`, `frontier.json`, `sweep.json`, `class4cpu.json`) and
`status.md` §3 #11–#21. **However**, the `ReleasePackage` slice's uncommitted rewrite (visible in the
working tree during this audit) replaces the blurb and rebuilds the status table with a "Measured
(local)" table citing the artifacts. *Minimal fix:* land that edit (verify it as the authoritative
README). Temporarily a reader of the pushed branch is misled.

**B2 — `README.md` (committed `HEAD`) measurement-classes row 4-CPU is false.** "…**nothing in this
scaffold produces it yet, so no class 4 number is claimed anywhere in this repository**" — but
`artifacts/sample-results/class4cpu/class4cpu.json` is a committed class-4-CPU measurement.
*Minimal fix:* the in-flight sibling edit changes the cell to "**yes** — available and measured
(`artifacts/sample-results/class4cpu/`)". Verify it lands.

**B3 — `status.md` §3 #21 mis-attributes a number to `lambda_round`.** "lambda_round drives the
rounding residual down −90.6 % **and improves dense-fidelity −36 %**, but dev NLL worsens (+638 %)".
The −36 % is the `full` arm vs `none` contrast; the λround direction check on
`output_error_vs_dense` is **+8.17 %** (it degrades). `sweep.json` confirms
`lambda_round_vs_output_error.relative_change = +0.081695`.
*Minimal fix:* rephrase to "the `full` arm improves dense-fidelity −36 % (vs `none`) and the λround
sweep drives the product rounding residual −90.6 %, while dev NLL worsens (+638 % for the full arm)".

**B4 — the "achieved MDE inside the predeclared band" summary is ambiguous.** `status.md` #17 and
`proxy-validation-report.md` §0 quote the achieved MDE as "0.476–0.715 inside the predeclared
0.33–0.79 band"; that is the **group-level mean-n_eff** MDE. The candidate's own contrast achieved
MDE is **0.968 (pooled) / 1.299 / 1.453**, i.e. **above** the conservative predeclared end. The
per-contrast tables disclose this, but the one-line summary does not.
*Minimal fix:* add the candidate-contrast MDE (0.97–1.45) alongside the group-level figure, or state
explicitly that 0.476–0.715 is the group mean-n_eff MDE.

**B5 — silent post-freeze schema change (freeze violation).** `artifacts/schemas/run-manifest.schema.json`
was changed in `c36f15f` (added `"svd"` to the method enum) with no entry in
`docs/research/preregistration-amendments.md`, although the schema was an M1 freeze checklist item.
*Minimal fix:* append an amendment A-0009 recording the schema extension (or state in the amendment
file that the schema is not a frozen item).

**B6 — `proxy-fixture-report.md` internal layer-count contradiction.** §1 says "13 compressible linear
modules per model" then "**9** compressible linear layers per model"; §4 repeats "9 layers". The
artifact and every ρ are computed over **13** layers (`damage.per_layer` has 13 keys).
*Minimal fix:* replace both "9" with "13".

**B7 — allocator-report numbers have no machine-readable artifact.** The Milestone-2 allocator
figures (oracle objectives 1.6600/0.5975/0.3788, uniform/greedy percentages, solver times) exist only
as prose in `allocator-report.md`; unlike every other slice, no JSON artifact backs them, so a
re-auditor cannot diff them without re-running probes. (The `+7.5–62.8 %` / `+20.6–94.1 %` symbols
themselves are arithmetically correct.) *Minimal fix:* commit the probe output as
`artifacts/sample-results/allocator/oracle-comparison.json` and cite it.

**Non-blocking observations:** `status.md` §2 says "Tiers 0–1 executable locally" whereas AGENTS
§2.3/§6 place Tier 1 on the cloud (stale phrasing). `proxy-validation-report.md` uses the word
"refuted" in the all-fixtures `combined` rows that its own §0 prose describes as "below the
predeclared MDE" (terminology drift: the JSON labels 3 equivalence-refutations as `refuted`); both
are consistent with frozen §11.1, which permits refutation by a CI narrower than the MDE.

---

## H. Report contract (AGENTS.md §10)

1. **Scope completed.** Independent claim-to-evidence audit of README, `status.md` §3/§5 and the
   seven `docs/results/*.md` reports (excluding sibling in-flight files); 20 ledger rows; 9
   independent re-derivations; 4 scripts re-run from a clean clone; freeze audit.
2. **Files changed.** `docs/results/verification/claim-audit.md` (this file) — only.
3. **Commands executed.** Listed in §B (Python recomputations) and §E (clone + `uv sync --all-extras`
   + 4 scripts + `git diff`); freeze checks via `git log`/`git show`/`git cat-file`.
4. **Tests and results.** No repo test suite run (out of scope; siblings mid-flight). Scoped proof:
   the four artifact scripts re-run and diffed — scientific content identical, only timing/provenance
   drift.
5. **Evidence created.** `docs/results/verification/claim-audit.md`.
6. **Assumptions.** Committed artifacts are the authoritative evidence (per report preambles);
   runtime-latency fields are permitted to drift (reports state this); `docs/results/ablation-report.md`,
   `reports/**`, `CAREER_EVIDENCE.md` are sibling work and were not judged.
7. **Risks / unresolved questions.** README staleness (B1/B2) is the highest-impact issue because it
   is the repository front door, but the `ReleasePackage` slice's uncommitted rewrite already fixes
   it — re-verify on landing. B5 is a genuine freeze-discipline gap (silent post-freeze schema edit).
   No headline scientific number was found to be wrong. The MDE framing (B4) should be tightened
   before any external reader sees it.
8. **Recommended next action.** Land/verify the sibling README rewrite (B1/B2); fix B3/B4 in
   `status.md`; record amendment A-0009 for the schema change (B5); tidy B6/B7.
