# SpectraQuant — Risk Register

Live register. Each risk has an id, category, description, likelihood, impact, mitigation, and an
owner (stream letter from `docs/coordination/ownership.md`; "orch" = orchestrator). Likelihood and
impact are `L`/`M`/`H`. Status: `open`, `mitigating`, `closed`, `accepted`.

Owners are *roles*, not people; the current assignee is whoever holds the stream. Risks are reviewed
whenever a gate in `charter.md` §8 is closed, and any change in a HIGH risk is recorded as an
amendment when it affects the protocol.

---

## 1. Technical risks

| ID | Risk | L | I | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| T1 | **No local GPU.** The workstation has no CUDA-capable device, so no training/QAT/GPU evaluation may run locally; Tiers 1–5 depend on a cloud substrate whose sessions can fail, be pre-empted, or be unavailable. | H | H | Local scope restricted to fixtures/analysis/class 4-CPU (`AGENTS.md` §2b); Tier 1–5 executed on the cloud notebook substrate with the manifest + checksum rules; nothing reported as a result until a validated `run_manifest.json` + checksum-validated artifacts exist; never present planned cells as completed. | orch + B | mitigating |
| T2 | **Fake-vs-real quantization confusion.** Float execution of quantization numerics (Class 2) is mistaken for low-bit storage (Class 3) or kernel-backed speed (Class 4-CPU/4-GPU/5), producing inflated claims. | M | H | Every number carries exactly one measurement-class label (`AGENTS.md` §5); conversion of fake quantizers to packed storage is a separate, tested code path; storage claims are backed by actually serialized bytes; **class 4-CPU** claims must name backend/op/container/threads/CPU and compare against a same-session fp32 CPU baseline; class 4-GPU and class 5 are always "not measured". | D (measurement taxonomy) + A | mitigating |
| T3 | **Upstream CUDA-only code** (bitsandbytes, TorchAO CUDA kernels, GPTQ/AWQ GPU kernels, vLLM) is imported and silently fails or falls back to different numerics on CPU. | M | H | Pin only MIT/Apache/BSD-3-Clause-Clear/Apache-2.0 repos (see `upstream-lockfile.md`); CUDA-requiring paths MUST raise `NotImplementedError`/config error, never a silent fallback; upstream imports are read-only references in the core path; CPU fake-quant semantics reimplemented in-house for portability. | B (scaffold) + D | mitigating |
| T4 | **Byte-accounting omissions.** Analytical Class-1 estimates ignore scales, zero-points, low-rank factors, codebooks, or padding, so "compression ratio" is optimistic. | M | H | Class-3 measured stored bytes are authoritative; Class-1 vs Class-3 discrepancy is a tracked secondary outcome (S5) with a predeclared tolerance; packing routines unit-tested (round-trip, group-size edge cases) per `AGENTS.md` §7. | D + B | open |
| T5 | **Allocator exploits accounting or proxy artifacts.** A greedy/knapsack allocator finds a degenerate assignment (e.g., collapse to 2-bit on layers the proxy underestimates) that wins the metric but is unusable. | M | H | Predeclared guardrails (minimum bit width per layer, per-arm sanity checks); report the allocation map, not just the frontier; equal-memory controls; hold out a dev set for allocation sanity; disclose failure modes of the proxy. | H + F | open |
| T6 | **Proxy cost makes it useless.** The output-aware proxy's probe/calibration cost exceeds its benefit, so it is not a practical allocator input. | M | M | Track proxy wall-clock and calibration size (S6); predeclare a practicality budget; compare against zero-calibration comparators (e.g. random-probe) honestly. | F | open |
| T7 | **Numerical instability at 2–3 bit / tiny ranks.** Divergence, or quantizer behavior dominated by edge cases (zero-range, constant groups, non-divisible group sizes). | M | M | Predeclared edge-case tests (`AGENTS.md` §7); automated exclusion rules (§9 preregistration) for non-finite values; report instability rather than hide it. | D | open |
| T8 | **CPU determinism breaks.** Non-deterministic CPU kernels or threading make bit-for-bit reproduction impossible, breaking the seed plan. | M | M | Pin thread counts; set all seeds; assert determinism in tests; record dirty-state and SHA in manifests. | B + J | open |
| T9 | **Tiny-model results do not transfer.** Tier-1 findings are claimed to hold at 1B+ without evidence. | H | H | Explicitly scope claims to the tiers actually measured (amendment A-0001); no scale-transfer claims; Tier-2+ listed as future work until run. | A + M | mitigating |

## 1b. Cloud-substrate risks (added 2026-10-08, `AGENTS.md` §2b / amendment A-0004)

| ID | Risk | L | I | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| C1 | **Free-tier session pre-emption / quota exhaustion** truncates the run matrix: Tier-1/2 sweeps (arms × seeds × budgets) do not complete within session or daily quota limits. | H | H | Size each run to fit a session; run the matrix in idempotent, resumable slices keyed by the persisted remote run id; prefer Kaggle (unattended, CLI) for long sweeps; checkpoint and export partial artifacts; a pre-empted session is recorded as failed with logs, never silently retried into a success; report incompleteness explicitly in the results registry (never impute missing cells). | B + orch | open |
| C2 | **Notebook drift.** A hand-edited notebook diverges from the generated, versioned config, so the executed code no longer matches the record. | M | H | Notebooks are **generated** from version-controlled configs by a generator; scientific logic lives in importable `src/spectraquant/**` modules, not cells; the `run_manifest.json` records the resolved config, config hash and generator/commit; a hand-edited notebook that cannot be regenerated is **not accepted as evidence**; CI checks that generation is reproducible. | B + M | mitigating |
| C3 | **Credential leakage into notebooks/logs.** Kaggle/GCP/HF/W&B/storage tokens end up in a notebook cell, a committed config, a log or a report. | M | H | Credentials only in platform secrets / environment variables (`AGENTS.md` §2b rule 7); secret-scanning pre-commit/CI; exported logs are scrubbed before they enter the research record; no credentials in configs, notebooks, reports or `run_manifest.json`; rotate on suspicion. | B + orch | mitigating |
| C4 | **A cloud run reported from logs without checksum-validated artifacts.** Numbers are quoted from console output or a partial upload, so the result cannot be audited. | M | H | The registry is updated **only** from downloaded, checksum-validated artifacts (`AGENTS.md` §2b rule 5); a run whose bundle fails validation is recorded as **failed** and excluded; `run_manifest.json` is schema-validated (`artifacts/schemas/`); the audit stream independently re-checks checksums before a result is cited. | J (audit) + B | mitigating |

## 2. Scientific risks

| ID | Risk | L | I | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| S1 | **Calibration-set leakage.** Calibration activations or statistics are drawn from dev/test, inflating quality and invalidating equal-memory comparisons. | M | H | Programmatic train/eval overlap detection before every run; calibration drawn only from the training split (preregistration §3); calibration-set hash recorded in the manifest; leakage is an admissible exclusion criterion. | I | mitigating |
| S2 | **Unfair comparison / unequal memory.** A method is compared against a baseline at unequal stored bytes or a lucky budget, favouring our method. | M | H | Every compression comparison reports an equal-stored-bytes counterpart (`AGENTS.md` §4.5); Pareto frontier evaluated over a predeclared budget ladder; byte accounting measured (Class 3). | J + H | mitigating |
| S3 | **Novelty overclaim.** A contribution is claimed without a cited closest neighbour despite a fast-moving 2024–2026 literature (RAM, MixQuant, MLoRQ, SRR, AutoQRA, KV-COBRA). | H | H | `novelty-risk.md` mandates a cited neighbour + explicit difference for every claim, uses "candidate contribution", and a re-check log; every claim re-checked before submission. | A | mitigating |
| S4 | **Hypothesis drift / HARKing.** After seeing results, a hypothesis is reframed so that the outcome supports it. | M | H | Preregistration frozen before confirmatory runs; hypotheses stated directionally; changes only by amendment; negative results published (`AGENTS.md` §4.10). | A + orch | mitigating |
| S5 | **Multiple comparisons inflate significance.** Many frontier points × arms × metrics produce a spurious "win". | H | M | Holm–Bonferroni within confirmatory families, Benjamini–Hochberg for declared exploratory sweeps (preregistration §7.4); fixed-n; effect sizes with CIs, not p-values alone. | J | mitigating |
| S6 | **Proxy is not causal.** A proxy that ranks layers well does not necessarily improve the allocated model — allocation benefit may vanish. | M | H | Two separate hypotheses (**H2** proxy validity, **H4** allocation benefit); H4 tested independently; if H2 passes and H4 fails the negative is reported and the contribution re-scoped. | F + H | open |
| S7 | **Regularizer is cosmetic.** The spectral/rounding penalty improves training metrics without changing post-quantization quality. | M | M | **H3** is explicitly an ablation (with/without; rounding term replaced by plain spectral term); evaluate at equal bytes on held-out data only. | G | open |
| S8 | **Comparator implementation is a straw man.** The HAWQ-V2/trace or magnitude baselines are implemented weakly, guaranteeing our win. | M | H | Comparators implemented from their own published definitions, unit-checked against hand computations; deviations recorded in method cards; a comparator review by a second stream. | F + J | open |
| S9 | **Metric gaming via dev peeking.** Repeatedly reading dev to steer the method, then reporting dev as if held out. | M | M | Dev is explicitly a selection set; the confirmatory read is on test, fixed-n, logged (preregistration §3.3, §10). | J | mitigating |

## 3. Reproducibility risks

| ID | Risk | L | I | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| R1 | **Cross-substrate non-reproducibility.** Local results depend on one CPU workstation (thread count, CPU flags); cloud results depend on the allocated instance type, driver and library versions, and free-tier sessions may differ between runs. | M | H | Pin the environment (`uv.lock`, Python 3.11), seeds and thread counts; require bit-for-bit CPU reproducibility for Tier-0 fixtures; every cloud run's `run_manifest.json` records hardware/driver/lock metadata; never compare numbers across differing substrates; document the workstation (`environment.md`). | B + J | mitigating |
| R2 | **Upstream drift.** Upstream repos move; a reimplementation matches the paper but not the released code (or vice versa). | M | H | Pin exact SHAs (`upstream-lockfile.md`); method cards record upstream commit + paper version + observed deviations; re-verify SHAs before release. | A + D | mitigating |
| R3 | **Missing/incorrect manifests.** A run's provenance (data hash, config, seeds, classes) is incomplete, so it cannot be audited. | M | M | Schema-validated manifests (`artifacts/schemas/`); invalid manifests are an exclusion criterion (preregistration §9.4); audit role (stream J) checks. | B + J | open |
| R4 | **Dataset revision drift.** A dataset is re-released and splits change, breaking comparability. | M | M | Record dataset identifiers and hashes at freeze; use a local pinned copy; record subsample indices. | I | open |
| R5 | **Environment/toolchain drift** (Windows-only tooling, Linux-only dependencies, torch version changes numerics). | M | M | Pin toolchain; Linux-only tooling via Docker if needed; record torch version and CPU flags in manifests; avoid cross-version comparisons. | B | open |
| R6 | **Post-freeze code changes silently alter results.** Code fixed after freeze changes numbers without a note. | M | H | Every confirmatory run records commit SHA; any post-freeze code change that affects results requires an amendment and a re-run; dirty-state flag disallows reporting a clean result from a dirty tree. | orch + J | mitigating |
| R7 | **Knowledge loss across agent waves.** Decisions taken by one agent session are unavailable to the next. | M | M | Decisions written to `docs/decisions/`, research docs, and this register; method cards persistent; ownership map maintained. | orch | mitigating |

## 4. Highest-priority watch list (reviewed at every gate)

1. **T1 no local GPU / cloud dependence** (H/H) — the single biggest constraint; drives the substrate split (`AGENTS.md` §2b).
2. **C1 session pre-emption / quota exhaustion** (H/H) — can silently truncate the run matrix; drives the resumable-slice design.
3. **C4 / R3 manifest and checksum integrity** (H) — a cloud number without a validated bundle and checksummed artifacts does not exist.
4. **T2 / T4 measurement and accounting integrity** (H) — a wrong class label or an omitted byte cost invalidates every compression claim.
5. **S1 calibration leakage** (H) — can silently inflate all quality numbers.
6. **C3 credential leakage** (H) — platform/secret hygiene; can compromise the whole substrate.
7. **S3 novelty overclaim / T9 scale overclaim** (H) — reputational and scientific risk in a crowded field.
8. **R6 post-freeze drift** (H) — turns confirmatory results into non-confirmatory ones.

Any change to a HIGH risk's status, likelihood, or mitigation is recorded as an amendment in
`preregistration-amendments.md` when it affects the protocol.
