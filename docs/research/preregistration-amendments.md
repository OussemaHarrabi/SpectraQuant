# SpectraQuant — Pre-registration Amendments (append-only)

This file is the **only** legal mechanism for changing the frozen protocol in
`docs/research/preregistration.md` (and for recording scope changes forced by hardware, per
`docs/research/environment.md` §5).

## Format rules (binding)

1. **Append-only.** Never edit or delete an existing entry. Corrections are made by *superseding*
   entries (`Amends: A-000N`), never by rewriting.
2. **One entry per decision.** Each entry uses this template:

```
## A-NNNN — <short title>
- Date (UTC): YYYY-MM-DDThh:mmZ
- Author: <agent/human id>
- Status: accepted | superseded by A-MMMM | withdrawn
- Amends: <section(s) of preregistration.md or earlier amendment id>   (omit if new)
- Trigger: <what was discovered / decided, with evidence>
- Change: <exactly what changes in the protocol>
- Rationale: <why; what would have been scientifically wrong without it>
- Evidence: <commands, file paths, hashes>
- Effect on frozen hypotheses: <none | which of H1..H5 are affected and how>
- Compute impact: <local Tier 0/1 | GPU-gated | none>
- Superseded-by: <amendment id or ->
```

3. **Numbers** run monotonically `A-0001`, `A-0002`, … Zero-padded to 4 digits, never reused.
4. An amendment that *changes a hypothesis, outcome, dataset, or test* MUST say so explicitly in the
   `Effect on frozen hypotheses` field. If it weakens a test, the weakening is stated in plain
   language and is itself a published fact.
5. Amendments recorded here are referenced from the results registry and from every report that
   depends on them.
6. Scope **reductions** (including "Tier 2+ cannot be run") are amendments; scope *expansions* that
   add claims are also amendments. Silent scope change is a protocol violation.

---

## A-0001 — Compute envelope forces Tier 0–1 local scope; Tier 2–5 gated on an unsecured GPU

- **Date (UTC):** 2026-10-08T20:20Z
- **Author:** LitAudit (research stream A)
- **Status:** **superseded in part** — point 4 by A-0003 (class 4-CPU is available), Tier-2+ framing by A-0004 (cloud substrate is the vehicle); the remainder stays in force. The original text below is unchanged (append-only).
- **Amends:** new (bootstrap); constrains `preregistration.md` §4, §8, §10.4 and `charter.md` §7
- **Trigger:** Environment audit performed at bootstrap on the machine of record
  (`docs/research/environment.md`, measured 2026-10-08) found: AMD Ryzen AI 7 350 (8C/16T),
  16.23 GB RAM, **no discrete GPU**, **no CUDA toolkit**, AMD Radeon 860M integrated GPU with no
  Windows ROCm support; PyTorch not installed and pinned to the CPU wheel index. Consequently
  CUDA-dependent components (bitsandbytes 4-bit kernels, TorchAO CUDA kernels, GPTQ/AWQ GPU kernels,
  vLLM CUDA, FlashAttention) are unavailable, and ≥1B-parameter training is not viable locally
  (≈2.2 GB fp16 weights plus activations/optimizer state against ≈15.1 GiB usable RAM).
- **Change:**
  1. Tier 0 (synthetic/tiny-net correctness) and Tier 1 (tiny-transformer proof, ≥5 seeds) are the
     **only locally executable** tiers and the only tiers that may produce confirmatory numbers now.
  2. Tier 2 (TinyLlama-1.1B-class), Tier 3 (second 0.6–1.7B), Tier 4 (ViT), Tier 5 (3B–7B) are
     reclassified as **GPU-gated / planned, not run**. They MUST be reported as "not run" until an
     external GPU is documented in `environment.md` and a subsequent amendment records the **GPU
     decision at the M1 freeze / M7 boundary** as secured.
  3. `preregistration.md` §8 marks each such cell **[GPU-GATED — NOT RUN locally]**.
  4. Measurement classes 4 (kernel-backed inference) and 5 (end-to-end service) are **not
     measurable** on this workstation and are always reported "not measured" (never estimated).
  5. Any code path requiring CUDA MUST fail loudly; silent numerical fallbacks are forbidden
     (`AGENTS.md` §2.2).
- **Rationale:** Without this amendment the charter would promise experiments the hardware cannot
  run, violating the binding compute constraint and `AGENTS.md` §2.3. Recording it at bootstrap
  keeps the compute limitation auditable and prevents later retrospective justification.
- **Evidence:** `docs/research/environment.md` §1 (hardware table), §2 (CUDA absent), §5 (compute
  policy); `AGENTS.md` §2 and §6; `references/upstream-verification.log` (upstream repos inspected
  on 2026-10-08).
- **Effect on frozen hypotheses:** none of H1–H5 is changed in *content*; their **executable scope**
  is constrained. H1–H4 may be tested at Tier 0/1 locally; H5 is only partially testable locally
  (packed-storage survival, class 3) and its kernel/latency half is not measurable. The Tier-2/3
  statements of all five remain unconfirmed projections and must be phrased as such. No hypothesis
  may be *claimed confirmed* from Tier-2+ cells until the GPU decision at M1/M7 is recorded as
  secured.
- **Compute impact:** defines local Tier 0/1 as the current scope; Tier 2–5 GPU-gated.
- **Superseded-by:** -

---

## A-0002 — Canonical specification hypothesis IDs (H1–H5) and milestone IDs (M0–M10)

- **Date (UTC):** 2026-10-08T21:05Z
- **Author:** LitAudit (research stream A)
- **Status:** accepted
- **Amends:** `preregistration.md` §5 (arm labels), §7.2 (comparators), §8 (test matrix),
  §10.4, §11 (failure modes, + new §11.0 traceability table), §13; `charter.md` §3, §4, §8, §9;
  `novelty-risk.md` §1–§3; `risk-register.md` S6/S7; `literature-review.md` §4.2/§5.1/§6;
  `upstream-lockfile.md` U1–U3; `literature-matrix.csv` (harness pin only)
- **Trigger:** orchestrator verification found a traceability defect: this file and the charter had
  *authored* an H1–H5 mapping ("preparation helps / order matters / proxy / allocation / regularizer")
  instead of using the hypothesis IDs pre-registered by the master specification. That broke the
  audit trail spec → test → result. The orchestrator also required milestone IDs M0–M10 instead of
  improvised gate labels (`G0`–`G7`).
- **Change:**
  1. **Canonical hypotheses adopted verbatim from the specification:** H1 = naive low-rank preparation
     can reduce factorisation error while increasing rounding sensitivity; H2 = a proxy incorporating
     activation covariance and quantization residuals correlates more strongly with downstream
     degradation than weight-space Frobenius error alone; H3 = a rounding-aware spectral regularizer
     improves compressed-model quality at equal memory relative to unprepared factorisation;
     H4 = layer-wise mixed rank and mixed precision under a global budget dominates uniform
     allocations on at least part of the quality–memory frontier; H5 = some apparent quality
     improvements under fake quantization will not translate into real latency or memory improvements
     unless a compatible packed kernel is used.
  2. **"Order of the two transforms" is demoted** from a hypothesis to supporting question **Q2** and
     a **comparator arm**; it never carries an H-id.
  3. **H5 is scoped operationally** to what this CPU-only machine can measure: the packed-storage
     survival test (class 2 → class 3). Its kernel-backed (class 4) and latency/throughput (class 5)
     half is **not measurable** and is reported "not measured"; H5 is therefore *partially* testable
     and is never claimed confirmed from local data alone.
  4. **Milestones:** improvised gates replaced by **M0–M10**; M1 = preregistration freeze (GPU
     decision recorded at the M1/M7 boundaries), M3 = reproduction, M4 = proxy gate, M7 = primary
     matrix. Milestones not defined in this repository are deferred to
     `docs/coordination/status.md`.
  5. **Traceability table added** at `preregistration.md` §11.0 (spec hypothesis → section/row →
     falsifier).
- **Rationale:** with a mismatched numbering, results tables could not be audited back to the
  specification hypotheses, and the "order matters" claim would have been silently promoted from a
  comparator to a hypothesis (a HARKing risk, register S4).
- **Evidence:** `docs/coordination/status.md` §5 (M0–M2 definitions); orchestrator IRC verification
  message (2026-10-08); this repository's `preregistration.md` §11.0.
- **Effect on frozen hypotheses:** **none of H1–H5 was reworded.** This entry fixes *labels* and
  *scope statements*; the five specification claims are unchanged. Note the document was DRAFT
  (never frozen), so this is a pre-freeze normalisation recorded for traceability, not a change to
  frozen text.
- **Compute impact:** none.
- **Superseded-by:** -

---

## A-0003 — Class 4 split into 4-CPU (available) / 4-GPU (deferred); H5 extended to a CPU-kernel survival test

- **Date (UTC):** 2026-10-08T21:40Z
- **Author:** LitAudit (research stream A)
- **Status:** accepted
- **Amends:** `preregistration.md` §1 (class note), §5 (H5 arm), §8 (test matrix rows), §11.0
  (traceability H5 row), §11.1 (H5 falsifier), §13 (checklist); `charter.md` §4 (H5), §5 (non-goal 3),
  §6 (class table), §7 (envelope correction); `risk-register.md` T2; `literature-review.md` §3.5;
  `upstream-lockfile.md` §5. **Amends A-0001 §Change point 4** (its blanket statement that class 4 is
  "not measurable" and always "not measured" is corrected by this entry); A-0001 remains otherwise in
  force and is not rewritten (append-only).
- **Trigger:** measurement during the same wave established that a **real CPU low-bit kernel path
  exists on this workstation**, contradicting the earlier blanket reading "no CUDA ⇒ no real low-bit
  kernel" (which A-0001 had encoded). Evidence: `docs/research/backend-capability.md` §2.4 and the
  "Corrected on 2026-10-08" section of `docs/research/environment.md` §1 — ONNX Runtime 1.30.0 CPU
  executes `MatMulNBits` (int4 weight-only, `com.microsoft`) and `MatMulInteger` (dynamic int8); a
  self-serialized int4 artifact holds a 1024 B payload plus 256 B of fp32 scales inside a 1501 B file
  and runs; torchao 0.18.0 `IntxWeightOnlyConfig(torch.int4, PerGroup(32))` forwards on CPU. The
  orchestrator amended `AGENTS.md` §5 accordingly and independently reproduced the finding.
- **Change:**
  1. **Class 4 is split by hardware:** **4-CPU available** — a real CPU kernel executes a low-bit
     artifact **SpectraQuant itself serialized** (permitted: ONNX Runtime CPU `MatMulNBits`/
     `MatMulInteger`, torchao intx weight-only); a 4-CPU claim MUST name backend, kernel/op, container,
     thread count and CPU model, MUST compare against an fp32 CPU baseline measured in the same session
     on the same machine, and MUST NOT be presented as comparable to published GPU latency/throughput
     numbers or used to make any latency/throughput claim. **4-GPU unavailable/deferred**. Third-party
     containers (llama.cpp GGUF, externally produced ONNX) are engineering telemetry only, never
     class 4 for a SpectraQuant artifact. **Class 5 remains unavailable.**
  2. **H5 operationalisation extended** to three parts: (a) packed-storage survival (class 2 → 3);
     (b) **CPU-kernel survival (class 4-CPU)** of our own container vs. an fp32 CPU baseline;
     (c) GPU-kernel (class 4-GPU) and service (class 5) halves — **not runnable here, "not measured"**.
     H5 may be reported at most as **partially supported / inconclusive at GPU scope**; it is
     confirmed only if the gain survives (a), (b) **and** a documented GPU kernel path (Tier 2, M7,
     GPU-gated).
  3. Locally available measurement classes become **1, 2, 3, 4-CPU** instead of 1, 2, 3.
- **Rationale:** leaving the old claim ("classes 4–5 unavailable") in place would be factually wrong
  and would wrongly forbid a legitimate, locally verifiable test of the H5 mechanism. The restriction
  to self-serialized containers plus same-session fp32 baselines keeps class 4-CPU honest and prevents
  it from being smuggled into latency or GPU-comparability claims.
- **Evidence:** `docs/research/backend-capability.md` §2.4 (§2.4 byte layout probes), `docs/research/environment.md` §1
  "Corrected on 2026-10-08", `AGENTS.md` §5 (amended), orchestrator IRC instruction 2026-10-08.
- **Effect on frozen hypotheses:** scope of **H5's testability only**; no hypothesis is reworded. The
  document was still DRAFT (unfrozen), so this is a pre-freeze scope normalisation recorded for
  traceability. H1–H4 are unaffected; H5 gains a locally testable half and remains **never fully
  confirmable** on this workstation.
- **Compute impact:** local Tier 0/1 now includes class 4-CPU kernel checks for self-serialized
  artifacts; Tier 2–5 remain GPU-gated.
- **Superseded-by:** -

---

## A-0004 — Execution substrate: local = orchestration/correctness + class 4-CPU; cloud notebook substrate = all Tier 1–5 runs

- **Date (UTC):** 2026-10-08T22:10Z
- **Author:** LitAudit (research stream A)
- **Status:** accepted
- **Amends:** `preregistration.md` §4 (model families + substrate), §6 (seed plan), §8 (test matrix,
  new substrate column), §10 (stopping rules 4), §11.0/§11.1 (substrate wording), §12 (cloud
  reproducibility), §13 (checklist); `charter.md` §4 (H5), §6 (class table), §7 (envelope + tier
  table + substrate note), §8 (milestones M1/M3/M4/M7); `risk-register.md` T1, new §1b (C1–C4), R1,
  watch list. **Amends A-0001** (its "Tier 2+ gated on an unsecured GPU / Tier 1 local" framing is
  replaced by the cloud-substrate framing); A-0001 otherwise unchanged (append-only).
- **Trigger:** user instruction (2026-10-08): the local machine may **not** run research training,
  QAT, large-scale inference or GPU evaluation. Local scope = repository management, CPU unit/property
  tests, tiny synthetic fixtures, static analysis, config validation, notebook generation, result
  analysis, figures/tables/reports. All Tier 1–5 experiments execute on a cloud notebook substrate —
  **Google Colab** (developer's chosen vehicle; consumer Colab has no submission API, so notebooks are
  generated and run by the developer), **Kaggle Notebooks** (preferred unattended backend, programmatic
  CLI), **Colab Enterprise** (only with an authorized GCP project). Full text: `AGENTS.md` §2b, §2.3,
  §6, §7; `docs/research/environment.md` §5.
- **Change:**
  1. **Substrate tags** introduced in the test matrix: `LOCAL-FIXTURE` (Tier-0 synthetic/unit work),
     `LOCAL-CPU-MEASUREMENT` (class 4-CPU kernel measurement on a self-serialized artifact),
     `CLOUD-COLAB` (Tier-1 training/eval), `CLOUD-GPU` (Tier 2–5). Tier-0 cells → `LOCAL-FIXTURE`;
     the class 4-CPU ONNX/torchao arm → `LOCAL-CPU-MEASUREMENT` (a measurement, not training);
     **all Tier-1 tiny-transformer cells → `CLOUD-COLAB`** (previously "local"); Tier 2/3/4/5 →
     `CLOUD-GPU`; anything requiring CUDA on the local host keeps a **NOT RUNNABLE locally** tag and
     must fail loudly.
  2. **"Cloud" is a plan, not a result**, until a validated `run_manifest.json` (run id, resolved
     config, dependency lock info, git commit SHA, dataset versions/checksums, seeds, hardware
     metadata, start/end, GPU-hours, status, metrics, artifact checksums) and checksum-validated
     artifacts exist. No Tier-1+ number may be reported as measured locally.
  3. **Stopping rules** replace the "GPU decision point": (a) free-tier cloud assumed for Tier 1–2
     subject to session limits; (b) Tier 3–5 and any paid instance require explicit user authorization
     + a stated maximum cost before submission; (c) a pre-empted session is resumed via the persisted
     remote run id or recorded as **failed**, never silently retried into a success; (d) a run whose
     artifact bundle fails checksum validation **does not exist**.
  4. **New cloud risks** C1–C4 added to the risk register (pre-emption/quota; notebook drift;
     credential leakage; unvalidated-artifact reporting), each with mitigation and owner.
  5. The earlier framing "GPU not yet secured ⇒ Tier 2+ planned" is **replaced** by: "the cloud
     notebook substrate is the execution vehicle; Tier 2+ still requires a cloud GPU session and
     remains reported as **not run** until a validated run exists."
- **Rationale:** the previous plan had Tier 1 running locally; the user's policy makes that impossible
  and moves the whole training/evaluation workload to the cloud. Leaving the old wording would
  misrepresent where results come from and would permit unauditable numbers. The manifest + checksum
  + "generated, not hand-edited notebook" rules keep the cloud record auditable from a machine that
  cannot itself run the experiments.
- **Effect on frozen hypotheses:** **execution substrate only.** No hypothesis, primary/secondary
  outcome, dataset, model family, seed plan (counts) or statistical test is reworded. H1–H4's test
  cells now specify the substrate; H5's CPU-scope halves remain local, its GPU half is `CLOUD-GPU` and
  claimable only from a validated cloud run. The document was still DRAFT (unfrozen), so this is a
  pre-freeze scope normalisation recorded for traceability.
- **Evidence:** `AGENTS.md` §2b (cloud execution policy), §2.3, §6, §7; `docs/research/environment.md`
  §5 ("Superseded 2026-10-08 by the cloud-compute execution policy"); user instruction relayed
  2026-10-08.
- **Compute impact:** all Tier 1–5 runs move to the cloud substrate; local compute limited to Tier-0
  fixtures, class 4-CPU measurement, and analysis. Free tiers default; paid instances authorized only
  with a cost ceiling.
- **Superseded-by:** -

---

## A-0005 — Adversarial-review corrections: novelty re-verdicts, H2 unit redesign, byte-parity/serializer freeze, cloud budget line, exclusion constants

- **Date (UTC):** 2026-10-08T22:40Z
- **Author:** LitAudit (research stream A)
- **Status:** accepted
- **Amends:** `novelty-risk.md` §0–§5; `literature-review.md` §2.2, §2.7, §5.1, §5.12, §5.13, §7;
  `literature-matrix.csv` (2 rows added, 5 rows corrected, 4 venue fields); `references/bibliography.bib`;
  `preregistration.md` §3.3, §4, §5, §6, §7 (new §7.0/§7.5), §8, §9, §10.4(a), §11.0/§11.1, §13.
  Also records that **A-0001 is now marked "superseded in part"** (point 4 by A-0003, Tier-2+ framing by
  A-0004) and that `eval-protocol.md` §1's stale class-4 sentence (B13), the pinned ONNX/torchao extra
  (B14) and the equal-memory gate implementation (B16) are **out of this stream's write-set** and are
  owned by the protocol/scaffold streams.
- **Trigger:** the independent adversarial review of the M1 foundation
  (`docs/results/verification/m1-novelty-review.md`, dated 2026-10-08) — its §1 citation audit and its
  §9 blocking list **B1–B17** — together with the raw evidence it produced under
  `docs/results/verification/raw/` (`citation_check.out`, `abs_2411.05007.html`, `abs_2607.12550.html`,
  `probe_stats.out`, `probe_compose.out`, `probe_estimator.out`, `probe_throughput.out`, `quote_ledger.txt`).
  The review found (a) two mischaracterised neighbour rows, one misattributed nearest neighbour and one
  uncited obvious neighbour; (b) an uncited work that falsified a §0 coverage claim; (c) four venue
  understatements; (d) a statistical plan that was "merely described" rather than pre-registered (no
  unit, no pinned layer count, no power/MDE, no family enumeration, no numeric margins or exclusion
  constants); (e) H5 row 8 tagged locally-runnable though it needs a quality evaluation the substrate
  policy forbids locally; (f) a byte-parity rule with no tolerance and an unfrozen serializer; and
  (g) a cloud plan with no budget line.
- **Change:**
  1. **Novelty corrections (B1–B6).** LQ-LoRA's P6/C6 rows and contribution-A's difference statement are
     rewritten to the fetched abstract (per-matrix **bit-width and block size** configured by an **ILP**
     under **one** total memory budget; **data-aware variant weights the reconstruction objective by a
     Fisher-information approximation**; per-matrix sequential allocation; no training-time regularizer).
     The false "bit-agnostic"/"fixed bit width" claims are deleted and A's difference is re-derived as the
     **estimator form + equal-stored-bytes evaluation protocol only**. HAWQ-V2's row is corrected (average
     Hessian eigenvalue; exact Pareto-frontier bit selection; layer Hessian from **layer input statistics**;
     extended to activation quantization) keeping only "no activation propagation, no rank". Spectrum is
     corrected to **SNR-based module selection with the rest frozen** (not reweighting).
     **SVDQuant (arXiv:2411.05007, ICLR 2025 Spotlight)** and **JoLT (arXiv:2607.12550)** are added to the
     matrix, the review and the novelty tables; the four venue fields (GPTQ/APTQ/SpectralNorm/SVD-LLM) are
     set from the abs-page comments field, keeping `venue_verified` honest.
  2. **Novelty re-verdicts.** A: "differentiated (weakly)" → **"differentiated (narrow, empirical)"**
     (the $(r,b)$ currency is already occupied — P6/SVDQuant; the old basis was false). B: "differentiated
     (moderate)" → **"differentiated (weak–moderate, empirical)"** (the search had missed SVDQuant, whose
     low-rank branch is designed around quantization's representational limits and whose Nunchaku engine
     answers the same kernel-survival question as H5). C: **unchanged** ("previously-known"), now
     additionally supported by JoLT (C7). The §0 coverage claim is replaced by the honest statement that
     *bit-rank allocation* was searched **and a title-exact 2026-07 hit was missed**, logged in §5 with
     the date and the reproducing queries.
  3. **H2 redesign — one primary unit (B7/B8).** H2's resampling unit becomes the **model**: one correlation
     per trained model, combined across seeds by a **Fisher-z random-effects model** (DerSimonian–Laird τ²),
     with layers/modules as a **within-model nuisance** and the module-level granularity declared
     (4 linear modules per block; pinned `L_b = 8` ⇒ n = 32). Justification is the review's measured
     numbers: modules are heterogeneous strata (0.919 vs 178.2 proxy at the same depth), the layer
     bootstrap is degenerate at small n with no tie rule, and the layer-level power for Δρ=0.17 is
     **0.036 (L=8) / 0.082 (L=12)**, rising only to **0.446 at L=48** (`raw/probe_stats.out`). A
     **model-level MDE statement** is added (§7.5): at S=5 seeds the paired Fisher-z difference has
     MDE(z) ≈ 0.33 (Δρ ≈ 0.12 at ρ≈0.8, optimistic n_eff=32) to ≈ 0.79 (Δρ ≈ 0.29, conservative n_eff=8),
     with the achieved `n_eff` reported per model. The Tier-1 layer/block count and config are **pinned**
     (`L_b=8`, `d=256`, seq 256, 5 epochs, ≈32 M params).
  4. **Statistical constants (B9/B10).** §7.4 enumerates the confirmatory families with their `m`
     (F1 = {H2 vs primary comparator}, m = 1, Holm deleted as vacuous; F2 = {H1, H3, H4}, m = 3;
     F3 = {H5a, H5b}, m = 2). Numeric constants are declared: frontier equivalence margin **5 %** relative
     byte saving; matched-quality tolerance **0.01 nats/token**; test reads **exactly 2** per arm; and the
     four §9 exclusion constants (divergence factor **2.0** over a **500-step** patience window; per-run
     wall-clock caps **1 h / 4 h / 24 h / 48 h** by tier; byte-accounting tolerance **0.5 %**).
  5. **H5 testability (B11/B12).** §8 row 8 is **split** into a local **class-3 byte** measurement and a
     `CLOUD-COLAB` **class-2 quality** re-measurement; the **serializer invocation is frozen** (one entry
     point, identified by `compare.serializer_build`), and the **byte-parity tolerance** is predeclared at
     `|Δbytes|/bytes ≤ 0.5 %` (container overhead is invocation-dependent: 221–363 B on a 1,280 B payload;
     nominal 4.0 → 5.86–6.42 measured bits/param). The packing/kernel chain is scoped to **b ∈ {4, 8}**
     (only `pack_int4`/`pack_int8` exist; permitted kernels are int4/int8), and the 2/3-bit arms are
     excluded from every H5 statement explicitly.
  6. **Cloud budget line (B15).** §10.4(a) gains a GPU-hour ceiling (**30 GPU-h/week** free tier), session
     (≤12 h) and quota assumptions, a maximum arm×seed count, and a predeclared ladder for what happens
     when the ceiling binds, based on the review's measured CPU throughput (1,158–2,375 tok/s ⇒ ≈0.9–9
     GPU-h per Tier-1 arm-seed at a 20–50× scale factor; 6 arms × 5 seeds ≈ 27–270 GPU-h).
  7. **B17 bookkeeping.** The stale "GPU decision at M1/M7" item is re-pointed to the A-0004 substrate
     decision in `preregistration.md` §13; A-0001's status is marked "superseded in part"; and no
     "classes 4–5 unavailable" phrasing remains as a *live* statement in this stream's files (locally
     available classes are 1, 2, 3, 4-CPU).
- **Rationale:** the review demonstrated that the previous novelty text was contradicted by the fetched
  abstracts it cited and that the statistical plan could not be executed as written (no unit, no `m`, no
  margins, no exclusion constants, no MDE). Leaving either in place would have made the M1 foundation
  unauditable and the confirmatory tests unfalsifiable-as-registered. Every correction is traced to a
  fetched artefact under `docs/results/verification/raw/` or to a command run by this stream.
- **Evidence:** `docs/results/verification/m1-novelty-review.md` §1 and §9 (B1–B17);
  `docs/results/verification/raw/citation_check.out` [P2][P3][P4a][P6][R3][R4][R5b][UNCITED-SVDQuant][UNCITED-JoLT];
  `raw/abs_2411.05007.html`, `raw/abs_2607.12550.html`; `raw/probe_stats.out` (layer-level power/MDE);
  `raw/probe_compose.out` (module heterogeneity, 0.919 vs 178.2); `raw/probe_throughput.out`
  (1,158–2,375 tok/s; 0.28–0.58 h/epoch); `raw/probe_estimator.out` (relative SE). Local checks run by
  this stream on 2026-10-08: a `python` CSV parse of `literature-matrix.csv` (**52 data rows, 19 fields,
  0 malformed, 0 duplicate keys**), a brace-balance check of `references/bibliography.bib`
  (**depth 0, 52 entries**), and the MDE computation
  `MDE(z) = (z_{0.975}+z_{0.80})·sqrt(2/(n_eff−3))/sqrt(S)` printed for n_eff ∈ {32, 8}, S = 5
  (**0.329 / 0.792**). Search queries reproducing the missed neighbours: `"joint rank-bit allocation"
  arXiv JoLT KV cache` (returns JoLT title-exact) and `SVDQuant low-rank branch absorb outliers 4-bit
  diffusion arXiv 2411.05007`.
- **Effect on frozen hypotheses:** **H2's test statistic and resampling unit change** — from a per-layer
  bootstrap to a **model-level Fisher-z random-effects paired contrast**, with layers as a within-model
  nuisance, and its falsifier gains an **inconclusive-if-wider-than-MDE** clause. This is a change to an
  item that would otherwise be frozen, and it is justified **before any data exists** precisely because
  the review measured the old unit to be non-exchangeable and underpowered (power 0.036–0.082 at the
  pinned size), and because pre-registration exists to fix the unit *before* seeing results — recording it
  now, pre-freeze and pre-data, is the honest ordering. **No other hypothesis is reworded**: H1, H3, H4
  keep their canonical wording; H5 keeps its three-part operationalisation and gains only the `b ∈ {4,8}`
  packing scope and the row split. No primary/secondary outcome, dataset, or seed *count* is changed.
  The document was still **DRAFT (never frozen)**, so this is a pre-freeze correction recorded for
  traceability.
- **Compute impact:** none new; the cloud budget line makes the existing Tier-1–5 cloud cost explicit and
  bounds the sweep. Local scope unchanged (Tier-0 fixtures, class 4-CPU measurement, analysis).
- **Superseded-by:** -

---

## A-0006 — M1 freeze: plan-vehicle change, comparator completion, deferred checklist items

- **Date:** 2026-10-09.
- **Trigger:** the M1 freeze audit of the §13 checklist, performed by the orchestrator after the
  adversarial review's blocking issues were closed and the milestone-2 slices landed.
- **Change:**
  1. **Tier-1 cloud vehicle.** The Tier-1 cloud pilot is the pinned pretrained model
     `HuggingFaceTB/SmolLM2-135M` (`configs/tier1/smollm2_135m.yaml`), while the *from-scratch*
     decoder-only transformer that §4 originally named remains the Tier-0/Tier-1 **fixture**
     (`src/spectraquant/evaluation/toy.py`, seeded, CPU, used for proxy validation). Both vehicles
     exist; the split is explicit so no reader has to guess which one produced a number.
  2. **Comparator set completed.** §13 required "weight magnitude, activation magnitude, Hessian
     trace, weight-space Frobenius" comparators. `weight_magnitude` and `activation_magnitude` are
     now implemented and unit-checked alongside `hessian_diag` and `weight_frobenius`
     (`src/spectraquant/proxies/variants.py`), so the H2 comparator set is *implemented* rather than
     merely named.
  3. **Deferred, with reason, at freeze:**
     - the **class-4-CPU ONNX fixture** (int4 `MatMulNBits` / int8 `MatMulInteger` container written
       by us, its runner and the same-session fp32 baseline) is deferred to M6/M9: the environment is
       pinned (`onnx` extra, `uv.lock`) and the capability is verified
       (`docs/research/backend-capability.md` §2.4), but the export path is not yet implemented, so
       **no H5 CPU-kernel result may be claimed** and the H5 row stays "not run".
     - the **random-probe comparator** (arXiv:2609.33923) is deferred: it is an *optional* addition to
       the predeclared comparator set, not part of it, and no hypothesis depends on it.
     - the **Tier-1 corpus subsample hash** cannot be produced before the run: the selection *rule*
       and seed are frozen (`eval-protocol.md` §3.2) and the hash is emitted by the cloud run's
       `run_manifest.json`, which is checksum-validated at collection. Locally hashing it would
       require downloading the corpus, which the substrate policy assigns to the run.
- **Effect on frozen hypotheses:** **none reworded.** H1–H5 keep their canonical wording; H2's
  comparator set is now implemented; H5's CPU-kernel half is explicitly *not runnable at freeze*.
  The vehicle split (change 1) is a scope clarification, not a hypothesis change.
- **Evidence:** `configs/tier1/smollm2_135m.yaml`, `configs/tier2/tinyllama_1_1b.yaml`,
  `configs/repro/lr_qat_smollm2_135m.yaml` (all schema-validated by `tests/unit/test_experiment_plan.py`);
  `artifacts/sample-results/proxy-fixture/proxy-fixture.json`; `docs/results/proxy-fixture-report.md`;
  `docs/results/allocator-report.md`; `docs/coordination/status.md` §5.
- **Amends:** A-0004 (substrate), A-0005 (review corrections). Append-only: neither is rewritten.
- **Superseded-by:** -

---

## A-0007 — Candidate proxy declared; M4 local-fixture half complete; class-4-CPU fixture implemented

- **Date:** 2026-10-09.
- **Trigger:** the M4 local-fixture sweep (`docs/results/proxy-validation-report.md`) reported that the
  frozen documents left *which variant is the candidate* ambiguous: the design note calls `combined`
  the candidate interface while the coordination record named the gain-aware variant. The sweep
  therefore carried both and produced a verdict table for each.
- **Change:**
  1. **The candidate proxy for H2 is `gain_aware_composed`** — the per-layer output error multiplied by
     the squared *estimated downstream gain*. The predeclared H2 falsifier applies to it.
  2. **`combined` is retained as an ablation**, not as the candidate: its weights are documented as
     untuned, and on the local fixture its model-level ranking is significantly *worse* than
     `weight_frobenius` and `weight_magnitude` in several aggregate contrasts (Fisher-z CI excluding
     0 below zero). It may not be reported as the candidate until its weights are justified by an
     ablation.
  3. **M4 local-fixture half recorded.** Substrate `LOCAL-FIXTURE`, 10 trained models, 12 cells
     (6 (rank, bits) settings × 2 fixtures), seeds `[0..4]`, model-level unit with Fisher-z
     random-effects aggregation. Result: `gain_aware_composed` beats every predeclared comparator in
     **all 15 aggregate contrasts** (positive Fisher-z advantage, random-effects CI excluding 0);
     achieved MDE(z) 0.476–0.715, inside the predeclared 0.33–0.79 band. The **cloud Tier-1 H2/H4 cells
     remain NOT RUN** and are reported as such.
  4. **Class-4-CPU fixture implemented** (closing the §13 deferred item): our own int4 `MatMulNBits`
     and int8 `MatMulInteger` containers, the CPU-kernel runner with the benchmark-protocol §8 rules,
     and the same-session fp32 baseline; measured container bytes 1662 B (int4, graph + external
     sidecar) and 2727 B (int8). This makes the H5 CPU-kernel cell runnable; it does **not** by itself
     confirm H5.
- **Effect on frozen hypotheses:** none reworded. The change is a *definition* of which variant the
  H2 falsifier applies to, plus the completion of a deferred checklist item. H5 keeps its three-part
  operationalisation and remains at most "partially supported / inconclusive at GPU scope".
- **Evidence:** `docs/results/proxy-validation-report.md`,
  `artifacts/sample-results/proxy-validation/proxy-validation.json`,
  `docs/results/class4cpu-report.md`, `artifacts/sample-results/class4cpu/class4cpu.json`
  (re-executed by the orchestrator, which reproduced the byte and error figures).
- **Amends:** A-0006 (which deferred the class-4-CPU fixture), A-0005 (H2 unit). Append-only.
- **Superseded-by:** -
