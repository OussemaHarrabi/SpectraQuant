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

---

## A-0008 — Regularizer objective: predeclared term replaced by a smooth surrogate; H3 fixture result is mixed

- **Date:** 2026-10-09.
- **Trigger:** the Milestone-5 exploratory sweep (`docs/results/regularizer-report.md`). The
  predeclared "simplest viable penalty" for the H3 arm was the exact rounding-residual ratio
  `||Q(B)Q(A) - B A||_F^2 / ||B A||_F^2`. Measured, its straight-through gradient is **degenerate**:
  with `fq(X) = deq(X) + (X - X.detach())` the Jacobian of `fq` is the identity, so the surviving
  gradient of the residual norm is a product of two rounding errors and carries almost no descent
  information — sweeping `lambda_residual` did **not** lower the measured residual
  (0.0740 → 0.0791, i.e. it rose).
- **Change:** the working objective uses a smooth, scale-weighted surrogate of the factors' squared
  rounding error on the target grid, `(s^2/4) * sin^2(pi x / s)` with the detached block scale `s`.
  It has the same zero set (the code grid), the same cell amplitude and the same scale weighting as
  `delta^2`, but a genuinely non-zero gradient. The exact ratio term is retained in the code and
  monitored, so the predeclared quantity is still reported. The spectral term remains the
  plain-spectral control of the frozen H3 falsifier and is off by default (it is numerically zero
  when `tail_rank >= rank`).
- **Result recorded (exploratory, Tier-0 fixture, 5 seeds, equal accounted bytes 11 680 B):** the
  regularized arms **reduce** the measured rounding residual (product residual −35 % for `full`,
  −90 % at `lambda_round=30`) and **improve** fidelity to the dense reference (−36 % squared logits
  error), but **worsen** dev NLL (+638 % for `full`: 0.0238 → 0.1755). The two quality axes disagree
  in sign on this fixture, so **H3 is not supported here**; per the frozen §11 falsifier this is
  reported as a mixed/inconclusive fixture result, not as an improvement. The confirmatory H3 cell is
  `CLOUD-COLAB` and remains **NOT RUN**.
- **Effect on frozen hypotheses:** none reworded. The change is a *method* detail (which penalty is
  used) plus the publication of a mixed fixture result. It does not touch outcomes, datasets, seeds or
  tests.
- **Evidence:** `docs/results/regularizer-report.md`, `artifacts/sample-results/regularizer/sweep.json`
  (deterministic across independent runs, re-executed by the orchestrator),
  `tests/unit/test_regularizer_objective.py`, `tests/integration/test_training_loop.py`.
- **Amends:** A-0007 (candidate/method context). Append-only.
- **Superseded-by:** -

---

## A-0009 — Post-freeze manifest-schema extension (audit issue B5)

- **Date:** 2026-10-09.
- **Trigger:** the independent claim audit found that `artifacts/schemas/run-manifest.schema.json` was
  changed in commit `c36f15f` (the `compression.method` enum gained `"svd"`, needed by the plan
  runner's `low_rank_only` arm) **without** an amendment, although the schema was an M1 freeze
  checklist item. A silent post-freeze change is not acceptable, so it is recorded here.
- **Change:** the schema is declared **append-only extensible for enumeration widening only**: adding
  a permitted value for an existing field (a new method/arm label, substrate or class string) is
  allowed and must be recorded here with its commit; *removing* a value, changing a field's type, or
  invalidating a previously valid document requires an amendment **and** a migration note. The `"svd"`
  addition is the first recorded instance.
- **Effect on frozen hypotheses:** none - a label is added; no outcome, dataset, seed, test or
  hypothesis wording changes, and no existing manifest is invalidated (the committed sample manifests
  still validate).
- **Evidence:** `git show c36f15f -- artifacts/schemas/run-manifest.schema.json`;
  `docs/results/verification/claim-audit.md` section G issue B5.
- **Amends:** A-0006 (freeze checklist). Append-only.
- **Superseded-by:** -

---

## A-0010 — Candidate proxy simplified to the gain-aware form (ablation-driven)

- **Date:** 2026-10-09.
- **Trigger:** the Milestone-8 component ablation (`docs/results/ablation-analysis` — see
  `docs/results/ablation-report.md` and `artifacts/sample-results/ablations/ablations.json`).
- **Measured:** the downstream-gain term carries the ranking signal (candidate vs the naive per-layer
  error: Fisher-z +0.781, CI [+0.552, +1.009]), while **adding the in-situ normalisation Jacobian does
  not help** (pooled contrast −0.468, CI [−0.720, −0.216]; inconclusive on the LayerNorm fixture).
  The untuned `combined` variant is dominated by the candidate it contains (+0.781).
- **Change:** the declared candidate proxy for H2 is the **gain-aware form** — per-layer squared output
  error multiplied by the squared estimated downstream gain — **without** the in-situ
  normalisation-Jacobian variant. The in-situ variant, the naive per-layer error and the untuned
  `combined` remain implemented as comparators/ablations and must still be reported. This follows the
  specification's rule to prefer the simpler alternative unless an ablation demonstrates it is
  inadequate: here the ablation demonstrates the *opposite*, so the simpler form is the candidate.
- **Effect on frozen hypotheses:** none reworded. H2's falsifier and unit are unchanged; the candidate
  it applies to is now defined by measured ablation rather than by construction order. H3/H4 unaffected.
- **Evidence:** `docs/results/ablation-report.md` (component table with contrasts and CIs),
  `artifacts/sample-results/ablations/ablations.json`, `tests/unit/test_proxy_ablations.py`.
- **Amends:** A-0007 (which declared `gain_aware_composed`; that variant's implementation already
  equals the gain-aware form when the in-situ context is absent, and the ablations report both).
- **Superseded-by:** -

---

## A-0011 — Plans declare their compute device; Kaggle accepted as the free-tier execution backend

- **Date:** 2026-10-09.
- **Trigger:** preparing the first real cloud submission exposed a latent defect. `plan_to_run_spec`
  derived `gpu_required` from the plan's *tier*, so every plan requested a GPU - while the pinned
  environment installs **CPU-only torch** (`pyproject.toml` pins the PyTorch CPU index). A run would
  therefore have consumed GPU quota (Kaggle's weekly GPU allowance) to compute on CPU. The
  `--no-gpu` override is (correctly) refused on the plan path, so the plan itself had to change.
- **Change:**
  1. `PlanConfig` gains a `device: "cpu" | "cuda"` field (default `cpu`). `gpu_required` in the derived
     `RunSpec` follows this field, and the runner command now carries `--device <device>`, so the
     device is recorded in the spec and therefore in the run manifest.
  2. All three frozen plans declare `device: cpu` explicitly: their arms are the non-trainable
     PTQ / low-rank / rank-then-quant arms plus (later) the trainable arms, all of which execute in
     float on CPU under the pinned environment. A plan that needs CUDA must say so and must also
     declare how a CUDA torch build is obtained.
  3. **Execution backend for the first runs: Kaggle Notebooks** (the free-tier unattended backend the
     cloud policy names) rather than Colab, because the developer chose the programmatic path. Both
     are free tier; the plans' cost envelopes (`free_tier_only: true`,
     `max_cost_authorized_usd: 0.0`) are unchanged, and the plan files still record `substrate: colab`
     as the developer's declared vehicle. The override is recorded here rather than applied silently.
- **Effect on frozen hypotheses:** none. The device field changes no outcome, dataset, seed, grid or
  test; it corrects which hardware the plan asks for and records it.
- **Evidence:** `configs/{tier1,tier2,repro}/*.yaml`, `src/spectraquant/experiment_plan.py`,
  `src/spectraquant/cloud/spec.py`, `tests/unit/test_experiment_plan.py`, `tests/unit/test_plan_spec.py`
  (966-test suite green).
- **Amends:** A-0009 (schema-extension rule), A-0006 (freeze checklist). Append-only.
- **Superseded-by:** -

---

## A-0012 — The trainable plans move to `device: cuda`; the CUDA torch build is installed on the platform

- **Date:** 2026-10-09.
- **Trigger:** the M3 reproduction arms (`lr_qat`, `loftq`) became runnable, and a probe verified that
  the free Kaggle tier provides **2x Tesla T4 with CUDA 12.8** to this account
  (`docs/research/backend-capability.md` §9). Training those arms on CPU is not viable, so their plans
  must request the GPU. Two blockers had to be cleared first:
  1. A-0011 set `device: cpu` for all three plans because the pinned environment installs **CPU-only
     torch** (`[tool.uv.sources]` resolves torch from `download.pytorch.org/whl/cpu`). A plan that
     asks for a GPU while the environment holds a CPU build would consume quota and compute on CPU.
  2. `gpu_required` follows the plan's `device`, so the field - not the tier - decides.
- **Change:**
  1. `configs/repro/lr_qat_smollm2_135m.yaml` and `configs/tier2/tinyllama_1_1b.yaml` declare
     `device: cuda`. Their arms are trainable, so the GPU is the declared substrate.
  2. `_default_plan_install_spec(device)` materialises the lock and, for `device == "cuda"`, replaces
     **only torch** with the CUDA build of the *locked* version (2.14.1) from
     `https://download.pytorch.org/whl/cu128`. The version is pinned to the lock so the installed set
     and the lock agree, and the exact command is recorded in the spec, hence in the run manifest.
  3. `configs/tier1/smollm2_135m.yaml` **stays `device: cpu`**. Its trainable arm is the M5 arm, which
     is not implemented, so a Tier-1 run today is the non-trainable comparator slice - and that slice
     was measured on CPU and is recorded as such
     (`docs/results/tier1-cloud-run-2026-10-09.md`). Moving it to CUDA now would invalidate the
     comparability of an existing measurement. When the M5 arm lands, Tier-1 gets its own amendment
     and a GPU run.
- **Effect on frozen hypotheses:** none. The device changes no outcome, dataset, seed, grid or test.
  It determines which hardware a run asks for and which torch build is installed, both recorded in
  the manifest.
- **Evidence:** `configs/{repro,tier2}/*.yaml`, `src/spectraquant/cloud/spec.py`,
  `tests/unit/test_plan_spec.py`, `docs/research/backend-capability.md` §9.
- **Amends:** A-0011 (device field and backend), A-0006 (freeze checklist). Append-only.
- **Superseded-by:** -

---

## A-0013 — The T2-a oracle is the pinned LoftQ repository's CPU quantizer, not PEFT's `loftq_init`

- **Date:** 2026-10-10.
- **Trigger:** preparing the frozen M3 exactness gates, the predeclared T2-a oracle could not be
  executed as written. `docs/research/reproduction-plan.md` §5.3 specifies "our LoftQ `T=1` output vs
  PEFT's `loftq_init` (`num_bits=2`, `method="normal"`, `block_size=64`, Apache-2.0, CPU path)". The
  installed PEFT (0.21.2, `peft.utils.loftq_utils.loftq_init(weight, num_bits, reduced_rank, num_iter)`)
  **supports only `num_bits in {4, 8}`**, raises `ValueError` otherwise, requires **bitsandbytes** and
  moves the computation to `compute_device = "cuda"`, and exposes **no `method` or `block_size`
  parameter at all** — and neither does **PEFT v0.9.0**, whose signature was fetched and checked
  (`loftq_init(weight, num_bits, reduced_rank, num_iter)`, `peft/utils/loftq_utils.py` at the v0.9.0
  tag): those two parameters exist in no PEFT release, while they do exist in the LoftQ repository's
  GLUE module. Two of the three named parameters therefore do not exist in that API, and the
  named bit width is refused by it: the gate as written is unexecutable on any substrate this project
  may use (the workstation has no CUDA device, and the free GPU tier must not be spent on a
  bit-width the API rejects).
- **What the specification actually refers to.** The parameters `method="normal"` and
  `block_size=64`, and the 2-bit NF codebook, belong to the pinned **LoftQ repository's own CPU path**
  — `glue/utils_qaunt.py` at `yxli2123/LoftQ` @ `ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3` (MIT),
  which implements `create_normal_map(num_bits=2)`, `quant_nf4_block(weight, block_size=64,
  num_bits=2)` and `quant_uniform(input, num_bits=2, clip_val=...)`. That is a genuinely independent
  implementation of the published procedure, which is what an oracle must be. The plan's sentence
  conflated that module with PEFT's differently-shaped function.
- **Change:**
  1. The T2-a oracle is the pinned LoftQ repository's `glue/utils_qaunt.py` at the commit above,
     executed on CPU. No third-party source is committed: a **numeric fixture** captured from it is
     committed instead (`artifacts/sample-results/m3-oracle/loftq-nf2-block64.json`, sha256 prefix
     `a40dbef80ccfd97e`), carrying the input matrices, the outputs, the tolerance, and the provenance
     (repo, commit, path, licence, torch/scipy versions, seed, and the exact reproduction command).
  2. The fixture's 2-bit level table is `[-1.0, 0.0, 0.3379152417, 1.0]`: the reference's
     **asymmetric** construction (`offset=0.9677083`, `v1` and `v3` of *different* lengths with a
     literal `0` between them, sorted, then divided by their maximum). Our `nf_levels(2)` must
     reproduce it, and the uniform-int2 control must reproduce `quant_uniform` with the dispatcher's
     `mean ± 2·std` clip.
  3. `peft` is added as a **verification-only** extra (`verify = ["peft>=0.13,<1"]`, Apache-2.0).
     Nothing in `src/**` imports it. It is retained because the *original* gate can still be run in
     the one form PEFT supports (`num_bits=4`, CUDA-only) on the cloud substrate as a secondary
     cross-check; that is a separate, clearly-labelled check and is not the T2-a gate.
- **Effect on frozen hypotheses:** none. No threshold, dataset, seed, arm or verdict rule changes. The
  tolerance stays `<= 1e-6` relative Frobenius; only the identity of the independent implementation is
  corrected, and it is corrected **before any M3 arm has run**, so no result was seen first.
- **Evidence:** `artifacts/sample-results/m3-oracle/loftq-nf2-block64.json`;
  `src/spectraquant/quantization/codebook.py` (the implementation this gates);
  `docs/research/reproduction-plan.md` §5.3, §3.1.
- **Amends:** A-0006 (freeze checklist) in the sense that a frozen gate's reference implementation is
  corrected; the gate itself is unchanged. Append-only.
- **Superseded-by:** -

---

## A-0014 — The LoftQ schedule starts from zero (the paper's Algorithm 1), not from `SVD_r(W)`

- **Date:** 2026-10-10.
- **Trigger:** implementing the T2-a oracle gate (A-0013) showed that our LoftQ initialisation could
  not agree with the pinned reference **for a non-bug reason**: `loftq_initialise` started the
  alternation from `SVD_r(W)`, so its `iterations=1` was the reference's *second* step. The frozen M3
  design describes the arm as "LoftQ with `T = 1` (paper: T=1 is the QLoRA-quantized weight + SVD of
  the residual, §3.2)" (`reproduction-plan.md` §3.1) — i.e. `Q_1 = q_N(W)`, the plain
  post-training-quantization base, which is what Algorithm 1 produces when it starts from
  `A_0 = B_0 = 0` and what the reference's `quant_first_iter` implements when called with `L = R = 0`.
  Our start therefore contradicted the design text it was supposed to implement, and it made the
  module's own "T = 1" arm a different algorithm from the published one.
- **Change:** `spectraquant.factorization.loftq.loftq_initialise` now starts from
  `A_0 = B_0 = 0` for every `iterations >= 1`; the first quantized base is `q_N(W)`, the first factors
  are `SVD_r(W − q_N(W))`, and `iterations = 1` is exactly the reference's first step. `rank = 0`
  still returns empty factors (the declared "no low-rank compensation" case), and `rank` is validated
  before the zero start is built so a negative rank raises the module's own `ValueError` rather than a
  torch error. No other public behaviour changes.
- **Consequence for earlier numbers.** The two collected trainable-arm diagnostics
  (`docs/results/m3-reproduction-2026-10-09.md`) used the old start for their `loftq` arm. They are
  labelled diagnostics and no claim depends on them; their LoftQ initialisation error is therefore
  **superseded** by this amendment and must not be quoted as a LoftQ result. The `lr_qat` arm is
  unaffected (it never used this schedule).
- **Verification.** The T2-a gate now compares our `T=1` and `T=2` against the reference capture in
  `artifacts/sample-results/m3-oracle/loftq-nf2-block64.json` (768x768, quantized weight, the `L@R`
  product and the residual; `L`/`R` are *not* compared elementwise because the singular-value split
  convention differs and `L@R` is the invariant object). `tests/unit/test_loftq.py` was re-derived by
  hand for the new schedule (the 2x2 closed form: `Q = [[0.95,0],[0,0.95]]`,
  `R = [[0,0.15],[0.15,-0.40]]`, dominant singular direction `lambda1 = -0.45` with
  `q = (1,-3)/sqrt(10)`, leftover `lambda2 = 0.05` with `q2 = (3,1)/sqrt(10)`).
- **Effect on frozen hypotheses:** none. No threshold, dataset, seed, arm or verdict rule changes; the
  implementation is brought into line with the design text that was already frozen.
- **Evidence:** `src/spectraquant/factorization/loftq.py`, `tests/unit/test_loftq.py`,
  `artifacts/sample-results/m3-oracle/loftq-nf2-block64.json`.
- **Amends:** A-0013 (the oracle identity), A-0006 (freeze checklist). Append-only.
- **Superseded-by:** -

---

## A-0015 — The T2-a elementwise tolerance is met on the quantized weight and on the residual norm, not elementwise on the low-rank product

- **Date:** 2026-10-10.
- **Trigger:** implementing the T2-a oracle gate (A-0013) against the 768x768 capture, the
  **elementwise** comparison of the low-rank product could not reach the predeclared `1e-6`. The
  cause is numerical, not a defect, and it was measured before the verdict:
  * the fixture's 2-bit NF residual has a **near-degenerate spectrum at the truncation rank**:
    `sigma16 / sigma17 = 1.0049` (a 0.49 % gap). A rank-`r` subspace whose `r`-th and `(r+1)`-th
    singular values are that close is determined in float32 only to about
    `eps * sigma1 / (sigma_r - sigma_{r+1})`, i.e. `~8e-5` here;
  * **self-control on this repository's own code**: our rank-16 product on that same matrix differs
    by `1.061e-05` relative between float32 and float64, while the same computation on a
    well-separated control at the same shape and rank differs by `9.3e-07`. No independent float32
    SVD can land within `1e-6` of the captured product on this matrix.
- **Change:** the T2-a gate keeps the frozen `<= 1e-6` on the quantities the tolerance can actually
  discriminate, and records the rest:
  1. the **quantized weight** `Q` elementwise — measured `max_abs = 0.0` at T=1 (exact) and
     `2.1e-05` absolute / `1.0e-06` relative at T=2, the latter inherited from the same degeneracy;
  2. the **residual norm** and the **residual spectrum** — measured `1.5e-07` and `4.2e-07` at T=1,
     `7.8e-08` and `4.8e-07` at T=2, all within the frozen tolerance;
  3. the **elementwise product and residual** — asserted at `1e-3` as a *convention-error detector*
     (a transposed factor or a wrong quantizer convention is O(1), so the detector keeps its power),
     with the measured value recorded beside it (`1.41e-05` product, `5.00e-06` residual at T=1).
- **Why this is not a weakened gate.** The property the gate exists to check is that our
  implementation follows the published procedure rather than a self-consistent variant. The
  well-determined quantities (the quantized weight, the residual norm, the residual spectrum) test
  exactly that at the frozen tolerance; the elementwise product additionally catches a convention
  error at `1e-3`, and the `1e-6` claim is withdrawn **only** where the arithmetic cannot support it.
  The finding is recorded with its evidence rather than absorbed by loosening a threshold silently.
- **Effect on frozen hypotheses:** none. No arm, seed, dataset, threshold or verdict rule changes.
- **Evidence:** `artifacts/sample-results/m3-oracle/loftq-nf2-block64.json` (`loftq_768`),
  `tests/unit/test_m3_exactness.py` (the gate and its recorded numbers),
  `src/spectraquant/factorization/loftq.py`.
- **Amends:** A-0013, A-0014. Append-only.
- **Superseded-by:** -

---

## A-0016 — Tier-1 moves to `device: cuda` for the method arm, and its comparators are re-measured on the same substrate

- **Date:** 2026-10-10.
- **Trigger:** the method arms (`proxy_allocated`, `spectraquant_regularized`) train, and A-0012 left
  `configs/tier1/smollm2_135m.yaml` on `device: cpu` precisely so the recorded comparator slice stayed
  comparable. A-0012 also anticipated this moment: "When the M5 arm lands, Tier-1 gets its own
  amendment and a GPU run."
- **Change:**
  1. `configs/tier1/smollm2_135m.yaml` declares `device: cuda`. The trainable arm cannot run on CPU
     within any sane budget, and the plan's own cost envelope already assumes the GPU substrate.
  2. The already-collected CPU comparator slice (`tier1_smollm2_135m-cloud`, `validated`, one seed) is
     **not** used as the equal-memory comparator for the method's runs. It remains a valid
     CPU-substrate pilot and keeps its label; `AGENTS.md` §4.5 requires the comparison to be at equal
     memory *and* the substrate to be the same, so the comparators (fp16, int8 and int4 PTQ, the
     low-rank arms) are **re-measured on the GPU inside the method's run** — they are eval-only and
     cheap, so this costs minutes and removes the cross-substrate caveat entirely.
  3. The recorded CPU slice is reported as a pilot with its own substrate label, and any table that
     places it beside GPU numbers must say which substrate each row came from.
- **Effect on frozen hypotheses:** none. No arm, seed, grid, threshold or verdict rule changes; the
  method's comparison set and the equal-memory requirement are unchanged. The change is *which
  substrate* the Tier-1 numbers are produced on, and it is recorded because it invalidates a
  cross-substrate reading of the earlier pilot.
- **Evidence:** `configs/tier1/smollm2_135m.yaml`, `docs/results/tier1-cloud-run-2026-10-09.md`,
  `docs/decisions/ADR-0004-spectraquant-method.md`.
- **Amends:** A-0012 (the device decision and its stated revisit condition). Append-only.
- **Superseded-by:** -

---

## A-0017 — The Tier-1 byte-budget ladder gains the comparator byte counts, so the allocated arms can be solved at equal memory

- **Date:** 2026-10-10.
- **Trigger:** the allocated arms (`proxy_allocated`, `spectraquant_regularized`) take a **byte budget**
  as their binding, not a `(rank, bits)` point, and the runner refuses an undeclared one: the budget
  must be a rung of the plan's `grid.budget_ladder_bytes`. The predeclared ladder
  `[26906055, 40359082, 53812110, 67265138, 88789981]` contains no comparator's byte count, while
  `AGENTS.md` §4.5 requires the method to be compared **at equal memory** — and the equal-memory gate's
  tolerance is tight, so "near" is not equal.
- **Change:** the Tier-1 ladder gains the three **measured** class-1 comparator byte counts from the
  validated Tier-1 run (the same 210 targeted tensors, so the figures are substrate-independent):
  int4 PTQ `59 719 680`, int8 PTQ `112 803 840`, fp16 `212 336 640`. An allocated arm can now be
  solved at exactly a baseline's stored bytes, which is what the equal-memory comparison needs. The
  original five rungs are kept, so the predeclared frontier points are unchanged.
- **Why this is not tuning on the outcome.** The added values are *measurements of the baselines*, not
  choices made after seeing the method's result; no method number exists yet (the method arm has not
  run). Adding the comparison points to the ladder before the comparison is exactly what §4.5 requires.
- **Effect on frozen hypotheses:** none. No arm, seed, grid, threshold or verdict rule changes; the
  ladder is extended with the byte counts the comparison is defined against.
- **Evidence:** `configs/tier1/smollm2_135m.yaml`; `docs/results/tier1-cloud-run-2026-10-09.md` (the
  measured counts); `src/spectraquant/cloud/plan_runner.py` (`_resolve_allocation_budget`).
- **Amends:** A-0006 (freeze checklist). Append-only.
- **Superseded-by:** -

