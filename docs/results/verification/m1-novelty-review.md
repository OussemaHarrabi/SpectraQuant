# M1 adversarial review — novelty verdicts, proxy soundness, statistics, testability, equal-memory

**Reviewer:** `AdversarialReview` (Agent L, adversarial verifier). **Role:** read-only with respect to
every existing file; the only write target is `docs/results/verification/**`
(`docs/coordination/ownership.md` line 52).
**Date (UTC):** 2026-10-08. **Branch:** `infra/bootstrap`. **Review started at** HEAD
`f77fc9a5199dce53dbaa9347e3da48918715c5aa`; **the tree advanced to** `98d1f60` (`docs: record the M0
gate result on a clean checkout`) during the review. All attacked documents are **tracked and clean**
at the end of the review (no `M` flag), and the drift check in
`raw/quote_ledger.txt` / `raw/git_status.txt` confirms the live files are byte-identical to the
snapshot, so every quote below refers to committed content.
**Purpose:** attack the M1 scientific foundation *before* `preregistration.md` is frozen, and return
the list of claims that must be weakened or removed first.

---

## 0. Revision anchors, method, and a warning about the moving tree

The tree is a **shared working tree under active edit** (`docs/coordination/status.md` §8), and it
changed *during* this review: `preregistration.md` was rewritten twice (cloud substrate, amendment
A-0004), `eval-protocol.md` twice, `AGENTS.md` once, and amendments A-0003/A-0004 appeared after my
first read. Every quote below is therefore anchored to a **frozen snapshot** copied into
`docs/results/verification/snapshot/` at the time of review, with hashes in
`docs/results/verification/raw/snapshot_hashes.txt`. Quotes were re-extracted verbatim from that
snapshot into `docs/results/verification/raw/quote_ledger.txt` after the last edit.

```
$ sha256sum docs/results/verification/snapshot/*
7feee8c998cee5469e0e4334164ab6efa2cd44a6ff23e938d86c00d7f6afa1ca  AGENTS.md
db3f836a36f38c59d16ed149956932ca0ccb58cfbb8012642c895617905823a1  charter.md
4724ede7c414eaccead497d3ce75105b8aa841a1cb4fddeb8dff57fcac6cf4e5  design-m2-interfaces.md
1177c7d8a8da96d663a9400034902c720364e0aa7286f698338fa6825dcafae7  eval-protocol.md
56a4cf4786a7452982c84bc871117a13c0be642098d41c0fa2367dae65abf334  literature-matrix.csv
21262f2d932f8bc8bd4cbe8d1acc3d4de8816937b2684bb9fbe8dae690e5f472  literature-review.md
978aed608eff1ce838368639ad355fac124884323db00434ab4715c876a11278  measurement-taxonomy.md
df07b13263dd0d2c1a2750836620ecd4d931da0fe386fede4ce45e2ca2d0fcc0  memory-accounting.md
9665f88c123460e94caab60a6a2979f27497904f34fea10a7677b65f7ea745e6  novelty-risk.md
fec60176fa1ff26764f7eeb1c93af0419fcfdd1acac52c0ad7accee2f4860fd1  ownership.md
0b74a8c43be875f2dcb5d0abe67a6fc8174eb4945ced3aa13627ff7622c89c85  preregistration-amendments.md
025273949b583396783d473b5bac05faae67e480372f267ccf46a40d5d7b3a23  preregistration.md
dd618feb7ec162135a440b1b9f830f2408fde49d8157d8a928c3fbffd766dbd3  status.md
```

Amendments present at review time: **A-0001, A-0002, A-0003, A-0004** (`grep -n "^## A-"`).
**If any hash above differs from the live file, re-run `quote_ledger.txt` generation before acting on
a quote** — a finding may already be fixed.

**Method.** (i) fetch every cited arXiv abstract page and diff the *claims* against the real text;
(ii) build adversarial numerical probes of the proxy and the statistical plan (4 scripts, all under
`docs/results/verification/`, no repo module imported); (iii) audit the documents for executable
mechanisms, not prose; (iv) re-read every quoted sentence after the tree stopped moving.

**Honesty note on probe scope.** Probes E3/E4 use a from-scratch 6-block, `d=192` decoder-only
transformer trained on a synthetic peaked-Markov corpus, and E4 measures this workstation's CPU. They
establish *mechanism* (does a layerwise quantity compose into downstream damage?) and *design power*
(what can the pre-registered test detect?). They are **not** measurements of the Tier-1 model, which
does not exist yet. Every claim below is labelled accordingly.

---

## 1. Citation audit — 17 cited works fetched, 2 uncited obvious neighbours found

(One further candidate was fetched and **rejected**: arXiv:2101.09671, which I had suspected to be
HAWQ-V3, is actually *"Pruning and Quantization for Deep Neural Network Acceleration: A Survey"* —
not a neighbour of this work, and correctly absent from the matrix.)

**Fetch** (all HTTP 200, raw HTML kept): `raw/abs_<id>.html`; **extraction**:
`uv run python docs/results/verification/check_citations.py > raw/citation_check.out`.

```
$ uv run python docs/results/verification/check_citations.py   # 17 cited + 2 uncited, 171 lines
```

| Tag | arXiv | fetched title / date | matrix year | matrix venue | venue on the fetched page | claim check |
|---|---|---|---|---|---|---|
| P1 | 2609.33923 | *Quantization Error Is Spectrally Flat…* / 2026-09-27 | 2026 ✓ | arXiv preprint (Black Sheep Ai) | (none) | **all claims verified** |
| P2 | 2210.17323 | *GPTQ…* / 2022-10-31 | 2022 ✓ | arXiv preprint | **ICLR 2023** | venue understated |
| P3 | 1911.03852 | *HAWQ-V2…* / 2019-11-10 | 2019 ✓ | arXiv preprint | (none) | **mischaracterised** |
| P4a | 2402.14866 | *APTQ…* / 2024-02-21 | 2024 ✓ | arXiv preprint | **DAC 2024** | venue understated |
| P4b | 2607.07964 | *KronQ…* / 2026-07-08 | 2026 ✓ | COLM 2026 | COLM 2026 ✓ | ✓ |
| P5a | 2607.23047 | *MixQuant…* / 2026-07-25 | 2026 ✓ | arXiv preprint (Preprint) | Preprint ✓ | ✓ |
| P5b | 2509.15455 | *CoopQ…* / 2025-09-18 | 2025 ✓ | arXiv preprint | (none) | ✓ |
| P6 | 2311.12023 | *LQ-LoRA…* / 2023-11-20 | 2023 ✓ | arXiv preprint | (none) | **misattributed (understated)** |
| R1 | 2602.02001 | *Preserve-Then-Quantize (SRR)* / 2026-02-02 | 2026 ✓ | ICML 2026 | ICML 2026 ✓ | ✓ |
| R2 | 2505.08022 | *Dynamical Low-Rank Compression…* / 2025-05-12 | 2025 ✓ | arXiv preprint | (none) | ✓ |
| R3 | 1802.05957 | *Spectral Normalization…* / 2018-02-16 | 2018 ✓ | arXiv preprint | **ICLR 2018** | venue understated |
| R4 | 2406.06623 | *Spectrum…* / 2024-06-07 | 2024 ✓ | arXiv preprint | (none) | **mischaracterised** |
| R5a | 2312.05821 | *ASVD…* / 2023-12-10 | 2023 ✓ | arXiv preprint | (none) | ✓ |
| R5b | 2403.07378 | *SVD-LLM…* / 2024-03-12 | 2024 ✓ | arXiv preprint | **ICLR 2025** | venue understated |
| C1 | 2507.09616 | *MLoRQ…* / 2025-07-13 | 2025 ✓ | arXiv preprint | (none) | ✓ |
| C3 | 2602.22268 | *AutoQRA…* / 2026-02-25 | 2026 ✓ | arXiv preprint | 15 pages ✓ | ✓ |
| C4 | 2609.24298 | *KV-COBRA…* / 2026-09-21 | 2026 ✓ | arXiv preprint | (none) | ✓ |
| **uncited** | 2411.05007 | *SVDQuant…* / 2024-11-07 | — | **absent from the 50-row matrix** | ICLR 2025 Spotlight | **obvious neighbour** |
| **uncited** | 2607.12550 | *A JoLT for the KV cache: … Joint Rank-bit Allocation* / 2026-07-14 | — | **absent** | under review ICLR 2027 | **falsifies a §0 coverage claim** |

```
$ grep -rin "svdquant\|hawq-v3\|2101.09671\|jolt\|2411.05007\|2607.12550" docs/ references/   → no matches
```

**No wrong year and no wrong first author** was found among the 17 checked works; every arXiv ID
resolves to the stated title. The defects are in *what the neighbour is said to do*.

### 1.1 Finding C1 — P6/C6 (LQ-LoRA) is misattributed; the strongest neighbour is understated

Attacked claim (`novelty-risk.md` line 45, P6 row): *"Ranks allocated across layers under a
**parameter budget** using a reconstruction objective."* / *"Rank-only; reconstruction (not
output-directed) cost; **fixed bit width**."*
and (`novelty-risk.md` line 116, C6 row): *"Rank allocation under a budget (rank-only)."* / *"**No bit
dimension**."*
and (`novelty-risk.md` line 51): *"P6's allocation is bit-agnostic"*.

Real abstract (fetched, `raw/citation_check.out`):

> "We present an integer linear programming formulation of the quantization component which enables
> **dynamic configuration of quantization parameters (e.g., bit-width, block size) for each matrix
> given an overall target memory budget**. We further explore a **data-aware version of the algorithm
> which uses an approximation of the Fisher information matrix** to weight the reconstruction
> objective…"

So LQ-LoRA (a) configures **bit-width** per matrix under **one overall memory budget**, (b) does so
jointly with a rank-allocated low-rank component, and (c) has an **activation/Fisher-weighted**
(data-aware) objective. "Fixed bit width", "no bit dimension" and "bit-agnostic" are each
contradicted by the source. This removes the specific gap the A-difference statement rests on
(`novelty-risk.md` line 49–53: *"P1's probe is rank-agnostic; P6's allocation is bit-agnostic"*).
The same error is repeated in `literature-review.md` §2.2 (*"allocation is over *rank* given a fixed
bit width … no joint rank × bit search"*) and §7.3 (*"LQ-LoRA for rank-only"*).

### 1.2 Finding C2 — P3 (HAWQ-V2) is mischaracterised in the *helpful* direction

Attacked claim (`novelty-risk.md` line 42): *"Hessian-**trace** sensitivity as an allocation proxy
under an **average-bit budget** …"* / *"**Weight-space trace**, no activation propagation, no rank."*

Real abstract: HAWQ-V2's stated contribution is that *"a better sensitivity metric is to compute the
**average of all of the Hessian eigenvalues**"* and *"a **Pareto frontier** based method for selecting
the **exact bit precision** of different layers **without any manual selection**"*, plus *"we extend
the Hessian analysis to **mixed-precision activation quantization**"*. The average-bit-width budget is
HAWQ-**V1**'s framing (the abstract lists it as one of V1's three limitations that V2 removes), and
the layer Hessian in HAWQ/Hessian-trace methods is built from **layer input statistics** — calling it
"weight-space" understates the overlap with an activation-aware proxy. `literature-review.md` §5.1
(*"Hessian-trace is a *weight-space* proxy"*) repeats it.

### 1.3 Finding C3 — R4 (Spectrum) is mischaracterised

Attacked claim (`novelty-risk.md` line 80): *"SNR-targeted training **reweighting**."*
Real abstract: *"…by **selectively targeting layer modules** based on their signal-to-noise ratio
(SNR), and **freezing the remaining modules**."* Spectrum selects which modules to train; it does not
reweight. It is also neither spectral nor a regularizer, so its presence in the *spectral regularizer*
neighbour list inflates the list without covering it.

### 1.4 Finding C4 — four venue understatements while claiming venue verification from the abs page

`literature-matrix.csv` marks `venue_verified = "yes (arXiv abs page)"` for GPTQ, APTQ, Spectral
Normalization and SVD-LLM while recording `venue = "arXiv preprint"` — yet those same abs pages state
ICLR 2023 / DAC 2024 / ICLR 2018 / ICLR 2025 in their comments field (table above). The verification
claim is falsified by the source it cites. (The matrix *can* read the comments field — SRR and KronQ
are recorded as ICML 2026 / COLM 2026 from it.)

### 1.5 Finding C5 — an uncited obvious neighbour: SVDQuant (arXiv:2411.05007, ICLR 2025 Spotlight)

Not present anywhere in `docs/` or `references/` (grep above). Its abstract:

> "…we use a **high-precision, low-rank branch** to take in the weight outliers with Singular Value
> Decomposition (SVD), while a **low-bit quantized branch handles the residuals**. … we **co-design an
> inference engine Nunchaku that fuses the kernels of the low-rank branch into those of the low-bit
> branch**…"

That is: a designated rank budget absorbing what quantization cannot represent, a quantized residual
branch, and a kernel-level answer to "does the low-rank branch survive contact with real low-bit
execution" — i.e. the same structural idea as contributions A and B and the same *mechanism* as the
H5 kernel-survival question, in the diffusion domain. Its omission is material for both A and B.

### 1.6 Finding C6 — an uncited neighbour that falsifies a §0 coverage claim

Attacked claim (`novelty-risk.md` line 22): *"Search coverage on 2026-10-08 … arXiv full-text/title
queries via the arXiv API and `arxiv.org/abs` pages for the terms … *bit-rank allocation* …"*.
arXiv:2607.12550 (2026-07-14) is titled *"A JoLT for the KV cache: Near-Lossless KV Cache Compression
via **Joint Rank-bit Allocation**"* and is not in the matrix. Its abstract states the identical gap
framing: *"Existing compression methods apply low-rank factorization or quantization independently,
without **jointly allocating rank and precision under a shared storage budget**… a single Lagrangian
dual allocates per-group Tucker ranks and residual bit-widths under a global byte constraint."* A
term-exact hit was missed, so the coverage claim as written is false; the honest version is "we
searched these terms and cannot claim coverage of the bit-rank-allocation term".

### 1.7 What the citations *did* support (see also §8c)

P1/RAM is quoted accurately and completely: unbiased probe, spectral flatness, *"the propagated probe
rank-correlates 0.81 to 0.83 with the GPTQ layer objective from real activations, while the isolated
estimator is uncorrelated with it"*, *"A knapsack solver allocates bits under an exact byte budget"*,
*"3.5 to 13.6% lower median WikiText-2 perplexity than size-comparable uniform 4-bit builds"*. R1/SRR,
C1/MLoRQ, C3/AutoQRA, C4/KV-COBRA, P5a/MixQuant, P5b/CoopQ, R2 and R5 also match their abstracts.

---

## 2. Novelty verdicts A / B / C

| Contribution | Stated verdict | This review's verdict | Basis |
|---|---|---|---|
| **A** — $(r,b)$ output-aware cost proxy | differentiated (weakly) | **overclaimed** | The difference statement is built on P6 being bit-agnostic; P6 is not (§1.1). With P6 removed as a "one-axis" neighbour, the $(r,b)$ currency is *already occupied by a rank+bit allocation under one memory budget with a data-aware (Fisher) objective*. A's remaining difference is the **estimator form** (per-layer output-error proxy vs an ILP on a reconstruction objective) and the evaluation protocol — narrower than stated, and now competing with an uncited neighbour (§1.5). |
| **B** — rounding-aware spectral regularizer | differentiated (moderate) | **overclaimed (mild)** | The conjunction (training-time + spectral + rounding-grid) is still not contradicted by anything I fetched; but the verdict's strength rests on the completeness of a search that missed SVDQuant (§1.5), whose low-rank branch is explicitly *designed around* quantization's representational limits. Keep the claim, drop "moderate" until SVDQuant/SRR are cited and distinguished in §2. |
| **C** — mixed rank/bit allocation under a stored-bytes budget | previously-known | **sound** | C1/C3/C4 + uncited JoLT all do it; the document already refuses to claim it. Conservative and correct. |

The *falsifier* structure of all three is honest and two-sided (each has a stated result that would
downgrade it) — that part of `novelty-risk.md` is good practice and I could not break it.

---

## 3. Proxy soundness — does a layerwise output-error proxy predict downstream damage?

### 3.1 The attacked claim, and what it is as mathematics

Attacked claim (`design-m2-interfaces.md` line 15): *"Proxy cost is the per-layer squared Frobenius
norm of the *layer output* error `E_X ||X W^T − X (Q(B) Q(A))^T||²`, **aggregated across layers by
summation** unless a slice documents otherwise."*

For a fixed compression operator, `E_X||X W^T − X Ŵ^T||² = tr(Δ Σ_X Δ^T)` with `Δ = W − Ŵ`. Two
properties follow immediately and are the source of every problem below:

1. It is a **local linearisation at the uncompressed activation distribution**: it is exactly the
   squared norm of the *first-order* output perturbation, and it assumes the layer's input `X` is the
   uncompressed one.
2. **Summation is not the correct composition.** For a residual stack, `e_{ℓ+1} = J_ℓ e_ℓ + δ_ℓ`, so
   the final error is `Σ_ℓ (Π_{j>ℓ} J_j) δ_ℓ`: every layer's error is multiplied by the **downstream
   gain product**, which the summed proxy sets to 1. LayerNorm/RMSNorm sit *between* the layer whose
   error is scored and the output, and they rescale the residual stream — so a proxy computed on the
   pre-norm activation is not the quantity that propagates.

### 3.2 Measured: with LayerNorm the layerwise proxy loses its ranking ability

Probe E3 (`uv run python docs/results/verification/probe_compose.py`, output
`raw/probe_compose.out`). Compress exactly one linear layer at a time (rank-then-int4, `r=32`,
`b=4`, 24 linear layers), measure the end-to-end final-hidden damage, and correlate with that layer's
proxy:

```
              config  rho(proxy,damage)  rho(in-situ,damage)  joint/sum_single joint/sum_proxy  gain spread
   trained-LN-r32-b4             0.1191               0.0209            1.1916          0.0001         45.6x
 trained-noLN-r32-b4             0.8130               0.6861            0.7404          0.0485         56.2x
   trained-LN-r64-b4             0.7348               0.7043            1.1643          0.0003         32.9x
   trained-LN-r32-b8             0.1374               0.0557            1.1229          0.0001         52.4x
 untrained-LN-r32-b4             0.4061               0.3870            0.5639          0.0277        109.7x
```

With LayerNorm present (as in every real transformer) and a trained model, `Spearman(proxy, downstream
damage) = 0.119` at 4 bits and `0.137` at 8 bits; the same measurement with LayerNorm replaced by
Identity gives **0.813**. The probe contains softmax attention in both variants, so the decoupling is
not attributable to attention — the LayerNorm control isolates it. Damage is measured on both the
final normalised hidden state and the logits; both columns show the same pattern. **Caveat, stated
plainly:** the effect is configuration-dependent (the `r=64` run with LayerNorm gives 0.735), so the
honest finding is not "the proxy never works" but *"the LayerNorm control moves ρ by up to 0.7, so the
proxy's validity is a property of the (rank, bits, normalisation) configuration rather than of the
model"* — which is precisely what an H2 test at a single pinned configuration would not reveal.
**Verdict: the aggregation claim is unsupported in the presence of normalisation layers**, which is
the only regime the project will ever run in.

### 3.3 Measured: the sum is not the joint damage, in either direction

Same probe, compressing *all* layers simultaneously:

* `joint damage / Σ(single-layer damage)` = **0.564, 0.740, 1.122, 1.164, 1.192** across the five
  configurations — sub-additive in some, super-additive in others. Non-additive in both directions
  means "aggregate by summation" cannot be rescued by a single global constant.
* `joint damage / Σ(layerwise proxies)` = **1.0e-4 … 4.9e-2** — three to four orders of magnitude off,
  and the offset itself varies by ~400× across configurations.

### 3.4 Measured: the same proxy value means 33–110× different damage depending on the layer

Per-layer `gain = downstream damage / proxy` spans **32.9×–109.7×** (table above). Concrete rows from
the untrained-LN run: `b0.proj` proxy `0.919` → damage `1.291` (gain 1.40) while `b0.fc1` proxy
`178.16` → damage `6.138` (gain 0.034). A proxy that ranks `b0.fc1` 194× above `b0.proj` while their
downstream damage differs by 4.8× carries almost no ranking information about *damage* — consistent
with the ρ=0.12 measured with LayerNorm.

### 3.5 Calibration estimator: bias and variance

Probe E2 (`uv run python docs/results/verification/probe_estimator.py`, output
`raw/probe_estimator.out`). For fixed `Δ`, the plug-in estimator `T̂_n = (1/n)Σ_i ||x_iᵀΔ||²` is
exactly unbiased for `tr(ΔΣ_XΔᵀ)` and, for `x ~ N(0,Σ)`, has the **exact** relative standard error
`sqrt(2·tr(G̃²)/n)/tr(G̃)` with `G̃ = LᵀΔΔᵀL`:

```
                regime       n        truth  exact rel SE  sim rel SE  sim rel bias
              gaussian      32        57.82        0.0271      0.0268      -0.00071
              gaussian     128        57.82        0.0136      0.0135      -0.00053
              gaussian    8192        57.82        0.0017      0.0017      -0.00007
heavy-tailed (LLM-like)      32       698.23        0.0637      0.0630      -0.00011
heavy-tailed (LLM-like)    8192       698.23        0.0040      0.0042       0.00003
   exact rel SE at n=128 / 524288 (the predeclared calibration): 0.0136 / 2.12e-04 (gaussian),
                                                                 0.0319 / 4.98e-04 (heavy-tailed)
```

Heavy-tailed (LLM-like) activations inflate the relative SE by ~2.3× but it is still ≈5e-4 at the
calibration size the project has predeclared (`eval-protocol.md` §3.2: 256 × 2048 = **524,288
tokens**). **So variance is not the problem, and the plan's `S6` "sampling information so variance is
reportable" is the wrong worry.** I also tested the obvious in-sample concern — fitting the operator's
per-channel scales on the same activations used to score it (AWQ-style) — and in this construction it
is negligible (`underestimate factor` held-out/in-sample = **1.0015 / 1.0000 / 0.9997** at
n = 128/512/2048; `raw/probe_estimator.out` §E2b). The estimator's error is therefore not sampling
error at all; it is **target error**, from two sources:

1. **Activation-distribution shift under joint compression.** Measured in E3: the summed proxy moves
   by ×0.972–×0.998 (small), but *per layer* the shift is systematic by layer type — in the trained
   runs the attention-projection and MLP-output layers shift by **−0.8 % to −17.5 %** while the
   normalised-residual layers (`qkv`, `fc1`) shift by ≤1 %; in the untrained run the same layers shift
   by up to **−86 %**. The ranking of proxies survives (ρ=0.98 fp vs in-situ) but the ranking against
   *damage* does not (0.735→0.704, 0.813→0.686, 0.406→0.387).
2. **Non-composition** (§3.3).

**Verdict on the estimator: sound, and tight, as an estimator of its own stated target; unsupported as
a proxy for downstream perplexity.** The document's `exact: bool` + sampling metadata
(`design-m2-interfaces.md` invariant 4) is not sufficient — it must also carry the *activation source*
(uncompressed vs in-situ) and the *aggregation* used.

### 3.6 The pre-registered correlation analysis has a sample-size problem that compute cannot fix

Attacked claim (`preregistration.md` line 152): *"Spearman ρ and Pearson r between proxy scores and
measured per-layer (and end-to-end) degradation, with **bootstrap CIs** (**per-layer resampling**) and
Fisher-z intervals."*

The resampling unit is the **layer**, and the number of layers is fixed by a Tier-1 config that §4
describes only as *"small: a few layers, modest width"* and that §13 will pin at freeze. It cannot be
increased by buying compute. Probe E1 (`raw/probe_stats.out`) gives the design power of exactly this
test:

| L (layers) | Fisher CI width at ρ=0.6 / 0.8 / 0.9 | P(CI excludes 0) at ρ=0.6 / 0.8 / 0.9 |
|---|---|---|
| 6 | 1.260 / 1.016 / 0.788 | 0.204 / 0.438 / 0.648 |
| 8 | 1.046 / 0.800 / 0.586 | 0.321 / 0.634 / 0.838 |
| 12 | 0.835 / 0.593 / 0.394 | 0.500 / 0.866 / 0.979 |
| 24 | 0.560 / 0.361 / 0.217 | 0.855 / 0.998 / 1.000 |

and for the *actual* H2 comparison (proxy vs the predeclared Frobenius comparator, true Δρ = 0.17,
paired bootstrap over layers):

| L | 6 | 8 | 12 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|
| P(declared winner) | **0.058** | **0.036** | 0.082 | 0.121 | 0.182 | 0.446 |

A Tier-1 model with "a few layers" has ~4–6 % power to detect a real, meaningful advantage. Two
disclosures about the bootstrap in this probe: (i) I resampled layers with replacement in the standard
way but rejected draws with fewer than 4 distinct indices (without a rule for ties/duplicates the
statistic is undefined at small `n`), and (ii) the numbers are design power under a Gaussian/exp
model, not a statement about the real Tier-1 model. Both push the estimate *conservatively*: the
preregistration specifies no tie rule at all. At L=4 the per-layer bootstrap is **degenerate** under
any distinctness guard (width 0.000 in the table) — the procedure is not fully defined at the layer
counts the project will actually use.

**Is the layer the exchangeable unit at all?** No. Layers are heterogeneous strata: in E3 the
per-linear proxy values span `0.919` (proj) to `178.2` (fc1) at the same depth — a 194× difference
driven by layer *type*, not by any random draw. A bootstrap that resamples across such strata is
estimating the sampling distribution of a mixture, not the uncertainty of the correlation. Seeds
*are* exchangeable; layers are not. The plan currently uses seeds for §7.1 and layers for §7.2 with no
rule connecting them.

### 3.7 Definitional ambiguity: how many "layers"?

`§7.2` says "per-layer" but `§2 S1` says "per-layer post-compression output distortion". Neither
defines the granularity: a transformer block, or each of the 4 linear maps in it. That choice changes
n by 4× and changes the answer (§3.6 table). `design-m2-interfaces.md` §3 has `per_layer: dict[str,
float]` keyed by name, which suggests per-module, while `score_model(layers: Mapping[str, LayerInputs])`
suggests per-block. Unresolved.

---

## 4. Statistical plan (§7) — pre-registered, or merely described?

**Answer: merely described.** It is a well-written plan with several genuinely pre-committed choices
(the primary comparator, fixed-n, no optional stopping, negative results published). But five things
a confirmatory run must have are absent, and one is internally contradictory:

| # | Attacked sentence (verbatim) | Problem |
|---|---|---|
| R1 | §7.1 *"paired differences over matched **seeds**"* vs §7.2 *"bootstrap CIs (**per-layer resampling**)"* | Two different resampling units for the same study, no rule for combining them, and the layer unit is not exchangeable (§3.6). |
| R2 | §7.2 *"H2 is tested against the primary comparator with **Holm–Bonferroni**"* | If the family is "H2 vs the primary comparator", **m = 1** and Holm is the identity — the sentence is vacuous. If the family is {H1…H5}, m = 5. Both readings are supported by the text; the reader chooses after seeing data. |
| R3 | §7.4 *"Within each hypothesis family, **Holm–Bonferroni** across the predeclared tests."* | No family is ever enumerated, and no `m` is stated. The correction cannot be executed without a further choice — which is precisely what pre-registration exists to prevent. |
| R4 | §7.3 *"the byte saving at matched quality exceeds a **predeclared equivalence margin**"* | The margin has no value anywhere in the document or in the §13 freeze checklist. Neither does the tolerance for "matched quality". |
| R5 | §9.2 *"a **predeclared factor** over a **predeclared patience window**"*, §9.3 *"the **predeclared per-run cap**"*, §9.6 *"beyond a **predeclared tolerance**"* | §9 asserts these criteria are *"predeclared, automatic"*; none of the four constants exists. An "automatic" exclusion rule with undefined constants is neither automatic nor pre-registered. |
| R6 | §3.3 *"The number of test evaluations per arm is predeclared (≤ 2 …)"* vs §10.1 *"stop after the predeclared seed count and the predeclared **test reads**"* | "≤2" is a bound, not a number. Fixed-n requires the exact count. |
| R7 | §6 *"≥ **5 seeds** for cheap configurations; ≥ **3 seeds** where a full-factorial sweep is too expensive"* | "cheap" and "too expensive" are undefined; the reduction is therefore chosen after seeing cost, i.e. after seeing which arms are inconvenient. §8 row 7 nevertheless demands "≥5 seeds" unconditionally. |

---

## 5. Testability under the declared compute

### 5.1 The local CPU measurement, and what it says about the (now superseded) local Tier-1 tags

Probe E4 (`raw/probe_throughput.out`), measured on this workstation, 8 of 16 threads, `torch
2.14.1+cpu`, AdamW, seq 256–512:

| config | params | tok/s | h per epoch (WikiText-2 train, 2.4 M tok) |
|---|---|---|---|
| vocab 50257, d=128, L=4 | 13.7 M | 2 375 | 0.28 |
| vocab 50257, d=256, L=4 | 28.9 M | 1 879 | 0.35 |
| vocab 50257, d=256, L=6 | 30.5 M | 1 506 | 0.44 |
| vocab 50257, d=384, L=6 | 49.2 M | 1 158 | 0.58 |

The **previous** revision tagged Tier-1 training as `[LOCAL]` with ≥5 seeds and six arms. Arithmetic:
6 arms × 5 seeds × 5 epochs = **42 h** for the *smallest* config and **86 h** for the largest — for
one rank/bit setting, before the predeclared grid (`b∈{2,3,4,8}` × group `{32,64,128}` ×
{sym, asym} × `r∈{2,4,8,16,32}`) or the budget ladder multiply it. Optimizer state is not the binding
constraint (29 M params × AdamW 16 B/param ≈ 0.5 GB against 15.1 GiB), nor are activations — **wall
time is**. Amendment A-0004 moves Tier 1–5 to the cloud, which is consistent with this measurement.

**Verdict on the current §8:** the cells tagged `LOCAL-FIXTURE` / `LOCAL-CPU-MEASUREMENT` are RAM- and
time-feasible *except* row 8 (§5.2) — but the *cloud* plan that replaces them has its own unquantified
constraints (§5.5).

### 5.2 Row 8 cannot execute as written: a local **quality** re-measurement is not local work

Attacked claim (`preregistration.md` line 188): *"| **Packed-storage survival:** re-measure at parity
after our own int4/int8 serialization + CPU dequant | **H5** | 0/1 | `LOCAL-CPU-MEASUREMENT` | M4 |
**runnable** (class 2 → class 3) |"*

But the same document's §4 says: *"**Local work = repository management, CPU unit/property tests, tiny
synthetic fixtures, static analysis, config validation, notebook generation, result analysis,
figures/tables/reports, and the class 4-CPU kernel *measurement*.**"* Re-measuring **quality** after a
packed round-trip is none of those — it is evaluation. And `eval-protocol.md` L301 agrees:
*"**Consequence for the paper's scope.** No evaluation cell is stated to run on this workstation: the
only local cells are the pipeline fixture (never a result) and the class-4-CPU measurement"*. So row 8
is marked `runnable` locally while the substrate policy forbids the local evaluation it requires.
**Minimal fix:** split the row into (i) the class-3 **byte** measurement (local, valid) and (ii) the
class-2 **quality** re-measurement, re-tagged `CLOUD-COLAB`; or add an explicit, bounded exemption to
§4 ("class-2 re-evaluation of a ≤ Tier-0-size self-serialized artifact is local work") and add the
matching row to `eval-protocol.md` §6.4.

### 5.3 Row 9's "runnable" tag is unsupported by the frozen environment

Attacked claim (`preregistration.md` line 189): the CPU-kernel survival cell is *"**runnable** (class
4-CPU …)"*, requiring ONNX Runtime `MatMulNBits`/`MatMulInteger` or torchao intx.

```
$ grep -n "onnxruntime\|torchao\|onnx" pyproject.toml      → no match
$ uv run python -c "import onnxruntime"                    → ModuleNotFoundError
```

`pyproject.toml` declares extras `dev`, `track`, `cloud` — none of which contains `onnx`,
`onnxruntime` or `torchao`. The class-4-CPU evidence in `backend-capability.md` §1 was produced in a
**scratch venv outside the repository** (`$TEMP/sq-probe`). A cell that a frozen protocol calls
"runnable" must be runnable from the frozen environment (`AGENTS.md` §7: uv + committed `uv.lock`).
**Minimal fix:** add a pinned `onnx`/`onnxruntime`/`torchao` extra to `pyproject.toml` + `uv.lock`,
and a §13 freeze item for it.

### 5.4 The byte-identity rule and the H5 parity rule

`eval-protocol.md` §8 freezes the evaluation *configuration* (`eval_config_hash`), and §6.3's only
command template is *"`--model hf --model_args "pretrained=<HF_ID>,revision=<MODEL_SHA>,…"`"*. A
SpectraQuant-serialized int4 container has no HF revision and no harness task path, so the H5 local
cells have no route through §6.3/§8 as written — which is presumably why `eval-protocol.md` §6.4 now
calls the class-4-CPU measurement *"not an eval cell"* and delegates to `benchmark-protocol.md` §8.
That delegation is fine for the *kernel* cell, but row 8's quality re-measurement (§5.2) still needs
an evaluation config, and the protocol does not define one.

Worse, the quantity H5 compares at is not well defined:

Attacked claim (`preregistration.md` line 128): *"re-measure quality after our own packed
serialization (int4/int8) with CPU dequantization **at measured stored-byte parity**"*.
On the project's **own** fixture (`memory-accounting.md` §7.5), the same 1 280 B payload produces
**221–347 B** of container overhead depending only on how the ONNX export was invoked
(`algo_config` 347 B → 1 627 B total; `raw onnx.load` 310 B → 1 590 B; constructor kwargs 313 B;
producer strings cleared 319 B; orchestrator's fixture 221 B → 1 501 B; the document's own summary
says "221–363 B"), so *nominal 4.0 bits/param becomes **5.86–6.42 measured***. Two arms can therefore be "at parity" or not depending on an
unfrozen serializer call, and the difference between export invocations (≈126 B on a 1 280 B payload,
≈10 %) is larger than the difference between adjacent bit widths on small tensors. **Minimal fix:**
freeze the serializer invocation in the protocol, and predeclare a byte-parity tolerance
(e.g. `|Δbytes| / bytes ≤ 0.5 %`); see §6.

### 5.5 The cloud plan has no budget line, and free-tier quotas are assumed

`preregistration.md` §10.4(a): *"The **free-tier cloud substrate** (Colab / Kaggle) is assumed for
Tier 1–2, subject to session and quota limits; **runs are sized to fit within a session**."* and
(b) *"Tier 3–5 and any paid instance require explicit user authorization…"*.
No GPU-hour figure, session-length assumption or weekly-quota assumption is recorded. From the
project's own cost model (`eval-protocol.md` §6.4), the confirmatory S4 suite alone is **0.4–18.3
GPU-h per model per seed** (× ≥3 seeds) and the Tier-1 training arms are *"budgeted separately under
`AGENTS.md` §2b rule 8"* with no number. Scaling my measured CPU throughput by a conservative 20–50×
for a T4-class runtime gives ≈ 0.9–9 GPU-h per Tier-1 arm-seed at 5 epochs; × 6 arms × 5 seeds = **27–270
GPU-h**, versus a typical Kaggle weekly quota of 30 GPU-h. "Sized to fit within a session" is
therefore an assumption, not a demonstration. **Minimal fix:** add a `CLOUD-COLAB` budget line
(GPU-hour ceiling + quota assumption + the resulting maximum seed/arm count), and make §10.4(a)
say what happens when the ceiling binds.

---

## 6. Equal-memory enforcement — is there a mechanism, or only prose?

`AGENTS.md` §4.5 requires *"Comparisons are only valid at equal memory; every compression comparison
MUST also report an equal-memory counterpart."* I searched for anything that could *make an
unequal-memory comparison impossible*:

| Candidate mechanism | What it actually enforces |
|---|---|
| `preregistration.md` line 31 *"P1 and P2 are **only ever compared at equal P2**."* | Prose. No artefact, no check. |
| `design-m2-interfaces.md` invariant 1 (*"One byte-accounting source of truth"*) + M2 gate item 2 (`accounted_bytes == measure_serialized_bytes`) | That the **accounting is correct**, not that two arms' byte totals are equal. |
| `AllocationProblem.cost_fn` / `Allocation.accounted_bytes` (`design-m2-interfaces.md` §4) | The budget constraint is applied to the **class-1 analytical** quantity (`accounted_bytes(w_shape, spec)`, *"includes scales, zero points, group metadata, padding, alignment"*) — container overhead is **not** in it. The reported compression outcome P2 is the **class-3 measured** quantity (`measure_serialized_bytes`). Nothing forces the two to agree across arms. |
| `eval-protocol.md` §8 "byte-identity rule" | Explicitly about **evaluation configuration** (harness commit, dtype, seeds, `--limit`, substrate). It never mentions model bytes. |
| `risk-register.md` S2 mitigation | Prose. |
| `src/spectraquant/allocation/allocator.py` | `AllocationPlan` carries `budget_bytes` and `budget_feasible`; there is no field carrying measured class-3 bytes and no function comparing two plans. |
| `src/spectraquant/quantization/accounting.py` (untracked, created during this review) | Separates class-1 `accounted_bytes` from class-3 `serialized_size_bytes`/`serialize_state` **for one tensor**, and states that for our own container *"the measured byte count equals the accounted sum **exactly**"*. Still **no cross-arm equality check** — nothing compares two arms' totals, and the allocator's `cost_fn` is not required to use the measured path. |

**Verdict: equal-memory enforcement is prose-only.** Proposal — the check that should exist:

1. **Schema.** A comparison row MUST carry `compare.bytes_a`, `compare.bytes_b`, `compare.class`
   (must be `3`), `compare.rel_diff`, `compare.tolerance`, `compare.serializer_build` (a hash of the
   frozen serializer invocation, §5.4). Validator rejects a row whose two byte counts are not both
   class 3 or whose `rel_diff > tolerance`.
2. **Code.** `spectraquant.reporting` MUST NOT emit a quality-delta row without the paired byte delta;
   `assert_equal_memory(a: AllocationPlan, b: AllocationPlan, tol: float) -> None` raises
   `UnequalMemory` unless the **measured** class-3 totals match within `tol`. The allocator's
   `cost_fn` MUST include `overhead_fn` (container overhead measured once per frozen format at freeze)
   so the constraint and the metric are the same quantity.
3. **Test.** A CI test with a positive fixture (two arms inside tolerance) and a negative fixture (a
   1 % byte gap) that must fail — the negative fixture is the part that makes the rule real.
4. **Predeclared constant.** `tol` goes into `preregistration.md` §7.1/§13 with a value (§4 R4).

---

## 7. H5 and measurement classes after the 2026-10-08 amendments

Amendments **A-0003** (class 4 split; H5 extended to a CPU-kernel survival test) and **A-0004**
(execution substrate: local = orchestration/correctness + class 4-CPU; cloud = all Tier 1–5) both
exist and are referenced correctly.

**`preregistration.md` itself is now consistent.** I checked every class-bearing sentence against
AGENTS.md §5 / A-0003: §1 (*"H5 uses additional classes … class 4-CPU … class 4-GPU and class 5 are
'not measured' here"*), §4 (substrate tags), §5 (H5 arm), §8 (rows 8–9 `LOCAL-CPU-MEASUREMENT`,
row 14 `CLOUD-GPU`), §11.0 (traceability now cites *"§8 rows 8–9 … + §8 row 14"* — correct after the
rewrite), §11.1 (three-part falsifier), §12, §13 (class-4-CPU fixture checklist item). No stale
"class 4 is unavailable" text remains. **I could not break this.** (An earlier revision of §11.0
pointed at "§8 row 13"; it has since been corrected — evidence that the tree moved during review.)

**But the amendment's reach did not cover the protocol the preregistration executes through.**
Attacked sentence (`eval-protocol.md` line 35):

> "…every analytical cost figure in §9 is **`analytical` (class 1)** and is never presented as
> measured bytes or measured time. **Class 4–5 numbers are unavailable here (`measurement-taxonomy.md`
> §4) and are reported as "not measured".**"

Two lines later the same file says (`eval-protocol.md` line 49):

> "| `LOCAL-CPU-MEASUREMENT` | developer CPU workstation, a **class 4-CPU kernel on our own serialized
> artifact** | Yes, but only under `benchmark-protocol.md`; not an evaluation cell here |"

and the document it cites, `measurement-taxonomy.md` §4, lists **`4-CPU` … AVAILABLE**. So
`eval-protocol.md` §1 both denies and grants class 4-CPU availability in the same section, and its
citation to `measurement-taxonomy.md` §4 is false. **Minimal fix:** replace the sentence with
*"Class 4-GPU and class 5 numbers are unavailable here and are reported as 'not measured'; class
4-CPU is available for self-serialized artifacts under `benchmark-protocol.md` §8 (see the substrate
table above and §6.4)."*

Related, non-blocking bookkeeping: `docs/coordination/status.md` §2 and §6 still say *"Measurement
classes 4–5 unavailable"*; A-0001's own `Status:` field still reads `accepted` while A-0003 amends its
point 4 (the file's format rules allow `superseded by A-MMMM`); and §13's last checklist item (*"GPU
decision recorded at the M1 freeze / M7 boundary (secured → Tier 2+ may be planned; not secured →
amendment scoping to Tier 0/1)"*) is stale after A-0004, because the GPU is now the *declared*
substrate for Tier 1–5 rather than a contingency.

---

## 8. (a) Claim ledger — one row per attacked claim

| # | Claim (verbatim, with source line) | Verdict | Evidence (quote + command output) |
|---|---|---|---|
| A1 | `novelty-risk.md` L45: P6 = *"Rank-only; reconstruction (not output-directed) cost; fixed bit width."* | **misattributed / overclaimed** | Fetched abstract: *"an integer linear programming formulation … enables **dynamic configuration of quantization parameters (e.g., bit-width, block size) for each matrix given an overall target memory budget**"* + *"a **data-aware** version … uses an approximation of the **Fisher information matrix**"*. `raw/citation_check.out` [P6] |
| A2 | `novelty-risk.md` L116: C6 = *"Rank allocation under a budget (rank-only). No bit dimension."* | **unsupported** | Same abstract. |
| A3 | `novelty-risk.md` L51: *"P6's allocation is bit-agnostic"* | **unsupported** | Same abstract; this is the sentence the A-difference statement rests on. |
| A4 | `novelty-risk.md` L54: *"Verdict: differentiated (weakly)."* | **overclaimed** | Follows from A1–A3: the $(r,b)$ currency is already occupied. |
| A5 | `novelty-risk.md` L90: *"Verdict: differentiated (moderate)."* (B) | **overclaimed (mild)** | Uncited neighbour SVDQuant (2411.05007) uses *"a **high-precision, low-rank branch** … while a **low-bit quantized branch handles the residuals**"*. `raw/citation_check.out` [UNCITED-SVDQuant] |
| A6 | `novelty-risk.md` L125: *"Verdict: previously-known as a concept"* (C) | **sound** | C1/C3/C4 fetched abstracts + uncited JoLT (2607.12550) all state joint rank/bit allocation under a storage budget. |
| A7 | `novelty-risk.md` L22: search covered *"bit-rank allocation"* | **overclaimed** | JoLT (2607.12550, title-exact) is absent from `literature-matrix.csv` (50 rows) and from `grep` over `docs/ references/`. |
| A8 | `novelty-risk.md` L42: P3 *"average-bit budget"*, *"Weight-space trace"* | **mischaracterised** | Abstract: *"the average of all of the Hessian eigenvalues"*, *"a **Pareto frontier** based method for selecting the exact bit precision … without any manual selection"*, *"extend the Hessian analysis to mixed-precision activation quantization"*. |
| A9 | `novelty-risk.md` L80: R4 *"SNR-targeted training reweighting."* | **mischaracterised** | Abstract: *"selectively targeting layer modules … and **freezing the remaining modules**"*. |
| A10 | `literature-matrix.csv` rows GPTQ/APTQ/SpectralNorm/SVD-LLM: `venue="arXiv preprint"` with `venue_verified="yes (arXiv abs page)"` | **unsupported** | The same pages say ICLR 2023 / DAC 2024 / ICLR 2018 / ICLR 2025. `raw/citation_check.out` [P2][P4a][R3][R5b] |
| P1 | `design-m2-interfaces.md` L15: proxy unit *"aggregated across layers by **summation**"* | **unsupported** | E3: `joint/Σ(single)` = 0.564–1.192 (non-additive both ways); `joint/Σ(proxy)` = 1.0e-4–4.9e-2; per-layer gain spread 32.9×–109.7×. `raw/probe_compose.out` |
| P2 | Implied claim: the proxy predicts **downstream** damage per layer | **unsupported** | E3: `Spearman(proxy, downstream damage)` = **0.119** (trained, LN, b=4) / 0.137 (b=8) vs **0.813** with LN removed. `raw/probe_compose.out` |
| P3 | `design-m2-interfaces.md` invariant 4: sampling info returned *"so variance is reportable"* | **mis-scoped** | E2: the estimator is unbiased; exact relative SE = 2.12e-4 (Gaussian) / 4.98e-4 (heavy-tailed) at the predeclared 524 288-token calibration, and in-sample fitting bias is ≈1.000 (1.0015/1.0000/0.9997). The error is **target error** (activation shift, non-composition), not sampling variance. `raw/probe_estimator.out` |
| P4 | Proxy computed on the uncompressed layer input is valid under joint compression | **unsupported** | E3 activation-shift: per-layer proxy moves −17.5 %…+0.7 % (trained runs; −86 % in the untrained run), systematic by layer type; `Σ proxy` ×0.972–0.998. `raw/probe_compose.out` |
| S1 | `preregistration.md` L152: *"bootstrap CIs (per-layer resampling)"* | **unverifiable as written** | No layer count is pinned; at L=6 the CI width is 1.26 (ρ=0.6); at L=4 the layer bootstrap is degenerate (width 0.000) and no tie/duplicate rule exists. `raw/probe_stats.out` |
| S2 | `preregistration.md` L148/L152: seeds (§7.1) vs layers (§7.2) | **contradictory** | Two resampling units, no combining rule; layers are not exchangeable (E3: proxy 0.919 vs 178.2 at the same depth). |
| S3 | `preregistration.md` L156/L160: *"Holm–Bonferroni"* | **unsupported** | No family enumeration, no `m`; under the "primary comparator only" reading m=1 and the correction is vacuous. |
| S4 | `preregistration.md` L160 (§7.3): *"a predeclared equivalence margin"* | **unsupported** | No value in the document or in §13. |
| S5 | `preregistration.md` L213/L215/L218 (§9.2/3/6): *"predeclared factor / patience window / per-run cap / tolerance"* | **unsupported** | No values exist; §9 calls the criteria *"predeclared, automatic"*. |
| T1 | `preregistration.md` L188: packed-storage row *"**runnable** (class 2 → class 3)"* | **unsupported / contradicts §4** | §4: local work = *"…and the class 4-CPU kernel *measurement*"*; `eval-protocol.md` L300: *"No evaluation cell is stated to run on this workstation"*. |
| T2 | `preregistration.md` L189: CPU-kernel row *"**runnable** (class 4-CPU …)"* | **unsupported by the environment** | `grep onnxruntime pyproject.toml` → no match; `import onnxruntime` → `ModuleNotFoundError`; evidence came from a venv outside the repo (`backend-capability.md` §1). |
| T3 | `preregistration.md` L128: H5 *"at measured stored-byte parity"* | **unexecutable as written** | No tolerance; container overhead is invocation-dependent (221–363 B on a 1 280 B payload; nominal 4.0 → **5.86–6.42** measured). `memory-accounting.md` §7.5 |
| T4 | `preregistration.md` §5 `b∈{2,3,4,8}` + H5 arm *"(int4/int8)"* | **inconsistent** | No `pack_int2`/`pack_int3` exists (`src/spectraquant/quantization/packing.py` defines only `pack_int4`, `pack_int8`), and permitted class-4-CPU kernels are int4/int8 only — so the H5 chain is undefined exactly where fake-quant gains are largest. |
| T5 | `preregistration.md` §10.4(a): free-tier cloud *"runs are sized to fit within a session"* | **unsupported** | No GPU-hour ceiling/quota assumption; from the project's own §6.4 numbers S4 alone is 0.4–18.3 GPU-h per model-seed. |
| E1 | `AGENTS.md` §4.5 / `preregistration.md` L31: equal-memory requirement | **prose-only** | §6 mechanism audit: no schema field, no test, no CLI gate; the allocator constrains class-1 `accounted_bytes` while P2 reports class-3 measured bytes. |
| M1 | `eval-protocol.md` L35: *"Class 4–5 numbers are unavailable here (`measurement-taxonomy.md` §4)"* | **contradicts the amendment** | The same file's L49 grants `LOCAL-CPU-MEASUREMENT`; `measurement-taxonomy.md` §4 says `4-CPU … AVAILABLE`. |
| M2 | `preregistration.md` §1/§4/§5/§8/§11/§12/§13 class text | **sound** | Re-read after A-0003/A-0004; no stale class-4 text remains (could not break). |

---

## 9. (b) BLOCKING ISSUES — what must change before the freeze

Ordered by severity. Each names the file, the sentence, and the minimal edit.

**B1 — `novelty-risk.md` §1 P6 row (L45) + C6 row (L116) + difference statement (L51).**
*Blocking for A and C.* Replace "Rank-only … fixed bit width" with the fetched content: rank allocated
under a total memory budget, **bit-width and block size configured per matrix by an ILP under the same
budget**, data-aware variant using a Fisher approximation; stops at per-matrix sequential allocation
with a reconstruction objective and no training-time regularizer. Delete *"P6's allocation is
bit-agnostic"* and re-derive the A-difference statement. Mirror the fix in `literature-review.md` §2.2
and §7.3.

**B2 — `novelty-risk.md` §2 (contribution B) and §1 (contribution A).**
Add **SVDQuant (arXiv:2411.05007)** to the neighbour tables and to `literature-matrix.csv`, and either
distinguish it explicitly (domain: diffusion weights+activations; its low-rank branch is
outlier-absorbing and kernel-fused) or downgrade B from *"differentiated (moderate)"* to
*"differentiated (weak–moderate, empirical)"*. Add **JoLT (arXiv:2607.12550)** to the C table.

**B3 — `novelty-risk.md` §0 (L22).** Replace the claim of coverage of the *bit-rank allocation* term
with the true statement: the term was searched and a title-exact 2026-07 hit (JoLT) was missed; log
the re-check in §5.

**B4 — `novelty-risk.md` §1 P3 row (L42) and `literature-review.md` §5.1.** HAWQ-V2: trace = average
Hessian eigenvalue, Pareto-frontier **exact** bit selection (not an average-bit budget), extended to
activation quantization; the layer Hessian is built from layer input statistics, so "weight-space" is
the wrong label. Keep "no activation propagation, no rank".

**B5 — `novelty-risk.md` §2 R4 row (L80).** "SNR-targeted training reweighting" →
"SNR-based **module selection with the remaining modules frozen** (not reweighting)".

**B6 — `literature-matrix.csv` venue fields (GPTQ, APTQ, SpectralNorm, SVD-LLM).** Either set the
venue from the abs page's comments field or downgrade `venue_verified` to
"yes (arXiv landing page; comments field not read)".

**B7 — `preregistration.md` §7.1/§7.2 (L148, L152).** Declare **one** primary resampling unit and its
justification. Recommended: matched **seeds** for arm-vs-arm (§7.1, already correct); for H2 make the
**model** the unit (one correlation per trained model, combined across seeds with a Fisher-z
random-effects model), with layers as a within-model nuisance — or, if layers are kept, state the
exchangeability assumption and justify it. Add the layer **granularity** (block vs linear map).

**B8 — `preregistration.md` §7.2 + §13.** Pin the Tier-1 layer count at freeze and add a power/MDE
statement for H2. Evidence to write into the document: with the declared per-layer bootstrap, power
to detect Δρ = 0.17 is **0.036–0.082 at L = 8–12** and 0.446 at L = 48 (`raw/probe_stats.out`).
Without this, H2 is pre-registered as an underpowered test.

**B9 — `preregistration.md` §7.3, §7.4 (L156, L160).** Enumerate the confirmatory families and their
`m` (e.g. a table: family F1 = {H2 vs primary comparator}, m = 1; F2 = {H1, H3, H4}, m = 3 …) and give
the equivalence margin and the matched-quality tolerance as numbers. Delete the Holm–Bonferroni claim
where m = 1.

**B10 — `preregistration.md` §9.2, §9.3, §9.6 + §13.** Give the four constants (divergence factor,
patience window, per-run wall-clock cap, byte-accounting tolerance) and the exact number of test reads
(§3.3). Otherwise §9's "predeclared, automatic" exclusion rules are neither.

**B11 — `preregistration.md` §8 row 8 (L188).** Split into a local **class-3 byte** measurement and a
`CLOUD-COLAB` **class-2 quality** re-measurement (or add the bounded §4 local exemption *and* the
matching `eval-protocol.md` §6.4 row).

**B12 — `preregistration.md` §8 rows 8–9 (L128, L188) + §5 bit-width grid.** Predeclare the byte-parity
tolerance and freeze the **serializer invocation** in the protocol (container overhead is
invocation-dependent: 221–363 B on a 1 280 B payload; 4.0 nominal → 5.86–6.42 measured). State that
the packed-storage/CPU-kernel H5 chain applies to `b ∈ {4, 8}` only, or define `pack_int2`/`pack_int3`
and a permitted kernel.

**B13 — `eval-protocol.md` §1 (L35).** Replace *"Class 4–5 numbers are unavailable here
(`measurement-taxonomy.md` §4)…"* with *"Class 4-GPU and class 5 numbers are unavailable here…;
class 4-CPU is available for self-serialized artifacts under `benchmark-protocol.md` §8"*. This is the
one place where the A-0003 amendment did not reach.

**B14 — `pyproject.toml` + `uv.lock` + `preregistration.md` §13.** Add the pinned
`onnx`/`onnxruntime`/`torchao` extra so the class-4-CPU cell is runnable from the frozen environment,
and add it to the freeze checklist. Without it, §8 row 9's "runnable" is not reproducible from the
repository.

**B15 — `preregistration.md` §10.4(a).** Add the cloud budget line: GPU-hour ceiling, session-length
and weekly-quota assumptions, and the maximum arm×seed count they permit; state what happens when the
ceiling binds.

**B16 — equal-memory gate (`AGENTS.md` §4.5 / `design-m2-interfaces.md` §4 / `artifacts/schemas/`).**
Implement the mechanism in §6 (schema fields + `assert_equal_memory` + a negative-fixture CI test +
a predeclared tolerance in §7.1/§13), and make the allocator's `cost_fn` include `overhead_fn` so the
constraint and the reported metric are the same quantity.

**B17 — `preregistration.md` §13 last item + `docs/coordination/status.md` §2/§6.** Re-point the stale
"GPU decision at M1/M7" item after A-0004 (the GPU is now the declared Tier-1–5 substrate), and update
the two status lines that still read "Measurement classes 4–5 unavailable".

---

## 10. (c) Things I tried to break and could not

1. **P1/RAM (arXiv:2609.33923).** Every quoted number is in the real abstract: unbiased probe,
   spectral flatness, propagated probe *"rank-correlates 0.81 to 0.83 with the GPTQ layer objective
   from real activations, while the isolated estimator is uncorrelated"*, *"knapsack solver allocates
   bits under an exact byte budget"*, *"3.5 to 13.6% lower median WikiText-2 perplexity than
   size-comparable uniform 4-bit builds"*. No misattribution, no overstated overlap. Its "no rank"
   stopping point is correct.
2. **Wrong years / wrong authors.** Zero errors across the 17 fetched works (table in §1); every ID
   resolves to the stated title, first author and year.
3. **Contribution C's "previously-known" verdict.** Cannot be broken — it is conservative and now
   additionally supported by an uncited work (JoLT) that the document did not know about.
4. **The preregistration's own class-4 / H5 text after A-0003+A-0004.** §1, §4, §5, §8, §11.0, §11.1,
   §12, §13 were re-read line by line against `AGENTS.md` §5 and both amendments; the earlier
   "§8 row 13" traceability error is gone. Only `eval-protocol.md` still carries the stale sentence
   (B13).
5. **The measurement-class separation as such.** `P1 = class 2` and `P2 = class 3` are stated
   consistently, and no text presents a fake-quantized float checkpoint as low-bit storage (the
   `memory-accounting.md` §6 rule and the taxonomy's forbidden-claim table are correct and consistent
   with each other). The one defect is the *chain* label in row 8 (B11).
6. **The byte-accounting arithmetic.** I hand-checked `memory-accounting.md` §3/§7: `4096·ceil(4096/128)
   = 131 072` blocks; `8 388 608 + 262 144 = 8 650 752 B` ⇒ `4.1250` bits/param; `blocks_B = out·ceil(r/g)
   = 4096·1` for `r=64,g=128`; `262 144 + 12 288 = 274 432 B` ⇒ ratio `0.008179`. All correct, including
   the trailing-short-group `ceil`. The rules are self-consistent and testable.
7. **H1–H4's falsifier structure.** Each hypothesis has a two-sided, pre-stated refuting result, and
   §11.1's consequences (drop the framing, downgrade the contribution) are honest. No HARKing pattern
   found in the hypothesis wording.
8. **Calibration/dev/test separation.** `preregistration.md` §3 + `eval-protocol.md` §2 define five
   disjoint roles with a programmatic overlap check and an exclusion rule; I found no text that lets a
   dev/test byte into calibration.
9. **The M2 interface invariants 1, 2, 5, 6** (one accounting source, one factor convention, no CUDA
   in core paths, pure torch/numpy) — internally consistent and consistent with `AGENTS.md` §2. The
   proxy slice is still a `NotImplementedError` stub (`src/spectraquant/proxies/sensitivity.py`), so
   the aggregation rule in invariant 3 is **still cheap to change before implementation** — which is
   the point of reviewing now.

---

## 11. Reproduction

```bash
# citation evidence (raw HTML kept; 17 cited + 2 uncited neighbours)
uv run python docs/results/verification/check_citations.py > docs/results/verification/raw/citation_check.out
# E1 design power + E2 estimator variance   (raw/probe_stats.out)
uv run python docs/results/verification/probe_stats.py
# E2 estimator bias/variance                (raw/probe_estimator.out)
uv run python docs/results/verification/probe_estimator.py
# E3 layerwise-proxy composition, LayerNorm control, activation shift (raw/probe_compose.out)
uv run python docs/results/verification/probe_compose.py
# E4 local CPU training throughput          (raw/probe_throughput.out)
uv run python docs/results/verification/probe_throughput.py
# quote ledger + snapshot hashes + git status
cat docs/results/verification/raw/{quote_ledger.txt,snapshot_hashes.txt,git_status.txt}
```

**Write-set confirmation** (`git status --short`, 2026-10-08T20:58:45Z, HEAD `98d1f60`, branch
`infra/bootstrap`; full output in `raw/git_status.txt`): the only paths I created are under
`docs/results/verification/**` (`find docs/results/verification -type f` in the same file — 4 probe
scripts, 3 evidence scripts/ledgers, 22 raw artifacts, 13 snapshot copies, this report). The modified
paths (`src/spectraquant/{factorization,quantization}/*.py`) and the other untracked paths
(`docs/results/`, `src/spectraquant/cloud/`, `src/spectraquant/quantization/accounting.py`,
`tests/unit/test_factorization_*.py`, `docs/research/spectral-notes.md`, …) belong to sibling streams
on the shared worktree; **I edited none of them**. Every attacked document is tracked and shows no
modification flag, and the drift check prints `SAME` for all ten — the quotes in this report match
committed content.
