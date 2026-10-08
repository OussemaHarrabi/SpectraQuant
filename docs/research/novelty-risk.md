# SpectraQuant — Novelty and Prior-Art Risk

Status: **Wave 1, draft 0.1 (2026-10-08).**

## 0. Novelty policy (binding)

1. No claim in this repository, in any report, or in any external communication may state that a
   component is novel unless this file gives **a cited closest neighbour** and **an explicit
   difference statement** that a reader can check.
2. Every candidate idea is labelled with exactly one verdict:
   - **previously-known** — the idea, as stated, exists in prior work; we may only *use* it.
   - **differentiated** — the idea overlaps prior work but a stated, testable difference remains.
   - **open** — we searched and found no work stating the idea; "open" is a statement about *our
     search on 2026-10-08*, not a proof of absence.
3. The phrase **"candidate contribution"** is used for every idea we might claim. It may be upgraded
   to a contribution claim only after the preregistered tests (§ `preregistration.md`) confirm it.
4. The literature is crowded and moving fast (several relevant 2026 preprints appear below). This
   file MUST be re-checked before any submission, and every re-check logged in §5.

Search coverage on 2026-10-08 (what we actually did): arXiv full-text/title queries via the arXiv
API and `arxiv.org/abs` pages for the terms *low-rank × quantization*, *quantization-aware low-rank
training*, *mixed-precision allocation*, *sensitivity proxy*, *bit-rank allocation*, *spectral
regularizer quantization*, plus forward-citation checks from LR-QAT/LoftQ/LQ-LoRA/HAWQ/MLoRQ. Not
covered: non-English venues, industry blogs, patents, code-only releases without papers. This is a
**bounded** search and its bounds are part of the risk.

---

## 1. Candidate contribution A — output-aware quantization-cost proxy with a rank dimension

**Idea.** A proxy that (i) estimates the damage a *compressed operator* does to its **output**
distribution (propagating activation statistics through the compressed map, not scoring weights in
isolation), and (ii) is defined for a **two-dimensional compression choice** — rank $r$ *and* bit
width $b$ — so that rank and bit width are compared in a common cost currency.

**Closest prior art (must be cited in any claim):**

| # | Work | What it already does | Where it stops |
|---|---|---|---|
| P1 | Kennedy & Kennedy, RAM, arXiv:2609.33923 (2026) | An **unbiased random-probe estimator of a layer's squared quantization-error norm**, shown to be well-behaved because round-to-nearest error is spectrally flat; the **propagated** form (input-statistics-carrying probes) rank-correlates 0.81–0.83 with GPTQ's real-activation layer objective, while the *isolated* form is uncorrelated; knapsack allocation under an exact byte budget; beats a uniform 4-bit build at matched bytes. | Bit width only; **no rank**; no low-rank preparation; calibration-free by construction but no rank-coupling term. |
| P2 | Frantar et al., GPTQ, arXiv:2210.17323 (2022) | The de-facto **output-aware layer objective** $\lVert WX-\hat WX\rVert_F^2$ from layer input statistics. | Defined only for a quantizer; no rank; used *inside* a quantizer, not as an allocator input in our sense. |
| P3 | Dong et al., HAWQ-V2, arXiv:1911.03852 (2019) | Hessian-**trace** sensitivity as an allocation proxy under an average-bit budget (output/Hessian aware in a weak sense). | Weight-space trace, no activation propagation, no rank. |
| P4 | Guan et al., APTQ, arXiv:2402.14866 (2024); Lee et al., KronQ, arXiv:2607.07964 (2026) | 2024–2026 output/attention/gradient-aware sensitivity for mixed precision. | Bit width only; no rank; PTQ framing. |
| P5 | Misra et al., MixQuant, arXiv:2607.23047 (2026); Zhao et al., CoopQ, arXiv:2509.15455 (2025) | Show that a layer's sensitivity depends on other layers' precisions and that **propagated / interaction-aware** scores beat isolated ones. | Bit width only; no rank; different estimator machinery (marginalization, Shapley). |
| P6 | Guo et al., LQ-LoRA, arXiv:2311.12023 (2023) | Ranks allocated across layers under a **parameter budget** using a reconstruction objective. | Rank-only; reconstruction (not output-directed) cost; fixed bit width. |

**Difference statement (candidate contribution A).** Relative to P1–P6, the candidate differs by
making the proxy a function of the **pair $(r,b)$** — i.e. it must return a comparable damage
estimate for "this layer at rank $r_1$, $b_1$" and "this layer at rank $r_2$, $b_2$" — so that a
single allocator can trade rank against bit width at equal **stored bytes**. P1's probe is
rank-agnostic; P6's allocation is bit-agnostic; P2–P5 are bit-only. That $(r,b)$ coupling, evaluated
under an equal-stored-bytes protocol, is the candidate's stated difference.

**Verdict: differentiated (weakly).** Output-awareness itself is **previously-known** (P1, P2, P4,
P5). The rank dimension in the cost currency is **differentiated**, and the strength of the
difference is *empirical*: if our $(r,b)$ proxy does not beat a rank-agnostic probe composed with a
separate rank heuristic (**H2** proxy validity paired with **H4** allocation), the difference collapses.

**Falsified if (tests spec H2, and H4 downstream):** the $(r,b)$ proxy's ranking of layers is no better than P1's probe for bit width
and no better than SVD-LLM-style truncation error for rank, at matched budget, on the models we can
measure. In that case we cite P1/P6 and drop the claim.

---

## 2. Candidate contribution B — rounding-aware spectral regularizer

**Idea.** A regularizer applied **during low-rank preparation** whose penalty is simultaneously
(i) *spectral* (it acts on the singular values / directions of the prepared factor) and (ii) *tied to
the quantization grid* (it penalizes energy redistribution that will be moved by round-to-nearest at
the target bit width), so it optimizes not just conditioning but the *survival* of dominant directions
through rounding.

**Closest prior art (must be cited in any claim):**

| # | Work | What it already does | Where it stops |
|---|---|---|---|
| R1 | Cho et al., SRR (Preserve-Then-Quantize), arXiv:2602.02001, ICML 2026 | **Splits a rank budget**: preserve the top-$k$ singular subspace of the activation-scaled weight *before* quantization; use the remaining $r-k$ ranks to reconstruct the quantized residual; theory-guided $k$ from "quantization-exposed energy" vs "unrecoverable error". | A closed-form **post-training selection criterion**, not a differentiable penalty applied during training; no joint rank × bit allocation; the "regularization" is on the *parameterization* (gradient scaling along preserved directions) rather than a rounding-aware spectral loss term. |
| R2 | Schotthöfer et al., arXiv:2505.08022 (2025) | A **spectral regularizer on the condition number of the low-rank core** during dynamical low-rank training, for adversarial robustness. | Motivated by robustness, **not** quantization rounding; no quantization grid in the penalty. |
| R3 | Miyato et al., Spectral Normalization, arXiv:1802.05957 (2018) | Bounds the spectral norm of each layer. | Generic Lipschitz control; no rounding, no low-rank preparation for compression. |
| R4 | Hartford, Spectrum, arXiv:2406.06623 (2024) | SNR-targeted training reweighting. | Signal-quality reweighting, not a spectral/rounding penalty. |
| R5 | Yuan et al., ASVD, arXiv:2312.05821 (2023); Wang et al., SVD-LLM, arXiv:2403.07378 (2024) | Truncation-aware / activation-aware SVD that accounts for the discarded part. | Rank selection heuristics; no training-time regularizer, no rounding term. |

**Difference statement (candidate contribution B).** Relative to R1–R5, the candidate differs by
being a **training-time differentiable penalty** that couples the **spectrum of the prepared factor**
with the **round-to-nearest grid at the target bit width**, applied *while* the low-rank preparation
happens. R1 is post-training and closed-form; R2 is spectral but quantization-blind; R3/R4 are not
about compression; R5 is post-hoc rank selection. The candidate is *not* claimed to be the first
spectral regularizer, nor the first rounding-aware method.

**Verdict: differentiated (moderate).** The *conjunction* (training-time + spectral + rounding-grid)
is not present in the works we found on 2026-10-08, but each ingredient is individually
**previously-known**. This is the component most exposed to a fast-moving literature; see §5.

**Falsified if (tests spec H3):** the regularizer gives no improvement over unregularized preparation at
equal stored bytes after controlling for rank/bit width, or the improvement disappears when the
"rounding-aware" term is replaced by a plain spectral penalty (i.e. the rounding coupling is not
doing any work). Either outcome downgrades the candidate to "spectral regularizer variant".

---

## 3. Candidate contribution C — mixed-rank / mixed-bit allocation under a stored-bytes budget

**Idea.** Allocate a per-layer pair $(r_\ell, b_\ell)$ under one **stored-bytes** budget using the
proxy of A, and compare the resulting quality–memory Pareto frontier against uniform compression at
equal stored bytes.

**Closest prior art (must be cited in any claim):**

| # | Work | What it already does | Where it stops |
|---|---|---|---|
| C1 | Gordon et al., MLoRQ, arXiv:2507.09616 (2025) | **Joint per-layer bit-width and rank assignment** under a memory constraint; two-stage intra-layer enumeration + inter-layer assignment; adaptive-rounding error mitigation. Vision Transformers. | Search-based (not a closed-form proxy), vision-domain, no output-aware propagated proxy, no regularizer. |
| C2 | Wang et al., ASP-DAC 2026, pp. 604–610 | Training-free **joint low-rank + mixed-precision** LLM compression with an input-aware sensitivity measure; unified difference matrix for the combined error. | Not verified to have an arXiv preprint; sensitivity is input-aware, not a $(r,b)$ propagated cost; no training-time regularization. |
| C3 | Zhou et al., AutoQRA, arXiv:2602.22268 (2026) | Joint search over per-layer bit width and **LoRA rank**; evolutionary + Bayesian optimization under a memory budget. | Rank is of *adapters*, not base compression; expensive search; no proxy. |
| C4 | Ha et al., KV-COBRA, arXiv:2609.24298 (2026) | Co-optimized **bit-rank allocation** formalized as a resource-allocation problem balancing rank-truncation vs quantization loss; redistributes budget across heads. | KV cache, not weights; per-head solver; no spectral regularizer; calibration-defined. |
| C5 | Misra et al., MixQuant, arXiv:2607.23047 (2026); Kennedy & Kennedy, RAM, arXiv:2609.33923 (2026) | Budget-aware mixed-precision allocation under an exact byte budget (bit-only). | No rank dimension. |
| C6 | Guo et al., LQ-LoRA, arXiv:2311.12023 (2023) | Rank allocation under a budget (rank-only). | No bit dimension. |

**Difference statement (candidate contribution C).** "Jointly allocating rank and bit width under a
budget" is **previously-known** (C1, C2, C3, C4). The candidate's only possible difference is
*operational and evaluative*: allocation driven by the **$(r,b)$ output-aware proxy of A** (rather
than search or a reconstruction objective), evaluated on a **quality–memory Pareto frontier at equal
stored bytes** rather than at a single budget, with **weights** (not KV cache, not adapters) as the
compressed object and with **measurement classes** (per `AGENTS.md` §5) labelled.

**Verdict: previously-known as a concept; differentiated only through A.** The allocation mechanism
itself is not a claimable contribution. If A's difference weakens, C weakens with it; C must never
be reported as a standalone novelty.

**Falsified if (tests spec H4):** proxy-driven allocation does not dominate uniform rank/bit-width on the
Pareto frontier (no byte-budget at which it wins beyond CIs), or it is matched by a trivial baseline
(e.g. uniform rank + per-layer bit from HAWQ-V2) at equal bytes.

---

## 4. Residual "openness" statements (bounded, dated)

- **Open (bounded):** a $\lVert\cdot\rVert$-style penalty that is *both* differentiable, spectral, and
  defined against the round-to-nearest grid of the *target* bit width, applied during low-rank
  preparation — we found no work stating this on 2026-10-08. Each ingredient is individually known.
- **Open (bounded):** an output-aware cost proxy whose domain is the *pair* $(r,b)$ for weight
  compression of transformers, evaluated at equal stored bytes — nearest works cover one axis each
  (P1/P6) and the pair via search (C1/C2/C3/C4).
- **Not open:** joint rank × bit allocation; output-aware sensitivity in general; spectral
  regularizers in general; low-rank-plus-quantized decomposition; QAT with low-rank auxiliary
  weights.

## 5. Re-check log (append-only)

| Date | Query / sources checked | Finding | Action |
|---|---|---|---|
| 2026-10-08 | arXiv API + `arxiv.org/abs` for themes in §0; forward citations from LR-QAT/LoftQ/LQ-LoRA/HAWQ/MLoRQ; web_search for 2025–2026 joint compression | RAM (2609.33923), MixQuant (2607.23047), CoopQ (2509.15455), KronQ (2607.07964), MLoRQ (2507.09616), AutoQRA (2602.22268), KV-COBRA (2609.24298), SRR (2602.02001), ASP-DAC 2026 joint-compression | Verdicts A=differentiated(weak), B=differentiated(moderate), C=previously-known; charter/novelty language constrained accordingly |

Re-check is mandatory before: any submission, any `reports/**` draft that uses the word "novel", and
any public README claim. A re-check that changes a verdict MUST be recorded here and, if it changes
the scientific plan, as a `preregistration-amendments.md` entry.
