# ADR-0004 — The SpectraQuant method: joint low-rank and low-bit training under a byte budget

Status: **proposed** (implementation not started). Required by `AGENTS.md` §3 (a shared interface is
specified before it is implemented) and by the continuation brief's Phase 2 ("write an ADR specifying
the mathematical objective, trainable tensors, base/low-rank decomposition, straight-through
estimator, proxy unit, allocator inputs, byte formula, checkpoint format and failure behavior").

## 1. Context

The repository has every component the method needs and **no arm that is the method**. What exists:
the quantizer (`fake_quantize`, `QuantSpec`), the 2-bit NF codebook, the factorizer
(`truncated_svd`, `initialize_svd`), LoftQ and LR-QAT initialisations, the rounding-aware
`PreparationObjective` (four ablable terms), ten proxy variants including the declared candidate
`gain_aware_composed`, the CP-SAT allocator over an injected `error_fn`, `QuantizedPlusLowRankLinear`
(frozen quantized base + trainable factors), `StraightThroughQuantizedLinear`, and the training loop
with checkpoint/resume. The M5 gate is: **the method beats its ablations on development outcomes
within the memory budget** (`docs/coordination/status.md`), and it is currently **NOT MET on the
fixture** — the regularizer reduced the rounding residual but *worsened* dev NLL, so the two quality
axes disagreed in sign (A-0008).

The method must therefore be defined so that its **primary quality outcome** — language-model quality
at equal measured stored bytes — is what improves, not its own penalty term. That is the failure mode
A-0008 already demonstrated, and this ADR is written to make it structurally hard to repeat.

## 2. The objective

For a target linear layer with weight `W`, the method trains factors `A (r, in)`, `B (out, r)` on a
**frozen** quantized base `Q`:

```
L = L_task  +  lambda_factor * factorization_proxy
            +  lambda_round  * rounding_grid_penalty
            +  lambda_residual * rounding_residual_ratio
            +  lambda_spectrum * spectral_tail_energy
```

The four penalty terms are the existing `PreparationObjective` terms, unchanged and individually
ablatable (every coefficient defaults to `0.0`, so an omitted weight is visible in the diagnostics
rather than silently applied). `L_task` is next-token cross-entropy on the training corpus. The
deployed weight is

```
W_eff = Q + B @ A            (paper Eq. 6 reconstruction)
```

**The declaration that matters:** the method's *primary* claim is about `L_task` on the held-out test
split at matched measured bytes. The penalty terms are mechanisms; a run that lowers a penalty while
raising test perplexity is reported as a **negative result for that configuration**, never as a
success, and the diagnostics must show both axes side by side.

## 3. Trainable tensors

| Tensor | Trainable | Note |
|---|---|---|
| `A`, `B` (per targeted layer) | **yes** | the only parameters of the method |
| `Q` (the quantized base) | **no** | a buffer; the artifact being approximated |
| embedding / LM head / RMSNorm | **no** | frozen, as in the reference recipes |
| the LR-QAT step size `s` | **no** | that is the `r2_lrqat_4bit` arm, not the method |
| `lambda_*` | **no** | predeclared constants, selected on dev only |

Rationale: the research question is whether *preparing* a factorization for quantization helps. That
preparation acts on the factors, so the factors are what trains; a trainable base would make the
result a QAT result and destroy the comparison against the QAT baselines.

## 4. Base / low-rank decomposition, and where the base comes from

Two initialisations are available and both are used, because the comparison between them *is* part of
the study:

* **`quantize-then-decompose`**: `Q = q(W)`, `(A, B) = SVD_r(W − Q)`. This is LoftQ's `T = 1` and
  LR-QAT's starting point (amendment A-0014 fixed our implementation to match).
* **`alternating`** (LoftQ `T = 5`): iterate `Q_t = q(W − B_{t−1} A_{t−1})`, `(A_t, B_t) =
  SVD_r(W − Q_t)`.

The method's default is the alternating initialisation at `T = 5` with the 2-bit NF codebook at block
64 (the frozen R1 configuration), because that is the preparation whose effect on subsequent
quantization the project is asking about. The `T = 1` and standard-init variants are its ablations.

**Invariant:** within one comparison the base `Q` must be *bit-identical* across the arms that differ
only in initialisation — the trend is about the initialisation, so a different base would confound it
(already enforced for R1 by a test).

## 5. Straight-through estimator

The method's forward pass uses `W_eff = Q + B @ A` with `Q` a *dequantized* float tensor: no
straight-through estimator is needed, and none is used. Gradients reach `A` and `B` through ordinary
autodiff. The STE exists in the repository only for the `r2_fullqat_4bit` baseline
(`StraightThroughQuantizedLinear`), where the weights themselves are trained through the quantizer.

Documented bias: the STE's gradient is the gradient of the *dequantized* weight, which is not the
gradient of the quantized artifact. Any arm using it must say so, and the method does not use it.

## 6. The proxy unit, and the allocation error model

The proxy unit is the frozen one (`design-m2-interfaces.md` §0.3):

```
E_X || X W^T  −  X (Q + B A)^T ||^2
```

estimated on a held-out **calibration** batch of the plan's calibration corpus — never on the test
split. The declared candidate is `gain_aware_composed`, which must beat `weight_frobenius`,
weight-magnitude, activation-magnitude, Hessian-diagonal and naive output-error comparators on the
predeclared criteria; the fixture half of that comparison passed (15/15 contrasts) and the real-model
half is **NOT RUN**.

The allocator's `error_fn` is exactly this proxy, injected through the existing proxy adapter. The
allocator itself is unchanged: `AllocationProblem(layer_shapes, ranks, bits, budget_bytes,
cost_fn=make_accounting_cost_fn(...), error_fn=<proxy>)`, solved by CP-SAT, with greedy and uniform
baselines and an exhaustive oracle on tiny cases.

## 7. The byte formula

One source of truth: `spectraquant.quantization.accounting`. For a layer the stored object is

```
bytes(layer) = accounted_bytes((out, in), quant_spec)              # the quantized base: codes + scales
             + 2 * (numel(A) + numel(B))                            # the factors, fp16
```

Class **1** (analytical from shapes and bit widths) unless the artifact is serialized, in which case
the class-**3** measured figure is recorded beside it and the equal-memory gate is applied to it. The
factors are counted at *storage* precision, not at their float32 training size, and the rank is the
**effective** rank (the SVD clamps it on a narrow layer — already handled).

The method is only ever compared at equal measured bytes, and every comparison reports the equal-memory
counterpart (`AGENTS.md` §4.5).

## 8. Checkpoint format and resumability

The loop's existing checkpoint (`model` + `optimizer` + batch-generator state + loop config + step
count) is reused unchanged, plus the method's own record:

* the resolved allocation (layer → rank, bits, group size) as it was applied;
* the proxy scores per layer and the allocator's objective value;
* the predeclared budget and the achieved bytes;
* the seed, the corpus checksum and the calibration batch identity.

A resume must reproduce the same allocation and the same corpus slice; a mismatch is refused rather
than silently re-allocated. `training.resumed_from` and the total steps reached are recorded.

## 9. Failure behaviour

* **Divergence** (perplexity > 1000, NaN, or a loss that has not decreased after 100 steps) is recorded
  as `training.divergent` with the trigger, and the number is not emitted as a quality measurement.
* **A short run is a failed run**: fewer steps than declared ⇒ `status: failed` with the count.
* **No silent fallback**: a missing extra, a CUDA request without CUDA, an unimplemented arm, or an
  allocation that exceeds the budget fails loudly and names the cause.
* **No test-split peeking**: the allocation and the coefficients are selected on dev; the test split is
  read once, after training.

## 10. What the method must beat, and what would falsify it

At equal measured stored bytes, on the same seeds, the method is compared against: fp16; uniform PTQ
8-bit and 4-bit; low-rank only; rank-then-quantize; quantize-then-residual; LR-QAT; LoftQ; proxy
allocation without the regularizer; the regularizer with a uniform allocation; and the method with
each major component removed (proxy ablations without downstream gain, and without each penalty term).

The method gate passes only if the full configuration improves the **primary quality metric** over
those baselines *and* over its own ablations, with a stable direction across seeds. If it does not, the
negative result is published and the diagnosis must say which component caused it (proxy, allocation
objective, regularizer, or the trainable representation). **A benefit that exists only in a penalty
term is not a benefit** (A-0008).

## 11. Open questions for the owner

1. **Rank and bit grids.** The method needs a candidate set per layer. The frozen Tier-1 grid is ranks
   `[2, 4, 8, 16, 32]` and bits `[2, 3, 4, 8]` at group sizes `[32, 64, 128]`; the R1 arms use rank 16
   at 2-bit NF. Should the method's candidates be restricted to the codebook quantizer (2/4-bit NF,
   block 64) so it is comparable with R1, or use the uniform grid so it is comparable with R2? Doing
   both doubles the matrix; my default would be the codebook, since that is where the frozen LoftQ
   trend lives.
2. **The regularizer's coefficients.** A-0008's mixed result came from the regularizer; the method
   needs predeclared coefficients (or a dev-selected rule). Selecting them on dev is legitimate but
   must be predeclared, and the selection rule has to be written before the run.
3. **Tier-1 or Tier-2 first.** The method gate is more credible at Tier-2 (TinyLlama-1.1B) but costs
   6× more per arm. My default: implement and shake out at Tier-1, then run the confirmatory cell at
   Tier-2 with 3 seeds.
