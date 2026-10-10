# Handoff — SpectraQuant, 2026-10-10 (final)

**Branch** `research/tier1-closure` · **remote** `github.com/OussemaHarrabi/SpectraQuant` ·
**tag** `repro-v0.1` · **substrate** local CPU (Tier 0) + Kaggle free tier (Tier 1)

This is the state of the repository at the end of the build, written so that a reader who was not
present can pick it up. It states what is measured, what is not, and the single next action.

---

## 1. What the project asked

Whether preparing a transformer for low-rank factorization helps or hurts its subsequent quantization,
and whether an output-aware sensitivity proxy can pick layer-wise ranks and bit widths that beat
uniform compression at **equal stored bytes**. Hypotheses H1–H5, frozen protocol
`docs/research/preregistration.md` (FROZEN 2026-10-09 at `6281c56`).

## 2. The measured answer

| configuration | perplexity | accounted B | substrate | class |
|---|---|---|---|---|
| fp16 reference | 14.0183 | 212 336 640 | CLOUD-GPU | — |
| uniform int8 PTQ | 14.0415 | 112 803 840 | CLOUD-GPU | 2 |
| **uniform int4 PTQ** | **18.6193** | **59 719 680** | CLOUD-GPU | 2 |
| allocation, untrained | 64.5477 | 59 719 488 | CLOUD-GPU | 2 |
| **method, regularizer off** | **23.1698** | 59 719 488 | CLOUD-GPU | 2 |
| **method, regularizer on** | **23.3131** | 59 719 488 | CLOUD-GPU | 2 |

Equal measured bytes to **192 B (0.0003 %)**. **The central hypothesis is not supported** at this
scale, and the preparation regularizer's own contribution is **negative** (+0.62 % perplexity).
LR-QAT's direction reproduces (24.1265 vs 26.9432) at **21.8 %** of the predeclared 25 % magnitude —
partially reproduced, not confirmed. Full records: `docs/results/tier1-method-comparison.md`,
`docs/results/m3-verdict.md`.

## 3. What is implemented and verified

* **Quantization**: symmetric per-group, 2-bit NF-codebook (block 64, reference-exact), a Q4.4
  fixed-point container, pack/unpack with a single byte-accounting source of truth.
* **Factorization**: SVD truncation, quantize-then-decompose (`T = 1`) and the alternating LoftQ
  schedule (`T = 5`), a differentiable `LowRankLinear`, `QuantizedPlusLowRankLinear`.
* **Proxies**: 10 variants incl. the declared candidate `gain_aware_composed`; 15/15 fixture contrasts.
* **Allocator**: uniform / greedy / CP-SAT exact, proxy adapter, exhaustive oracle, byte-costed.
* **Regularizers**: the four-term preparation objective, CPU primitives, device-attached penalty.
* **Cloud**: RunSpec, generated thin notebooks, Kaggle submit/poll/fetch/collect, run registry,
  checksum validation, budget guard — executed end to end on Kaggle.
* **Measurement**: the five-class taxonomy, the equal-memory gate, class-4-CPU ONNX kernels, the
  claim guard, the release-artifact generator.

## 4. What is NOT done (never to be reported as measured)

* **Tier-2** (TinyLlama-1.1B, 3 seeds) and the frozen M3 protocol at full scale — **NOT RUN** (A-0018).
* The **H2 real-model proxy-ranking** cell and the **H4 frontier** — **NOT RUN**.
* The multi-seed H3 comparison — **NOT RUN** (one run per configuration).
* **Class 4-GPU** and **class 5** — unavailable on this workstation; nothing here claims them.
* The LM-Eval task suite — **NOT RUN** (DEV-0009).
* LoftQ as a trained baseline (T2) — **NOT RUN**.

## 5. Honest caveats that must travel with the numbers

* One seed per cloud cell; one model; one corpus; **21 758 evaluation tokens** (a cap).
* The byte figures cover the **210 targeted projection tensors**, not the deployed model.
* Class 2 is fake quantization: it licenses **no** storage or latency claim.
* The T1 LR-QAT criterion is at equal **bit width**, not equal memory (the treatment costs +35.6 % bytes).
* The M3 T1 subset is a **subset** of the frozen protocol, not the protocol.

## 6. Reproducing

```bash
git clone https://github.com/OussemaHarrabi/SpectraQuant.git && cd SpectraQuant
git checkout research/tier1-closure      # or tag repro-v0.1
uv sync --all-extras
uv run pytest -q                          # full CPU suite
uv run ruff check . && uv run ruff format --check . && uv run pyright
uv run spectraquant --help
uv run spectraquant smoke --config configs/experiment/smoke.yaml
```

Cloud cells: the four commands in `docs/results/tier1-method-comparison.md` §5 and
`docs/results/m3-verdict.md` §6 (notebook → submit → fetch → collect).

## 7. Where everything lives

| need | path |
|---|---|
| the question and its frozen protocol | `docs/research/preregistration.md`, `docs/research/reproduction-plan.md` |
| the charter and the binding rules | `AGENTS.md`, `docs/research/charter.md` |
| the central result | `docs/results/tier1-method-comparison.md` |
| the reproduction verdict | `docs/results/m3-verdict.md` |
| every claim and its evidence | `docs/results/experiment-index.md`, `docs/results/verification/claim-audit.md` |
| the write-up | `reports/paper/paper.md`, `docs/report/spectraquant-technical-report.md` |
| the amendment log | `docs/research/preregistration-amendments.md` (A-0001…A-0020) |
| the decisions | `docs/decisions/ADR-0001…0004` |
| the compute ledger | `docs/results/compute-budget-ledger.md` |
| career-facing summary | `CAREER_EVIDENCE.md` |

## 8. Recommended next action

**Do not re-run anything that already ran.** The single next action is the **Tier-2 primary
experiment** on the cloud substrate: run the frozen protocol's arm set on TinyLlama-1.1B at three
seeds with the pinned plan, after pinning a corpus-sized schedule from a measured step cost, and add
the H2 real-model proxy-ranking cell so the proxy's ranking is measured against observed damage on the
model it would be used on. Everything else is downstream of that.

If Tier-2 is out of reach, the next-best action is to **re-solve the method at the int4 baseline's
measured bytes** (instead of equal bit width) so the central comparison is equal-memory by
construction — the current comparison is equal *accounted* bytes, which is the same thing here but
would not be if the container overhead changed.
