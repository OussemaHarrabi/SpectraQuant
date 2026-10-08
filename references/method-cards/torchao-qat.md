# Method card — torchao QAT (library baseline, not a paper method)

Field list follows the assignment in the Milestone 3 ticket (wave 1). `references/method-cards/TEMPLATE.md`
was not present when this card was written (owned by another stream) — if the template fixes different
field names, this card must be re-synchronised.

**This card documents a *software artifact*, not a paper method.** torchao is the pinned third-party
implementation of the *full-model QAT reference arm* of the LR-QAT reproduction
(`docs/research/reproduction-plan.md` §3.2, decision D5). It is a **dependency**, never a source of
copied code, and its numbers are never presented as a published result.

---

## 1. Exact citation

- **Software**: `torchao` — *TorchAO: PyTorch-Native Training-to-Serving Model Optimization*
- **Paper**: `arXiv:2507.16099` (Or, Jain, Vega-Myhre, Cai, Hernandez, Zheng, Guessous, Kuznetsov,
  Puhrsch, Saroufim, Rao, Tran, Samardžić; submitted 2025-07-21)
- **Venue**: ICML 2025 Workshop on Championing Open-source DEvelopment (CODEML 2025), 5 pages —
  i.e. a **workshop paper about the library**, not a method paper. There is **no peer-reviewed paper
  for torchao's QAT recipe**; the recipe is documented in the library's own workflow guide.
- **Repository**: `https://github.com/pytorch/ao`; PyPI project `torchao`
- **Method lineage** (for the QAT mathematics itself, which is textbook): Jacob et al., *Quantization
  and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, `arXiv:1712.05877`
  (CVPR 2018) for the fake-quant + STE formulation, and Nagel et al., *A White Paper on Neural Network
  Quantization*, `arXiv:2106.08295` (2021) for the affine/decomposed formulation.

## 2. Upstream repo + pinned commit

| Field | Value |
|---|---|
| Repository | `pytorch/ao` |
| Pinned commit (provenance) | `cff77b46ef85ba8472b4a286598073d6c01e18e5` (2026-10-07), `version.txt` = `0.19.0` (unreleased dev line) |
| **Version actually depended on** | **`torchao==0.18.0`** — the latest released version (`https://pypi.org/pypi/torchao/json`), distributed as `torchao-0.18.0-py3-none-any.whl` (pure Python, installs on Windows/CPU) |
| Verified by | `git ls-remote … HEAD` and `gh api repos/…/commits/main --jq .sha` agree; PyPI metadata read live (`upstream-notes.md` §1, §3.4) |
| Licence (file read) | `LICENSE` — **BSD-3-Clause** (`Copyright 2023 Meta`, Arm contributions); `CITATION.cff` also declares `license: "BSD-3-Clause"`. GitHub's classifier reports `NOASSERTION` because of the multi-holder text — the file and `CITATION.cff` govern. |
| Clone location | `%TEMP%/sq-upstream/ao` — outside the working tree; **nothing is vendored** |
| QAT surface | `torchao/quantization/qat/` (10 modules); public API in `torchao/quantization/qat/__init__.py:35-67` |
| Docs recipe | `docs/source/workflows/qat.md:73-97` (prepare → train → convert); in-repo README is a 3-line stub pointing at `https://pytorch.org/ao/main/workflows/qat.html` |

## 3. Reimplemented vs imported vs adapted

**Imported as a pinned dependency** (`torchao==0.18.0`), for **one arm only**: the full-model QAT
reference in the LR-QAT design. Rationale: BSD-3-Clause makes a wheel dependency unconditionally safe
(no source redistribution, no notice obligation beyond pip metadata), and vendoring would fork numerics
that upstream's own tests define. The alternative — reimplementing the full-QAT arm too — would mean
this project wrote both sides of its own comparison, which is exactly the risk D5 exists to remove.

**Not used from torchao**: the int4 *convert* targets (`Int4Tensor`, `Int4PreshuffledTensor`), which
require the CUDA-index package `mslk` (`int4_tensor.py:139-140`); the MX/NVFP4 prototypes (Triton/CUDA);
and any tensor-subclass packing. The arm uses only the CPU-supported `prepare`/fake-quant surface.

Recipe actually used (the documented CPU-eager path, `docs/source/workflows/qat.md:73-97`):

```python
from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig, IntxFakeQuantizeConfig

weight_cfg = IntxFakeQuantizeConfig(
    torch.int4, group_size=128, is_symmetric=True
)  # paper's W4 g128
quantize_(
    model, QATConfig(weight_config=weight_cfg, step="prepare")
)  # nn.Linear -> FakeQuantizedLinear
# ... train all linear weights with AdamW ...
quantize_(model, QATConfig(step="convert"))  # swap back to nn.Linear; no PTQ base config
```

The convert step deliberately omits a `base_config`: converting *with* `Int4WeightOnlyConfig` is
CUDA/mslk-only here, whereas converting without one is a pure CPU module swap
(`qat/api.py:288`, proven by upstream test `test_qat.py:2548-2574`). Storage/quality claims therefore
come from *this project's* quantizer and containers, never from torchao's convert path.

Fake-quant arithmetic it applies (`torchao/quantization/quant_primitives.py:512-514, 888-891`):
`q = clamp(round(x/scale) + zero_point, qmin, qmax)`, `x̂ = (q − zero_point)·scale`, with
`_Round`/`_ClampSTE` straight-through estimators (`:215-260`) — i.e. the same class-2 mathematics as
LR-QAT's Eq. (1), which is why it is a valid reference arm.

## 4. Model / data / preprocessing / seed / hyperparameters

| Item | Value |
|---|---|
| Role | **full-model QAT** arm of the LR-QAT reproduction (the paper's upper reference), not a method under test |
| Model / data | identical to every other arm: Tier-1 pilot `HuggingFaceTB/SmolLM2-135M` and Tier-2 primary `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (both Apache-2.0) on `Salesforce/wikitext` `wikitext-2-raw-v1`, 512-token blocks, CLOUD-RUN |
| Quantization | int4 weight-only, symmetric, **group size 128** (matches LR-QAT's W4 g128 setting) |
| Trainable | **all linear weights** (this is the definition of the arm); embeddings / LM head / RMSNorm excluded, matching the paper's `FP_head_embd_norm` convention |
| Optimizer / LR | AdamW, `β = (0.9, 0.95)`, `lr_W` searched over `{1, 5, 10, 50, 100}·1e-5` (LR-QAT Table B1), weight decay 0.1 as the paper uses for full-model W |
| Schedule / budget | linear warmup (10 %) + linear decay, 1000 steps × 512 tokens, grad-clip 1.0 |
| Seeds | `{0,1,2,3,4}` |
| Verification before use | the arm must reproduce a *known* fake-quant invariant before it counts: `FakeQuantizedLinear` weights after `prepare` must equal our own `fake_quantize(W, QuantSpec(4, "per_group", 128, True))` within the class-2 tolerance of `reproduction-plan.md` §5.3, else the arm is dropped and the deviation logged |

## 5. Training and evaluation compute

**Upstream**: torchao declares **no torch requirement** in `pyproject.toml`/`setup.py`; it expects
torch ≥ 2.11 for prebuilt `.so` files (`torchao/__init__.py:69-79`) and CI pins `torch==2.13.0`. The
pure-Python wheel has no compiled extension, so it installs and imports on this host
(`torch 2.14.1+cpu`).

**On this workstation (verified by the D-lite probe, not by this card)**:
`IntxWeightOnlyConfig(torch.int4, PerGroup(32))` forwards on CPU with no warnings;
`Int8WeightOnlyConfig`, `Int8DynActInt8WeightConfig` also work; `Int4WeightOnlyConfig` (int4 tinygemm)
fails with `Requires mslk >= 1.0.0`; and upstream's own `test/quantization/test_qat.py` runs on CPU CI
runners (`regression_test.yml:34-43,105-108`), with ungated CPU tests at
`test_qat.py:945-1022, 1177-1190, 1280-1285, 1943-1992`. Source: `docs/research/backend-capability.md`
§2.3–§2.4, §3 rows 2–3.

**Cost in this reproduction**: the arm is **CLOUD-RUN** like every other training arm
(`AGENTS.md` §2b) — identical to the `LRQAT` arm at each model size (same model, same step budget):
≈0.5–1 cloud GPU-h at the Tier-1 pilot (135 M, 5 seeds) and ≈1.5–3 cloud GPU-h at the Tier-2 primary
(1.1 B, 3 seeds), plus the optimizer-state overhead of training all weights (at 1.1 B this requires
bf16 weights + 8-bit AdamW + gradient checkpointing to fit a 16 GB T4; an OOM is recorded as a failure,
DEV-0012). Measurement classes available to it: 1, 2, 3 and 4-CPU (the last only locally, on containers
we serialize ourselves — `environment.md` §1 correction). Class 4-GPU/class 5 remain unavailable until a
cloud GPU run actually executes a supported low-bit kernel.

## 6. Known deviations from the paper

This artifact has no paper protocol of its own, so "deviations from the paper" means deviations from
**LR-QAT's** full-model-QAT baseline definition (`reproduction-plan.md` §7, entries DEV-0001/0002/0003/
0005/0009 apply verbatim):

1. **Different implementation of the same arm.** LR-QAT's own full-model QAT is their `train-quantized`
   path with all weights trainable; ours is torchao's `prepare` path plus our training loop. The arm is
   therefore an *implementation-independent* reference, not the paper's code.
2. **Convert path differs**: we convert without a PTQ base config (CPU-supported) instead of to
   `Int4WeightOnlyConfig` (CUDA/mslk-only), so the arm's *stored* representation comes from our own
   container writer, not from torchao.
3. **Version**: dependency on the released `0.18.0` while the pinned provenance commit is post-release
   (`0.19.0` dev). Any API difference that matters must be resolved in favour of the released wheel and
   recorded.
4. **Not a benchmarked target upstream**: TorchAO's hardware matrix does not list CPU as a supported
   target for intx weight-only; the CPU path works (probe-verified) but carries no upstream performance
   guarantee — which is irrelevant here because no performance claim is made from this arm.
5. **Quality numbers are ours, not torchao's**: torchao publishes no WikiText-2/135M QAT result.

## 7. Expected output (what is expected, and what the *papers* report for the same arm)

Because this is a library artifact, the "expected output" is a *role*: the arm supplies the upper
reference in the LR-QAT trend T1. What the LR-QAT paper reports for its own full-model QAT baseline:

| Claim | Reference | Published value (LLaMA-2 7B) |
|---|---|---|
| Full-model QAT (LSQ) at W4 pc | Table 3 | WT2 **5.77 ± 0.02**, zero-shot 68.96 ± 0.29, peak memory 62.2 GB (98.5 GB without checkpointing) |
| Full-model QAT at W3 pc | Table 3 | WT2 **6.14 ± 0.01**, zero-shot 67.14 ± 0.13 |
| Cost per 100 steps | Table 3 / A8 | **3248 ± 7 s** (4×8 grad-accumulation); OOM without checkpointing or optimizer offloading at 1×32 |
| Full-model QAT is *worse than* LR-QAT on all four metrics | Table 3 | 5.77 vs **5.66** ppl; 68.96 vs **69.72** zero-shot |

The expected outcome in this project's terms is defined by the predeclared criteria in
`reproduction-plan.md` §5.2 (the LR-QAT arm must not be materially worse than this one), not by any
number above.

## 8. Observed

**`observed: not yet run (Milestone 3 pending)`**

No reproduction has been performed. The probe results quoted in §5 belong to the D-lite backend
capability stream (`docs/research/backend-capability.md`) and are cited, not re-run here. No number in
this card is a result of this project.
