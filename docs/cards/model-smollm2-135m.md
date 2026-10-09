# Model card — HuggingFaceTB/SmolLM2-135M

> Pinned in `configs/tier1/smollm2_135m.yaml` and `configs/repro/lr_qat_smollm2_135m.yaml`. Card
> author: M10 release slice. Every fact below is quoted from those configs or from
> `docs/research/model-dataset-licenses.md` §1 (row M13), whose values were obtained from live
> Hugging Face API calls on 2026-10-08.

## Identity

- **Model id:** `HuggingFaceTB/SmolLM2-135M`
- **Role in SpectraQuant:** Tier-1 pilot model (bounded reproduction + proxy validation) and the
  reproduction host model for the M3 plan.
- **Revision (pin this; pass as `revision=`):** `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- **License (SPDX, live API):** `apache-2.0`
- **Parameters:** 134 515 008 (`model.safetensors`, 269 060 552 B on disk, bf16)
- **Tokenizer:** shared with SmolLM2-360M (`tokenizer.json` sha256 `9ca9acddb6525a19…` for both).
- **Gated:** no (`gated:false`); no HF account token required.

## Intended use

- The cheap, mandatory cloud pilot: validate the proxy ranking and the method arms before any Tier-2
  claim. It is the host model of the frozen M3 bounded-reproduction plan (LR-QAT / LoftQ semantics).
- Forward evaluation and short continued-training/calibration on the **cloud notebook substrate**
  (`CLOUD-COLAB`) only.

## Out-of-scope use

- **It is not used locally.** No training, QAT, large-scale inference or GPU evaluation runs on the
  development workstation (`AGENTS.md` §2.3/§2b). It is never downloaded by the local test suite.
- It is not a Tier-2/Tier-3 model (those are TinyLlama-1.1B-class and a second 0.6–1.7 B model).

## Status

**NOT YET EVALUATED AT THIS SCALE.** No SpectraQuant number exists for this model: the Tier-1
confirmatory cells are **NOT RUN** (no validated `run_manifest.json`, no checksum-validated
artifacts). All SpectraQuant evidence in this repository is `LOCAL-FIXTURE` (Tier 0) and class 1–3,
plus the class-4-CPU single-matrix fixture — none of it involves this checkpoint. No perplexity,
accuracy or compression number for SmolLM2-135M may be quoted from this repository.

## Limitations

- We have not measured this model's behaviour under any of our transforms; its published
  characteristics (from the upstream card) are not verified by us.
- The pinned revision is a specific HF commit; the `main` branch may move. A run must fail loudly
  rather than silently use an unpinned revision.
- Apache-2.0 covers the weights; the model was trained on undisclosed upstream data, so downstream
  evaluation datasets still require their own contamination audit and license review.

## Provenance

- Pin source: `docs/research/model-dataset-licenses.md` §1 (M13) and §7; live API commands in §0/§5.
- Configs: `configs/tier1/smollm2_135m.yaml`, `configs/repro/lr_qat_smollm2_135m.yaml`.
- Plan validation: `tests/unit/test_experiment_plan.py`.
