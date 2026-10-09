# Dataset card — WikiText-2 (document level)

> Pinned in `configs/tier1/smollm2_135m.yaml` and `configs/repro/lr_qat_smollm2_135m.yaml`. Card
> author: M10 release slice. Facts quoted from `docs/research/model-dataset-licenses.md` §2 (row D3)
> and §3, whose values were obtained from live Hugging Face API calls on 2026-10-08.

## Identity

- **Dataset id:** `EleutherAI/wikitext_document_level`, config `wikitext-2-raw-v1`
- **Role in SpectraQuant:** development (validation) and test-perplexity split for the Tier-1 pilot
  and the M3 reproduction plan.
- **Revision (pin this; pass as `revision=`):** `647234772b9554e208af6c826f23b99e3cac88c8`
- **License (live API):** `cc-by-sa-3.0` (the upstream `Salesforce/wikitext` repo additionally
  carries `gfdl`)
- **Size:** test split 62 rows / 1 290 775 B in memory; train 629 rows
- **Why this variant:** it is the *exact* source the lm-evaluation-harness `wikitext` task reads,
  so the harness comparison is apples-to-apples.

## Intended use

- Perplexity / NLL of a compressed vs uncompressed model, on the cloud notebook substrate, following
  the frozen protocol (`preregistration.md` §3): calibration is drawn only from the training split
  and is disjoint from dev and test; the test split is read **exactly 2 times per arm** (one
  confirmatory read plus at most one recorded re-read after a bug fix).

## Out-of-scope use

- No local training or evaluation. The dataset is **not downloaded by the local test suite**; local
  test fixtures use the deterministic synthetic corpus (`configs/data/synthetic.yaml`).
- Not redistributed: only aggregate metrics are published. If sample outputs containing eval text
  were ever published they would inherit CC BY-SA 3.0 / GFDL terms and must be marked as such.

## Status

**NOT YET EVALUATED AT THIS SCALE.** No SpectraQuant number has been computed on this dataset: the
Tier-1 cells that use it are **NOT RUN** (no validated `run_manifest.json`, no checksum-validated
artifacts). The `LOCAL-FIXTURE` results in this repository use the synthetic LCG corpus, not
WikiText-2. No WikiText-2 perplexity may be quoted from this repository.

## Limitations

- Share-alike / GFDL terms constrain redistribution of derivatives; we avoid redistribution entirely.
- The dataset is Wikipedia text and overlaps the C4 calibration corpus; the mandatory 13-gram
  train/eval overlap audit (`docs/protocols/eval-protocol.md` §3) must run before any Tier-1 result
  counts.
- Document-level segmentation differs from the raw word-level variant; quote the config and revision
  with any number.

## Provenance

- Pin source: `docs/research/model-dataset-licenses.md` §2 (D3), §5.
- Configs: `configs/tier1/smollm2_135m.yaml`, `configs/repro/lr_qat_smollm2_135m.yaml`.
- Protocol: `docs/protocols/eval-protocol.md` §3.
