# Dataset card — C4 (`en`) calibration / continued-training corpus

> Pinned in `configs/tier1/smollm2_135m.yaml` and `configs/repro/lr_qat_smollm2_135m.yaml`. Card
> author: M10 release slice. Facts quoted from `docs/research/model-dataset-licenses.md` §2 (row D4)
> and §3, whose values were obtained from live Hugging Face API calls on 2026-10-08.

## Identity

- **Dataset id:** `allenai/c4`, config `en`
- **Role in SpectraQuant:** training split and calibration source — GPTQ/AWQ-style layer input
  statistics, activation statistics for the sensitivity proxy, and any short continued-training /
  calibration schedule.
- **Revision (pin this; pass as `revision=`):** `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`
- **License (live API):** `odc-by` (attribution required) **plus** the Common Crawl terms of use pass
  through to the text.
- **Size:** whole repo 33.1 TB (`usedStorage` 33 055 538 287 129 B) — must be **streamed**, never
  downloaded whole.

## Intended use

- The **default** calibration corpus because the PTQ literature this project is measured against
  (GPTQ) calibrates on C4, keeping the calibration distribution identical to the baseline family.
- **Predeclared bounded slice (frozen):** 256 sequences × 2 048 tokens = **524 288 calibration
  tokens**, drawn from the C4 `en` **train** split with a fixed, committed selection rule
  (`docs/protocols/eval-protocol.md` §3.2: `sha256(text) mod 1000 < 1`, seed `20261008`; slice hash
  recorded in the run manifest).

## Out-of-scope use

- No local download. The dataset is **not fetched by the local test suite**; local fixtures use the
  deterministic synthetic corpus. The local machine only validates the selection rule on fixtures.
- Not redistributed: attribution is recorded in the manifest and the paper; only the streamed slice
  is touched.

## Status

**NOT YET EVALUATED AT THIS SCALE.** No SpectraQuant number has been computed on this dataset: the
Tier-1 cells that calibrate on it are **NOT RUN** (no validated `run_manifest.json`). The
`LOCAL-FIXTURE` results in this repository are synthetic. No calibration-activation statistic from C4
may be quoted from this repository.

## Limitations

- Wikipedia-derived text overlaps WikiText-2; the mandatory 13-gram train/eval overlap audit must run
  before any Tier-1 result counts (required for either calibration candidate).
- ODC-BY attribution plus Common Crawl ToU constrain reuse; the corpus must stay streamed, not
  redistributed.
- The license chain of the *alternative* corpus (SlimPajama) is unresolved
  (`docs/research/model-dataset-licenses.md` §6.1–§6.2); C4 was chosen partly to avoid an
  `[UNRESOLVED]` license in the reproducibility package.

## Provenance

- Pin source: `docs/research/model-dataset-licenses.md` §2 (D4), §3, §5.
- Configs: `configs/tier1/smollm2_135m.yaml`, `configs/repro/lr_qat_smollm2_135m.yaml`.
- Protocol: `docs/protocols/eval-protocol.md` §3.2.
