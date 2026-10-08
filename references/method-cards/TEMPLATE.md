# Method Card — TEMPLATE

> Copy this file to `references/method-cards/<method-key>.md` and fill **every** field. A method card
> is required for every external method we reproduce, reimplement, or import (see
> `docs/research/upstream-lockfile.md`). One card per method *and* per upstream commit; if the
> upstream commit changes, create a new card and mark the old one superseded.
>
> Rules: never leave a field empty — write `N/A` or `UNKNOWN (reason)` explicitly. Never claim a
> result you did not observe: put expected values under *Expected* and observed values under
> *Observed*, and leave *Observed* as `NOT RUN` until it is measured. Every number carries its
> measurement class (`AGENTS.md` §5).

---

## Identity

- **Method name / key:** `<e.g. gptq>`
- **Paper citation key (BibTeX):** `<key from references/bibliography.bib>`
- **Paper title + arXiv id / venue:** `<...>`
- **Upstream repository:** `<owner/repo>`
- **Pinned commit SHA (verified):** `<40 hex; from gh api .../commits/main>`
- **License (SPDX, verified):** `<...>`
- **Implementation:** `reimplemented` | `imported (unmodified)` | `imported (patched — describe)`
- **Card author / date (UTC):** `<agent-id> / YYYY-MM-DDThh:mmZ`
- **Supersedes / superseded by:** `<card path or ->`

## Scope and semantics

- **What the method does (one paragraph, no adjectives):** `<...>`
- **Problem setting (PTQ / QAT / finetuning / training-from-scratch):** `<...>`
- **Compression object (weights / activations / KV cache / adapters):** `<...>`
- **Quantization type (granularity, symmetry, bit width, group size):** `<...>`
- **Rank treatment (rank, allocation rule, or none):** `<...>`
- **Objective actually optimized:** `<...>`
- **What is NOT included (explicitly out of scope for this card):** `<...>`

## Faithfulness to upstream

- **Bits of the method implemented exactly vs approximated:** `<...>`
- **Deviation 1 (what, why, expected numerical effect):** `<...>`
- **Deviation 2:** `<...>`
- **Upstream code paths we deliberately did not port:** `<...>` (e.g. CUDA-only kernels — deferred)
- **Cross-check performed:** `<e.g. reproduced paper's tiny-config result; matched hand-computed
  fixture; reproduced upstream unit test>` — with the command and the file the evidence lives in.

## Run configuration

- **Model / checkpoint (id + revision):** `<...>`
- **Dataset (id + revision/hash) and split:** `<...>`
- **Calibration set (source split, size, sampling seed, hash):** `<...>`
- **Seed(s):** `<list>`
- **Hyperparameters (full):** `<lr, batch/seq, steps, optimizer, group size, rank, fake-quant scheme, ...>`
- **Precision of execution (fp32/fp16/bf16; CPU threads):** `<...>`
- **Config file / command:** `<path + exact command>`
- **Commit SHA + dirty flag of this repo:** `<...>`

## Compute

- **Tier (charter §7):** `<0 | 1 | 2 | 3 | 4 | 5>`
- **Hardware actually used:** `<CPU model / GPU id; RAM; no CUDA flag>`
- **Wall-clock:** `<measured>`
- **Peak memory:** `<measured>`
- **Measurement classes emitted:** `<1, 2, 3 only locally; 4–5 not measured>`

## Measurement and accounting

- **Training-time memory (class):** `<value + class>`
- **Stored checkpoint bytes (class 3, measured):** `<value>`
- **Analytical estimate (class 1):** `<value>`
- **Relative accounting error (|3−1|/1):** `<value>`
- **Deployed inference representation:** `<description; class 4/5 => "not measured">`
- **Equal-stored-bytes counterpart used for comparison:** `<how the baseline was matched>`

## Results

| Quantity | Expected (from the paper) | Observed (ours) | Class | Notes |
|---|---|---|---|---|
| `<e.g. WikiText-2 perplexity>` | `<paper's number + which config>` | `NOT RUN` | `<2>` | `<...>` |
| `<e.g. stored MB>` | `<...>` | `NOT RUN` | `<3>` | `<...>` |

- **Agreement with the paper:** `<within tolerance / divergent — explain>`
- **Failures observed (divergence, NaN, OOM, edge cases):** `<...>`

## Known deviations and limitations

- **Deviations from the paper's protocol:** `<...>`
- **Reasons they matter / reasons they are acceptable:** `<...>`
- **What would falsify our reproduction of this method:** `<...>`
- **Known upstream issues (open bugs, undocumented defaults):** `<...>`

## Provenance

- **Evidence files (paths):** `<manifests, logs, artifacts, result registry entries>`
- **Commands to reproduce:** `<...>`
- **Related risk-register ids:** `<e.g. T3, R2>`
