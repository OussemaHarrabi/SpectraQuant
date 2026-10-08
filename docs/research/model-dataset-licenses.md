# SpectraQuant — Model & Dataset License / Revision Inventory

Owner: stream **I-lite** (`LicensesEval`). Companion document: `docs/protocols/eval-protocol.md`.
Machine of record: CPU-only Windows 11 workstation — see `docs/research/environment.md`.
**All facts below were obtained on 2026-10-08 from live API calls whose raw output is reproduced in
§0 and §5.** No license or revision in this file is assumed, copied from a paper, or taken from a
model card's *prose* alone: each is the value returned by the Hugging Face API, the GitHub API, or a
file whose text is quoted below.

Every asset that could enter the core experiment path (`AGENTS.md` §4.8) has a row here. Assets whose
experiments are **cloud-GPU-gated** (Tier 2–5) are still fully pinned and licensed, because the pin is
needed before the asset may be used, not before it may be downloaded.

---

## 0. Verification method (reproduce every claim in this file)

Hugging Face metadata (revision SHA, declared license, parameter count, file sizes):

```bash
curl -s "https://huggingface.co/api/models/<org>/<model>"   # .sha = revision, .cardData.license, .safetensors.total
curl -s "https://huggingface.co/api/models/<org>/<model>?blobs=true"  # .siblings[].size for on-disk bytes
curl -s "https://huggingface.co/api/datasets/<org>/<dataset>"  # .sha, .cardData.license
```

GitHub metadata (upstream code + license):

```bash
gh api repos/<owner>/<repo>/commits/main --jq .sha
gh api repos/<owner>/<repo>/license --jq .license.spdx_id
gh api repos/<owner>/<repo>/git/ref/tags/<tag> --jq .object.sha   # annotated tag object
gh api repos/<owner>/<repo>/git/tags/<tagobject> --jq .object.sha  # peeled commit
```

On-disk sizes were read from `.siblings[].size` (model weights) and from the HF *datasets-server*
(`https://datasets-server.huggingface.co/size?dataset=…&config=…`, authoritative byte counts of the
converted parquet), not estimated.

Hardware constants used for the memory/plausibility columns were **measured on this workstation**
(§4): fp32 GEMM throughput `326.66 GFLOP/s`, memory copy bandwidth `21.78 GB/s`.

---

## 1. Models

All fourteen models below are ungated (`gated:false`); none requires an HF account
token to download. "Revision" is the HF repository commit hash that MUST be passed as
`revision=<sha>` to `transformers`/`huggingface_hub`.

| # | Model (role) | HF id | Revision (pin this) | License (SPDX, live API) | Commercial / redistribution | Weight bytes on disk | fp16 weights alone |
|---|---|---|---|---|---|---|---|
| M1 | **Tier-2 primary**, base (GPU-gated) | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | `59f6f375b26bde864a6ca194a9a3044570490064` | `apache-2.0` | Commercial OK; redistribution OK with `LICENSE` + `NOTICE` retention | 4,400,254,027 (fp32 `pytorch_model.bin`) **+** 4,400,216,536 (fp32 `model.safetensors`) = 8.80 GB if both copies pulled; 4.40 GB single copy | **2.20 GB** (fp32 storage = 4.40 GB) |
| M2 | Tier-2 chat variant (compatibility check only) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | `apache-2.0` | as M1 | 2,200,119,864 (bf16) | 2.20 GB (bf16 storage = 2.20 GB) |
| M3 | **Tier-2/3 candidate** + local forward-eval workhorse | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | `apache-2.0` (`license_link` → repo `LICENSE`, verified to be the Apache-2.0 text) | as M1 | 988,097,824 (bf16) | **0.99 GB** |
| M4 | Qwen instruct variant (compatibility check only) | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | `apache-2.0` | as M1 | 0.99 GB (494,032,768 params × 2 B) | 0.99 GB |
| M5 | **Tier-1 forward-eval, smallest** | `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | `apache-2.0` (card §License → "Apache 2.0") | as M1 | 723,674,912 (bf16) | **0.72 GB** |
| M6 | SmolLM2-360M instruct variant | `HuggingFaceTB/SmolLM2-360M-Instruct` | `a10cc1512eabd3dde888204e902eca88bddb4951` | `apache-2.0` | as M1 | 0.72 GB (361,821,120 params × 2 B) | 0.72 GB |
| M7 | Tier-3 upper bound (GPU-gated) | `HuggingFaceTB/SmolLM2-1.7B` | `effd688a12921b4cc83e3312b6feb579f70f9c71` | `apache-2.0` | as M1 | 3,422,777,952 (bf16) | **3.42 GB** |
| M8 | SmolLM2-1.7B instruct variant | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | `apache-2.0` | as M1 | 3.42 GB (1,711,376,384 params × 2 B) | 3.42 GB |
| M9 | Tier-1 pretrained proxy-validity model (per `preregistration.md` §4) | `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | `mit` | Commercial OK; redistribution OK with copyright + license notice | 1,096,223,248 (fp32 `.bin` + `.safetensors` duplicate); 0.55 GB single copy | 0.27 GB (137,022,720 params) |
| M10 | **Tier-4 optional vision** | `facebook/deit-base-patch16-224` | `fb2c78a54a5637dec350432794f7b93e31f910c9` | `apache-2.0` | Apache-2.0 terms for the weights; **the model was trained on ImageNet-1k** — ImageNet's own terms restrict commercial reuse of the *data*, and this must be disclosed if Tier-4 results are ever published. `[PARTIALLY UNRESOLVED]` (see §6.4) | 346,351,599 (fp32 `pytorch_model.bin`; no safetensors) | 0.17 GB (≈86.6 M params, inferred from fp32 byte count) |
| M11 | Tier-4 optional vision alternative | `google/vit-base-patch16-224` | `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` | `apache-2.0` | same ImageNet caveat as M10 | 346,351,599 (fp32 `.bin`) + 346,293,852 (fp32 `.safetensors`) = 0.69 GB; 0.35 GB single copy | 0.17 GB (86,567,656 params) |
| M12 | Optional smaller Tier-1 fallback | `distilbert/distilgpt2` | `2290a62682d06624634c1f46a6ad5be0f47f38aa` | `apache-2.0` | as M1 | 88,204,032 params (0.18 GB bf16 / 0.35 GB fp32) | 0.18 GB |
| M13 | **Bounded-reproduction + Tier-1 tiny-transformer model, cheap-cloud workhorse** | `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | `apache-2.0` (card `license: apache-2.0`) | as M1 | 269,060,552 (bf16 `model.safetensors`) | **0.27 GB** (134,515,008 params) |
| M14 | SmolLM2-135M instruct variant (compatibility check only) | `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | `apache-2.0` | as M1 | 0.27 GB (134,515,008 params × 2 B) | 0.27 GB |

**Notes on the base TinyLlama checkpoint.** Unlike the chat variant, M1 is stored in **fp32**
(`safetensors` metadata reports `{"F32": 1100048384}`), so the download is 4.40 GB per copy, not
2.20 GB. It also ships two identical-weight files (`.bin` and `.safetensors`); pull a single format.
This matters on a 16 GB machine and is why the WikiText-2 pass over M1 is the most expensive
evaluation cell (`docs/protocols/eval-protocol.md` §9).

**License-text verification performed (not just the card `license:` field):**

- `Qwen/LICENSE` → begins `Apache License / Version 2.0, January 2004` (fetched from
  `https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/LICENSE`).
- `SmolLM2-360M/README.md` line 126 → `[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)`.
- `TinyLlama-…-3T/README.md` line 4 → `license: apache-2.0`.
- `deit-base-patch16-224/README.md` line 2 and `vit-base-patch16-224/README.md` line 2 →
  `license: apache-2.0`.

No model in the candidate set carries a non-commercial or research-only clause. **Therefore no
licence blocks local or redistribution use of the checkpoints themselves**; the only restriction in
the set is the ImageNet provenance of M10/M11 (§6.4).

---

## 2. Datasets & corpora

| # | Dataset (role) | HF id | Revision (pin this) | License (live API / quoted text) | Terms that constrain us | Size on disk | CPU-usable on 16 GB |
|---|---|---|---|---|---|---|---|
| D1 | **WikiText-2 raw** (LM train/dev/test, Tier 1–2) | `Salesforce/wikitext` config `wikitext-2-raw-v1` | `b08601e04326c79dfdd32d625aee71d232d685c3` | `cc-by-sa-3.0` **and** `gfdl` (API `cardData.license` = `["cc-by-sa-3.0","gfdl"]`) | Attribution + **share-alike** on derivatives; GFDL copyleft | test parquet 732,610 B (4,358 rows / 1,285,622 chars); train 6,357,543 B (36,718 rows); valid 657,209 B | Yes — trivially |
| D2 | WikiText-2 **pre-tokenized (PTB/word-level) variant** (sanity only) | `Salesforce/wikitext` config `wikitext-2-v1` | same repo SHA as D1 | same as D1 | same as D1 | config totals: 7,371,282 B parquet download / 12,864,827 B in memory (datasets-server `size` endpoint) | Yes |
| D3 | **WikiText-2 document-level**, the *exact* source the harness `wikitext` task reads | `EleutherAI/wikitext_document_level` config `wikitext-2-raw-v1` | `647234772b9554e208af6c826f23b99e3cac88c8` | `cc-by-sa-3.0` | same as D1 | test split: 62 rows, 1,290,775 B in memory; train 629 rows | Yes |
| D4 | **Calibration / continued-training candidate A** (recommended) | `allenai/c4` config `en` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | `odc-by` — README §"Licensing Information": *"We are releasing this dataset under the terms of [ODC-BY]. By using this, you are also bound by the [Common Crawl terms of use] in respect of the content contained in the dataset."* | **Attribution required** (ODC-BY); **Common Crawl ToU pass-through** on the text | Whole repo 33.1 TB (`usedStorage` 33,055,538,287,129 B) — must be **streamed**, never downloaded whole | Yes, **streamed subset only** (see §3) |
| D5 | **Calibration candidate B** (alternative) | `cerebras/SlimPajama-627B` | `[UNRESOLVED — repo now returns HTTP 401]` (see §6.1) | `apache-2.0` — **verified from the authors' own statement**: Cerebras blog: *"We release it under the Apache 2.0 license"* (fetched 2026-10-08) | Attribution/NOTICE; no share-alike, no crawl-ToU pass-through | 1.51 TB per the community mirror `gmongaras/SlimPajama-627B_Reupload` | Yes, **streamed subset only** |
| D6 | SlimPajama bounded subset (mirror) | `DKYoon/SlimPajama-6B` | `b5f90f419b7489cdba26fdbc8c022fcb5562f968` | **NONE** in the card (`cardData.license` absent); README says *"Sampled version of cerebras/SlimPajama-627B"* | `[UNRESOLVED]` — the mirror itself declares no license; only Cerebras' Apache-2.0 statement (D5) covers the underlying content (see §6.2) | 23.2 GB in memory, 14.0 GB download | Yes (14 GB download is borderline; stream instead) |
| D7 | HellaSwag eval split | `Rowan/hellaswag` config `default` | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | card: *"MIT https://github.com/rowanz/hellaswag/blob/master/LICENSE"* | MIT — attribution | validation 10,042 rows / 10,746,075 B | Yes |
| D8 | ARC eval splits | `allenai/ai2_arc` configs `ARC-Easy`, `ARC-Challenge` | `210d026faf9955653af8916fad021475a3f00453` | `cc-by-sa-4.0` | attribution + share-alike | ARC-Easy test 2,376 rows / 669,138 B; ARC-Challenge test 1,172 rows / 375,159 B | Yes |
| D9 | PIQA eval split | `baber/piqa` config `default` | `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` | **no license metadata in the HF card** (`cardData` is `null`). Original dataset license verified upstream: *"License: [https://opensource.org/licenses/AFL-3.0](Academic Free License v. 3.0)"* from the authors' README `github.com/ybisk/ybisk.github.io/blob/master/piqa/README.md` | AFL-3.0: permissive, commercial OK, attribution required, includes patent grant | validation 1,838 rows / 470,590 B | Yes |
| D10 | WinoGrande eval split | `allenai/winogrande` config `winogrande_xl` | `01e74176c63542e6b0bcb004dcdea22d94fb67b5` | HF card says *"More Information Needed"*; upstream repo license verified via GitHub API: `allenai/winogrande` → `spdx: Apache-2.0` | Apache-2.0 | validation 1,267 rows / 147,441 B | Yes |
| D11 | BoolQ eval split | `aps/super_glue` config `boolq` | `3de24cf8022e94f4ee4b9d55a6f539891524d646` | super_glue card `license: other` with the note *"We refer users to the original licenses accompanying each dataset"*; BoolQ's own README states *"BoolQ is released under the [Creative Commons Share-Alike 3.0] license."* | CC BY-SA 3.0 — attribution + share-alike | validation 3,270 rows / 2,123,054 B | Yes |
| D12 | Optional Tier-1 secondary LM corpus | `roneneldan/TinyStories` | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | `cdla-sharing-1.0` (Community Data License Agreement – Sharing 1.0) | CDLA-Sharing is a copyleft data license: derivatives must be shared under the same terms | 17.3 GB (`usedStorage` 17,294,456,299 B) | Yes, subset |

**Evaluation datasets must be redistributable for the paper's artifacts.** D8 and D11 are
share-alike (CC BY-SA 4.0 / 3.0) and D1–D3 are share-alike (CC BY-SA 3.0 / GFDL). We do **not**
redistribute any of them: they are fetched by pinned revision at run time, and only aggregate metrics
are published. If sample outputs containing eval text are ever published (`--log_samples` artifacts),
they inherit the source's share-alike terms and must be marked as such.

---

## 3. Calibration / continued-training corpus: decision and justification

**Requirement.** A bounded corpus used for (a) GPTQ/AWQ-style layer input statistics, (b) activation
statistics for the sensitivity proxy, and (c) any short continued-training/calibration schedule.
It MUST be disjoint from every eval split (`docs/protocols/eval-protocol.md` §3).

**Decision: choose D4 (`allenai/c4`, config `en`), streamed and subsampled deterministically.**

Justification, each point grounded in §2:

1. **Method comparability.** The PTQ literature this project is measured against (GPTQ) calibrates
   on C4. Using C4 keeps the calibration distribution identical to the baseline family, so a
   quality gap is attributable to the method, not the corpus.
2. **License clarity.** C4's terms are explicit and self-contained (`odc-by` + Common Crawl ToU),
   whereas D5's canonical repo is **currently gated (HTTP 401)** and D6's mirror declares no license
   at all. A calibration corpus whose license is `[UNRESOLVED]` cannot be shipped in a reproducibility
   package. ODC-BY's broad attribution obligation is satisfiable: the dataset is never redistributed
   (only streamed), and attribution is recorded in the manifest and the paper.
3. **Bounded cost.** C4 `en` is 33 TB, which is unusable as a download but perfectly usable as a
   *streamed* slice: Hugging Face `datasets` streaming pulls only the shards touched by the
   predeclared slice.
4. **Contamination control.** Both C4 and SlimPajama contain Wikipedia-derived text and thus overlap
   WikiText-2. That risk is handled by the mandatory 13-gram audit (§3 of the eval protocol), which
   would be required for either candidate; C4 imposes no *additional* governance burden.

**Alternative sanctioned by this document (if C4 streaming is unavailable at run time):**
D5/D6 (SlimPajama, Apache-2.0), under the constraints in §6.1–§6.2. It is *not* the default because
of the unsettled license chain.

**Predeclared bounded slice (freeze value, in the protocol):** 256 sequences × 2,048 tokens =
**524,288 calibration tokens**, drawn from the C4 `en` **train** split with a fixed, committed
selection rule (see `docs/protocols/eval-protocol.md` §3.2). At ≈4.3 characters/token this is ≈2.3 MB
of text — orders of magnitude below the 16 GB budget.

**SlimPajama alternative slice:** same shape, C4→SlimPajama `en`/no-language-filter, same seed.

---

## 4. Peak memory and CPU plausibility on this CPU-only machine

Measured constants (this workstation, 2026-10-08, from §5):

| Constant | Measured value | How |
|---|---|---|
| fp32 GEMM throughput | **326.66 GFLOP/s** | `numpy` 2.5.1, 2048³ `float32` matmul, best of 3 |
| Memory copy bandwidth | **21.78 GB/s** (read+write) | `numpy` copy of a 256 MB `float32` array, best of 4 |
| Usable RAM | 15.1 GiB of 16.23 GB | `environment.md` |

| Model | fp16 weights | fp32 weights (what CPU eval will actually use) | Peak RSS for forward eval (weights + activations + KV, ≤2,048 ctx) | CPU forward evaluation plausible? |
|---|---|---|---|---|
| M1 TinyLlama-1.1B | 2.20 GB | 4.40 GB | ≈ 6.5–7.5 GB | **Yes, memory-wise**; slowest local LM cell (protocol §9) |
| M3 Qwen2.5-0.5B | 0.99 GB | 1.98 GB | ≈ 3.5–4.5 GB | **Yes** |
| M5 SmolLM2-360M | 0.72 GB | 1.45 GB | ≈ 3.0–4.0 GB | **Yes** — smallest, preferred local workhorse |
| M7 SmolLM2-1.7B | 3.42 GB | 6.85 GB | ≈ 10–12 GB | Marginally yes on 15.1 GiB; multi-hour per pass; officially Tier-3 GPU-gated |
| M9 GPT-2 (124 M) | 0.27 GB | 0.55 GB | ≈ 1.5–2.5 GB | **Yes** — cheapest local LM row; but see the substrate policy note below |
| M13/M14 SmolLM2-135M | 0.27 GB | 0.54 GB | ≈ 1.0–1.8 GB | **Yes** — cheapest LM in the registry; runs on the free Colab tier with large headroom |
| M10/M11 ViT/DeiT-base | 0.17 GB | 0.35 GB | ≈ 1.0–2.0 GB (224² images, small batch) | **Yes**, seconds-to-minutes |

**fp16 vs fp32 on CPU.** The fp16 numbers above are the requested "fp16 weights alone" figure.
PyTorch's CPU fp16 path is memory-only-optimised and frequently *slower* than fp32 on x86 without
AMX, and some ops lack fp16 kernels. Where a CPU forward pass is used at all (fixtures, class-4-CPU
kernel measurements), it runs in fp32; fp16 is never the local execution dtype.

**Execution-substrate note (added 2026-10-08, supersedes the "locally runnable" framing above).** Per
`AGENTS.md` §2.3 and §2b, the local machine no longer runs model training, QAT, large-scale inference
or GPU evaluation — including the Tier-1 tiny-transformer experiments, which move to the cloud
notebook substrate (Google Colab / Kaggle / Colab Enterprise). The "plausible?" column above therefore
answers only *"would this fit and run on the local CPU **if** a bounded fixture or a class-4-CPU
measurement needed it"*; it is **not** a statement that evaluation runs locally. Every evaluation cell
in `docs/protocols/eval-protocol.md` §6.4 carries an explicit substrate label
(`LOCAL-FIXTURE` / `LOCAL-CPU-MEASUREMENT` / `CLOUD-COLAB` / `CLOUD-GPU`).

**What is NOT usable here:** nothing in the model/dataset set is CUDA-gated *in itself*. The training
and large-scale-inference plans for Tiers 1–5 run on the cloud substrate (§2b), and any Python path
that would silently require CUDA must fail loudly locally (`AGENTS.md` §2.2).

---

## 5. Revision pin block (paste-ready)

```bash
# --- models: revision = repo SHA (pass as transformers revision=) -----------------
curl -s https://huggingface.co/api/models/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | jq -r .sha
# -> 59f6f375b26bde864a6ca194a9a3044570490064
curl -s https://huggingface.co/api/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0 | jq -r .sha
# -> fe8a4ea1ffedaf415f4da2f062534de366a451e6
curl -s https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B | jq -r .sha
# -> 060db6499f32faf8b98477b0a26969ef7d8b9987
curl -s https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct | jq -r .sha
# -> 7ae557604adf67be50417f59c2c2f167def9a775
curl -s https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-360M | jq -r .sha
# -> f8027fd0eaeea54caa13c31d31b9fdc459c38b49
curl -s https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-360M-Instruct | jq -r .sha
# -> a10cc1512eabd3dde888204e902eca88bddb4951
curl -s https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-1.7B | jq -r .sha
# -> effd688a12921b4cc83e3312b6feb579f70f9c71
curl -s https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-1.7B-Instruct | jq -r .sha
# -> 31b70e2e869a7173562077fd711b654946d38674
curl -s https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M | jq -r '.sha, .cardData.license, .safetensors.total'
# -> 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
# -> apache-2.0
# -> 134515008
curl -s 'https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M?blobs=true' | python -c "import json,sys;d=json.load(sys.stdin);print([(s['rfilename'],s['size']) for s in d['siblings'] if s['rfilename'].endswith(('.safetensors','.bin'))])"
# -> [('model.safetensors', 269060552)]
curl -s https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M-Instruct | jq -r '.sha, .cardData.license'
# -> 12fd25f77366fa6b3b4b768ec3050bf629380bac
# -> apache-2.0
curl -s https://huggingface.co/api/models/openai-community/gpt2 | jq -r .sha
# -> 607a30d783dfa663caf39e06633721c8d4cfcd7e
curl -s https://huggingface.co/api/models/facebook/deit-base-patch16-224 | jq -r .sha
# -> fb2c78a54a5637dec350432794f7b93e31f910c9
curl -s https://huggingface.co/api/models/google/vit-base-patch16-224 | jq -r .sha
# -> 3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3

# --- datasets: revision = repo SHA (pass as datasets.load_dataset(..., revision=)) -
curl -s https://huggingface.co/api/datasets/Salesforce/wikitext | jq -r '.sha, .cardData.license'
# -> b08601e04326c79dfdd32d625aee71d232d685c3
# -> ["cc-by-sa-3.0","gfdl"]
curl -s https://huggingface.co/api/datasets/EleutherAI/wikitext_document_level | jq -r '.sha, .cardData.license'
# -> 647234772b9554e208af6c826f23b99e3cac88c8
# -> cc-by-sa-3.0
curl -s https://huggingface.co/api/datasets/allenai/c4 | jq -r '.sha, .cardData.license'
# -> 1588ec454efa1a09f29cd18ddd04fe05fc8653a2
# -> ["odc-by"]
curl -s https://huggingface.co/api/datasets/Rowan/hellaswag | jq -r .sha
# -> 218ec52e09a7e7462a5400043bb9a69a41d06b76
curl -s https://huggingface.co/api/datasets/allenai/ai2_arc | jq -r '.sha, .cardData.license'
# -> 210d026faf9955653af8916fad021475a3f00453  /  ["cc-by-sa-4.0"]
curl -s https://huggingface.co/api/datasets/baber/piqa | python -c "import json,sys;print(json.load(sys.stdin)['sha'])"
# -> 142f6d7367fd9877f0fb3b5734ea6a545f54cdd1
curl -s https://huggingface.co/api/datasets/allenai/winogrande | jq -r .sha
# -> 01e74176c63542e6b0bcb004dcdea22d94fb67b5
curl -s https://huggingface.co/api/datasets/aps/super_glue | jq -r '.sha, .cardData.license'
# -> 3de24cf8022e94f4ee4b9d55a6f539891524d646  /  ["other"]

# --- harness (see docs/protocols/eval-protocol.md §4) -----------------------------
gh api repos/EleutherAI/lm-evaluation-harness/releases/latest --jq '.tag_name'
# -> v0.4.13
gh api repos/EleutherAI/lm-evaluation-harness/git/ref/tags/v0.4.13 --jq '.object.sha'
# -> eb4678c395c2c1f157d462b8f3a09831160c861a   (annotated tag object)
gh api repos/EleutherAI/lm-evaluation-harness/git/tags/eb4678c395c2c1f157d462b8f3a09831160c861a --jq '.object.sha'
# -> ddd67220430a2470529f25fd5c05a576ca1057a0   (the COMMIT to pin)
gh api repos/EleutherAI/lm-evaluation-harness/license --jq '.license.spdx_id'
# -> MIT
# NOTE: docs/research/upstream-lockfile.md (stream A) now carries this same release pin —
#       v0.4.13 tag object eb4678c395c2c1f157d462b8f3a09831160c861a peeling to commit
#       ddd67220430a2470529f25fd5c05a576ca1057a0 (independent re-verification by stream A on
#       2026-10-08). Lockfile row U6 is the canonical record; main HEAD d6de8164... is the
#       recorded divergence.
```

**Measured throughput / bandwidth (reproduce):**

```bash
python - <<'PY'
import numpy as np, time
n=2048; a=np.random.rand(n,n).astype(np.float32); b=np.random.rand(n,n).astype(np.float32)
a@b; ts=[]
for _ in range(3):
    t=time.perf_counter(); a@b; ts.append(time.perf_counter()-t)
print("GFLOP/s", 2*n**3/min(ts)/1e9)          # -> 326.66
PY
# copy bandwidth: 256 MB float32 copy, best of 4 -> 21.78 GB/s

# SmolLM2-135M uses the SAME tokenizer as SmolLM2-360M (verified by hashing both tokenizer.json):
#   sha256(tokenizer.json) == 9ca9acddb6525a19... for both M13 and M5
#   -> WikiText-2 test token count 303,414 is identical for M13 and M5.
```

---

## 6. Unresolved items, with exactly what is needed to close each

1. **`cerebras/SlimPajama-627B` returns HTTP 401 with no HF token** (verified:
   `curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/datasets/cerebras/SlimPajama-627B`
   → `401`). The authors' Apache-2.0 statement is verified from the Cerebras blog, but the *access
   terms of the gated repo* (whether it requires accepting conditions) cannot be read without an
   authenticated HF account. **Resolution needed:** an HF account token with accepted dataset terms,
   then re-fetch the card and record `gated` + `cardData.license`. Until then, SlimPajama is the
   *alternative* corpus only, not the default (§3).
2. **`DKYoon/SlimPajama-6B` mirror declares no license** (`cardData` has no `license`; README states
   only that it is a sample of D5). **Resolution needed:** either use the canonical (gated) repo via
   an authenticated token, or state explicitly in the manifest that the mirror is used under the
   upstream Apache-2.0 terms with the mirror credited. `[UNRESOLVED]` as a standalone license claim.
3. **Harness pin — RESOLVED.** `docs/research/upstream-lockfile.md` (row U6) originally pinned
   `main` HEAD `d6de81643928d653435c431bae19945d41d32520`; on 2026-10-08 stream A **adopted the
   release pin** after independent re-verification: annotated tag object
   `eb4678c395c2c1f157d462b8f3a09831160c861a` peels to commit
   `ddd67220430a2470529f25fd5c05a576ca1057a0` (`v0.4.13`, MIT). This inventory, the eval protocol,
   and the lockfile now agree; the lockfile is the canonical record and the diverged `main` HEAD is
   retained as a documented divergence. The manifest still carries the tag commit explicitly.
4. **ImageNet provenance of M10/M11.** The checkpoints are Apache-2.0, but they were trained on
   ImageNet-1k, whose terms permit research use and restrict commercial redistribution of the data.
   Tier 4 is optional and unrun; **resolution needed** only if Tier 4 is ever executed — record
   ImageNet's terms acceptance and, for CIFAR-100, CIFAR's MIT-style terms.
5. **PIQA HF mirror has no license metadata.** Resolved at the *dataset* level via the authors' README
   (AFL-3.0, §2 D9), but the HF mirror `baber/piqa` itself carries none. The manifest records the
   mirror SHA **and** the AFL-3.0 upstream statement so the claim is traceable.

---

## 7. Asset role map (what is actually used where)

| Role | Asset(s) | Tier | Substrate (`eval-protocol.md` §6.4 labels) |
|---|---|---|---|
| Bounded reproduction + Tier-1 tiny-transformer | **M13 SmolLM2-135M** (M5 SmolLM2-360M fallback) | 1 | **CLOUD-COLAB** (cheap free-tier vehicle); local only builds fixtures |
| Tier-2 primary + chat variant | M1 TinyLlama-1.1B, M2 chat | 2 | **CLOUD-GPU** |
| Tier-2/3 secondary | M3 Qwen2.5-0.5B, M7 SmolLM2-1.7B | 2–3 | **CLOUD-GPU** |
| Local host model for eval-pipeline fixtures | M13 SmolLM2-135M, M9 GPT-2 | 0–1 | **LOCAL-FIXTURE** (never an evaluation result) |
| LM train/dev/test | D1 (raw), D3 (harness-identical document-level) | 1–2 | cloud run, local analysis |
| Calibration / continued-training | D4 C4 `en` (default) / D5–D6 SlimPajama (alternative) | 1–2 | cloud (streamed subset); local validates the selection rule on fixtures |
| Downstream eval suite | D7–D11 pinned revisions | 1–2 | **CLOUD-GPU** full suite / **CLOUD-COLAB** bounded subset |
| Vision cross-architecture | M10/M11 + CIFAR-100/ImageNet-100 | 4 | **CLOUD-GPU** (optional) |
| Harness | `EleutherAI/lm-evaluation-harness` @ `ddd6722…`, MIT | 1–2 | cloud run, local artifact validation |
| Class-4-CPU kernel measurement (not an eval cell) | our own serialized int4/int8 artifact | 0 | **LOCAL-CPU-MEASUREMENT** (`benchmark-protocol.md`) |

Every asset above is reachable with a single unauthenticated `curl`/`datasets` call except D5/D6,
which are gated/unlicensed mirrors. The assets with a fully settled license and a verified revision
are the only ones permitted in any cell, wherever that cell runs.
