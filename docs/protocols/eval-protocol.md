# SpectraQuant — Frozen Evaluation Protocol

Owner: stream **I-lite** (`LicensesEval`). Companion: `docs/research/model-dataset-licenses.md`
(asset inventory), `docs/research/preregistration.md` (hypotheses, statistics, exclusion rules),
`docs/research/environment.md` (hardware), `AGENTS.md` (contract; §5 measurement classes, §2 compute
rules).

Status: **DRAFT — to be frozen (dated + commit SHA) by the orchestrator at the preregistration
freeze, before the first confirmatory run.** After freeze, any change to a value in this file is an
entry in `docs/research/preregistration-amendments.md`, timestamped, with a stated reason. The suite
was selected **before any method development**, deliberately, so that no method can be tuned against
the evaluation set.

**Binding rule (§8).** Every evaluation setting in this file is **byte-identical across all methods
and all arms**. A method evaluated with any deviation (different revision, dtype, few-shot count,
seed, prompt file, batch size, limit, or tokenizer) is **invalid** and its numbers are discarded.

---

## 1. What is evaluated

| Purpose | Metric | Where |
|---|---|---|
| Primary quality outcome P1 | per-token cross-entropy → **perplexity**, plus raw mean NLL | WikiText-2 document-level test split (§6.1) |
| Secondary S4 | zero-shot/few-shot accuracy on 6 multiple-choice tasks | harness suite (§6.2) |
| Compression outcome P2 | stored bytes of the serialized artifact (class 3) + class-1 analytical estimate | not a harness cell — see `docs/protocols/memory-accounting.md` |

Both P1 and S4 are always reported with the **equal-memory counterpart** of the baseline
(`AGENTS.md` §4.5), and with the measurement class of the executed arithmetic. Class naming follows
`docs/protocols/measurement-taxonomy.md` §1 exactly — the `class_name` strings used in manifests are
`analytical` (class 1), `fake_quant_quality` (class 2), `packed_storage` (class 3),
`kernel_backed_inference` (class 4), `end_to_end_service` (class 5). All local quality numbers are
**`fake_quant_quality` (class 2)** because local execution is float arithmetic that simulates
quantize→dequantize; every analytical cost figure in §9 is **`analytical` (class 1)** and is never
presented as measured bytes or measured time. Class 4–5 numbers are unavailable here (`measurement-
taxonomy.md` §4) and are reported as "not measured". P2 rows follow `docs/protocols/memory-
accounting.md` §1: the memory number is named explicitly (e.g.
`compression_ratio(mem.checkpoint.total_bytes / fp16_total_bytes)`), never a bare "N×", and the memory
axis never implies a measured speed axis.

**Execution substrate (binding, `AGENTS.md` §2.3 + §2b).** The local CPU workstation is an
orchestration/correctness host only: it no longer runs training, QAT, large-scale inference or GPU
evaluation. Every cell below therefore carries an explicit substrate label, and the label is part of
the frozen configuration (§8):

| Label | Where it runs | May produce an evaluation result? |
|---|---|---|
| `LOCAL-FIXTURE` | developer CPU workstation, tiny synthetic inputs | **No** — pipeline/unit-fixture only, never a result |
| `LOCAL-CPU-MEASUREMENT` | developer CPU workstation, a class 4-CPU kernel on our own serialized artifact | Yes, but only under `benchmark-protocol.md`; not an evaluation cell here |
| `CLOUD-COLAB` | Google Colab notebook generated from a versioned config (`AGENTS.md` §2b) | Yes, after artifact validation (§6.5) |
| `CLOUD-GPU` | cloud GPU runtime (Colab GPU / Kaggle GPU / Colab Enterprise) | Yes, after artifact validation (§6.5) |

A cell evaluated on one substrate is **not comparable** to the same cell on another: the substrate,
hardware model, driver and kernel versions are recorded in the manifest and are part of the
comparability key (§8).

---

## 2. Split separation (mandatory)

Five disjoint roles. A single byte of text MUST NOT occupy two roles; overlaps are detected
programmatically (§7) and a violation excludes the run (`preregistration.md` §9.5).

| Role | Exact asset (pinned revision) | Used for | Touched how often |
|---|---|---|---|
| **Train / continued-training** | C4 `en` **train** split, `allenai/c4` @ `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` (fallback: SlimPajama, see licenses doc §3) | (a) any learning-based preparation/QAT arm; (b) the Tier-1 from-scratch LM | freely, with fixed seeds |
| **Calibration** | **deterministic 256×2048-token slice of the same train split** (§3.2) | GPTQ/AWQ layer input statistics, proxy activation statistics | freely, fixed slice hash |
| **Development** | WikiText-2 **validation** split, `EleutherAI/wikitext_document_level` config `wikitext-2-raw-v1` @ `647234772b9554e208af6c826f23b99e3cac88c8` | method selection, budget sweeps, hyper-parameters | freely |
| **Final perplexity** | WikiText-2 **test** split, same dataset @ same revision (62 documents, 1,290,775 B) | P1 | **once per arm** (≤2 reads, `preregistration.md` §3.3) |
| **Final downstream** | the 6 tasks of §5, evaluation splits only | S4 | **once per arm** |

Rules:

1. **Calibration = subset of train**, not a separate corpus and never part of dev/test. Its slice
   indices and SHA-256 are committed at freeze.
2. **Dev is the only tuning surface.** No hyper-parameter is selected on test.
3. **Test is read once per arm** (a second read is allowed only after a recorded bug fix, logged).
4. **No harness evaluation question ever enters train or calibration** (`AGENTS.md` §4.7). Enforced
   by the §7 audit; a positive overlap in train/calibration is a run-excluding fault.
5. The Tier-1 from-scratch LM is trained **only** on the train role and its own synthetic Tier-0
   data; its validation curve uses the dev role.

---

## 3. Calibration slice (predeclared, not chosen after seeing results)

### 3.1 Source
`allenai/c4`, config `en`, split `train`, revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`,
**streamed** (`datasets.load_dataset(..., streaming=True)`); the dataset is never fully downloaded
(33 TB). License `odc-by` + Common Crawl ToU — see licenses doc §2 D4. Justification for C4 over
SlimPajama is in the licenses doc §3.

### 3.2 Deterministic selection rule (freeze value)
- `calibration_seed = 20261008` (independent stream from the training seed, per `preregistration.md` §6).
- Walk the C4 `en` stream in dataset order; keep a document iff
  `sha256(text.encode("utf-8")) mod 1000 < 1` (≈0.1 % sampling).
- Concatenate kept documents with a single `\n`; tokenize with the **evaluated model's own tokenizer**
  (revision = model SHA, §4.3); take the first `256 × 2048 = 524,288` tokens; drop the remainder.
- Record: the number of source documents consumed, the concatenated-text SHA-256, the token count,
  and the tokenizer revision. **The same slice is used for every arm of a given model** — the hash is
  the proof.
- Size: 524,288 tokens ≈ 2.3 MB of text (≈4.3 chars/token). Memory to hold it: < 10 MB. This is the
  GPTQ-standard calibration shape (128–256 sequences × 2,048 tokens), so it is directly comparable
  with the baseline family.
- **Substrate.** The slice is materialized **inside the cloud run** (Colab/Kaggle), because it feeds
  training/calibration there (`AGENTS.md` §2.3). Locally only the *selection rule* is validated, on a
  tiny synthetic fixture (`LOCAL-FIXTURE`) — never on a real run's corpus.

---

## 4. Harness pin and environment

| Item | Frozen value | Verification |
|---|---|---|
| LM Evaluation Harness **commit** | `ddd67220430a2470529f25fd5c05a576ca1057a0` | peeled commit of annotated tag `v0.4.13`: `gh api repos/EleutherAI/lm-evaluation-harness/git/ref/tags/v0.4.13 --jq .object.sha` → `eb4678c3…`; then `gh api .../git/tags/eb4678c3… --jq .object.sha` → `ddd6722…` |
| Harness release | `v0.4.13` (published 2026-08-31) | `gh api repos/EleutherAI/lm-evaluation-harness/releases/latest --jq .tag_name` |
| Harness license | `MIT` | `gh api repos/EleutherAI/lm-evaluation-harness/license --jq .license.spdx_id` |
| Install | `uv pip install "lm-eval==0.4.13"` **or** a git checkout at the SHA above; whichever is used is recorded in the manifest's `harness_commit` field | `pip show lm-eval` / `git rev-parse HEAD` in `scripts/reproduce/` |
| Runner | `lm_eval` console entry point (`lm_eval.__main__:cli_evaluate`) | `pyproject.toml` `[project.scripts]` at the pinned commit |
| Device | `<SUBSTRATE_DEVICE>` (`cuda` on cloud GPU, `cpu` only for `LOCAL-FIXTURE`), dtype `float32` | see §4.1 |

> **Pin reconciled 2026-10-08.** `docs/research/upstream-lockfile.md` (row U6) now pins this same
> immutable release commit `ddd6722…` (`v0.4.13`, MIT), after stream A independently re-verified the
> tag peel; the diverged `main` HEAD `d6de81643928d653435c431bae19945d41d32520` is recorded there as a
> divergence. Lockfile = source of truth; this protocol and the run manifest carry the tag commit
> explicitly.

### 4.1 Execution dtype, batch size, seeds

- `dtype=float32` for **all** cells (cloud and any fixture). x86 CPU fp16 kernels are not
  performance-neutral and are sometimes unavailable, and on GPU fp16 would change the numerics; the
  same dtype is used for every method — this is a fairness requirement, not a preference.
- `--batch_size 1` for every confirmatory cell (padding/masking numerics must be identical across
  arms; a fixed batch size is part of byte-identity). `--batch_size auto` is permitted **only** for
  exploratory timing runs, which may never be reported as confirmatory.
- `--seed 0,0,0,0` (python, numpy, torch, fewshot). The v0.4.13 CLI defaults are
  `0,1234,1234,1234`; we override explicitly so the few-shot sampler is deterministic and identical
  across arms. Set `PYTHONHASHSEED=0` and a **fixed** `OMP_NUM_THREADS` for the run; both are recorded
  in the manifest (§8).
- `--apply_chat_template` **False** for every primary cell (all primary checkpoints are base models).
  Instruct variants are evaluated only in a clearly-labelled separate "template sensitivity" table and
  never mixed with the primary numbers.
- No task in the suite uses free-form generation, so **generation parameters do not apply**:
  `--gen_kwargs` is left unset, and `temperature`, `top_p`, `max_gen_toks` are never set. (If a future
  amendment adds a generative task, the amendment must freeze the full decoding config here.)

### 4.2 Tokenizer revision
The tokenizer is **the evaluated model's tokenizer, loaded at the same immutable commit as the
weights** — `revision=<model SHA>` is passed once in `--model_args` and the harness loads
`tokenizer.json` / `tokenizer.model` / `tokenizer_config.json` from that revision. There is no
separate tokenizer pin. Consequences that are recorded in every manifest: token counts (and therefore
perplexity) are **tokenizer-specific and not comparable across models**; only same-model,
same-tokenizer comparisons are made.

Measured token counts for the fixed WikiText-2 test text (62 documents, 1,288,493 characters) under
each candidate tokenizer — obtained by tokenizing the pinned test split with each model's own
`tokenizer.json` (see licenses doc §0 for the command shape):

| Tokenizer (= model revision) | Vocab | Tokens for the fixed test text |
|---|---|---|
| `gpt2` (M9) | 50,257 | 286,158 |
| TinyLlama-1.1B @ `59f6f375…` (M1) | 32,000 | 335,626 |
| Qwen2.5-0.5B @ `060db649…` (M3) | 151,665 | 299,085 |
| SmolLM2-360M @ `f8027fd0…` (M5) | 49,152 | 303,414 |
| SmolLM2-135M @ `93efa2f0…` (M13) | 49,152 | 303,414 — **byte-identical tokenizer** to M5 (`tokenizer.json` sha256 `9ca9acddb6525a19…` for both), verified by hashing both files |

---

## 5. The bounded task suite (chosen before method development)

Six tasks, all named in the task specification, fixed here. Task *names* are the harness
keys; **all** template/choice/normalization behaviour comes from the pinned harness YAML files at the
commit above (no local task overrides, no custom prompts — this is what makes "byte-identical across
methods" mechanically checkable).

| Task key | Dataset (path / name) | Dataset revision | Split | Output type | Few-shot (frozen) | Metric(s) | Choice normalization |
|---|---|---|---|---|---|---|---|
| `hellaswag` | `Rowan/hellaswag` / `null` | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | `validation` | `multiple_choice` | **10** | `acc`, `acc_norm` | harness `acc_norm` (continuation byte-length normalisation) |
| `arc_easy` | `allenai/ai2_arc` / `ARC-Easy` | `210d026faf9955653af8916fad021475a3f00453` | `test` | `multiple_choice` | **25** | `acc`, `acc_norm` | as above |
| `arc_challenge` | `allenai/ai2_arc` / `ARC-Challenge` | `210d026faf9955653af8916fad021475a3f00453` | `test` | `multiple_choice` | **25** | `acc`, `acc_norm` | as above |
| `piqa` | `baber/piqa` / `null` | `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` | `validation` | `multiple_choice` | **0** | `acc`, `acc_norm` | as above |
| `winogrande` | `allenai/winogrande` / `winogrande_xl` | `01e74176c63542e6b0bcb004dcdea22d94fb67b5` | `validation` | `multiple_choice` | **5** | `acc` | n/a |
| `boolq` | `aps/super_glue` / `boolq` | `3de24cf8022e94f4ee4b9d55a6f539891524d646` | `validation` | `multiple_choice` | **0** | `acc` | n/a |

**Why these few-shot counts.** They are the suite's canon: `hellaswag`=10, `arc_challenge`=25 and
`winogrande`=5 are the values the Open LLM Leaderboard v1 used, **verified from a model card that
reports leaderboard results** (TinyLlama-1.1B `README.md`: *"ARC-Challenge (25-Shot)"*, *"HellaSwag
(10-Shot)"*, *"Winogrande (5-Shot)"*). `arc_easy`=25 matches its sibling `arc_challenge`.
`piqa`=0 and `boolq`=0 are the standard 0-shot convention for those two and are deliberately kept at
0 to bound cost (§9). **These values were fixed before any method was written.**

**Why the harness YAMLs do not fix few-shot.** At the pinned commit, the six YAMLs declare no
`num_fewshot` field, so the value must be supplied on the command line — and `--num_fewshot` is a
single global integer. Therefore **each task is run in its own invocation** (§6.3). Few-shot
exemplars are drawn from each task's `training_split` by the harness's seeded sampler.

**Prompt choices (from the pinned YAMLs, quoted verbatim — no local override):**

| Task | `doc_to_text` at the pinned commit |
|---|---|
| `hellaswag` | `"{{query}}"` |
| `arc_easy` / `arc_challenge` | `"Question: {{question}}\nAnswer:"` |
| `piqa` | `"Question: {{goal}}\nAnswer:"` |
| `winogrande` | `preprocess_winogrande.doc_to_text` (function in the pinned repo) |
| `boolq` | `"{{passage}}\nQuestion: {{question}}?\nAnswer:"` |

Choices are taken from `doc_to_choice` (`"choices"`, `"{{choices.text}}"`, `"{{[sol1, sol2]}}"`,
`["no","yes"]`, or the winogrande function) as declared in those YAMLs. Normalization uses the
harness's own `acc_norm` implementation; we never post-process predictions.

**Decontamination flag.** The pinned YAMLs set `should_decontaminate: true` with
`doc_to_decontamination_query` for `arc_easy`, `arc_challenge`, `piqa`, `winogrande`, `boolq` (and
`wikitext`). At the pinned commit the `evaluate()` entry point no longer accepts an n-gram index path
(verified by reading `lm_eval/evaluator.py` at `ddd6722…`), so the flag is inert at runtime — our own
audit (§7) implements the check.

---

## 6. Cells, and how each is executed

### 6.1 Primary perplexity cell
Task key `wikitext` (pinned YAML: `dataset_path: EleutherAI/wikitext_document_level`,
`dataset_name: wikitext-2-raw-v1`, `output_type: loglikelihood_rolling`, `doc_to_target:
preprocess_wikitext.wikitext_detokenizer`), metrics `word_perplexity`, `byte_perplexity`,
`bits_per_byte`. Because the harness task reads only `wikitext-2-raw-v1`, the **dev** role is the
`validation` split and the **test** role is the `test` split of the same pinned dataset — the split
separation is enforced by the harness task's own `validation_split`/`test_split` fields, and the
test split is read once per arm.

### 6.2 Downstream suite
The six tasks of §5, at their declared splits and few-shot counts, never the `test` split of a task
that the harness marks as `test_split: null` (hellaswag/piqa have `test_split: null` → `validation`).

### 6.3 Exact command template (every confirmatory cell)

```bash
# Run from a generated Colab/Kaggle notebook (thin notebook, thick modules — AGENTS.md §2b rule 1).
# <SUBSTRATE_DEVICE> is cuda on a cloud GPU runtime; the eval is never run on the local workstation.
OMP_NUM_THREADS=<N_THREADS> PYTHONHASHSEED=0 lm_eval \
  --model hf \
  --model_args "pretrained=<HF_ID>,revision=<MODEL_SHA>,dtype=float32,trust_remote_code=False" \
  --tasks "<TASK_KEY>" \
  --num_fewshot <N_FROZEN> \
  --batch_size 1 \
  --device <SUBSTRATE_DEVICE> \
  --seed 0,0,0,0 \
  --output_path "artifacts/manifests/eval/<method_id>/<model_id>/<task_key>.json"
```

- `--device`: `cpu` only for `LOCAL-FIXTURE` runs (synthetic data, no result); `cuda` for every
  cloud GPU cell. The device is part of the frozen configuration (§8) and of the comparability key.
- `dtype=float32` is frozen for byte-identity across arms; if a substrate makes fp32 impractical, the
  change must be an amendment applied to **all** arms and the substrate recorded (§8) — never a
  per-arm choice.

- One invocation per task (the global `--num_fewshot` cannot differ within a run).
- `--limit` is **omitted** for confirmatory cells and set only for bounded smoke cells (§9.4); a
  confirmatory number is never produced with `--limit` set.
- `--log_samples` is **not** passed (the CLI attribute is `store_true` with `default=SUPPRESS`, so it
  is falsy unless requested; verified at the pinned commit). If a debugging run enables it, that run
  is exploratory and its samples are not published without an amendment (the eval datasets are
  share-alike; see licenses doc §2). Samples never affect request construction, so they cannot break
  byte-identity — they only add output bytes.
- The manifest (`run_manifest.json`, `AGENTS.md` §2b rule 3) records: run id, harness commit,
  `pip freeze` hash, model id + revision, dtype, batch size, seeds, task key, few-shot count, limit,
  calibration hash, **substrate label, hardware model, driver and kernel versions, `OMP_NUM_THREADS`,
  GPU-hours, start/end times**, the exact command string, and artifact checksums.
- `--trust_remote_code=False` everywhere; the candidate models need no remote code.
- A cell is **valid** only if all of the above match the frozen values byte-for-byte.

### 6.4 Cells, substrate and cost (per cell)

Substrate labels are defined in §1. Cost basis: `wall ≈ 2·N·T / (f·10¹²)` seconds, with the effective
fp32 throughput `f` = **2–8 TFLOP/s** on a T4-class Colab runtime and **5–15 TFLOP/s** on an
A100-class runtime [INFERENCE, class-1 analytical; grounded in the §9.1 model]. Runtime cap per free
Colab session ≈ 12 h; M7-class cells may need a resumed session. Model downloads and dataset fetches
are excluded from the figures (WikiText-2 test ≈ 1.3 MB; the six eval datasets are small) and are
amortised once per runtime. Peak RAM/VRAM is [INFERENCE] from weights + activations.

| Cell | Model | Substrate | Tokens | Est. cloud cost (T4-class) | Peak VRAM/RAM |
|---|---|---|---|---|---|
| Eval-pipeline fixture (synthetic MC + toy tokenizer) | any | **LOCAL-FIXTURE** | ≈2 k (synthetic) | 0 (local seconds) | < 0.5 GB |
| `wikitext` test PPL | M13 SmolLM2-135M | **CLOUD-COLAB** | 303,414 (measured; tokenizer identical to M5) | < 0.01 GPU-h (≈10–60 s) | < 2 GB |
| `wikitext` test PPL | M9 GPT-2 (124 M) | **CLOUD-COLAB** | 286,158 (measured) | < 0.01 GPU-h | < 2.5 GB |
| `wikitext` test PPL | M5 SmolLM2-360M | **CLOUD-COLAB** | 303,414 (measured) | 0.01–0.03 GPU-h | < 4 GB |
| `wikitext` test PPL | M3 Qwen2.5-0.5B | **CLOUD-COLAB** | 299,085 (measured) | 0.01–0.04 GPU-h | < 5 GB |
| `wikitext` test PPL | M1 TinyLlama-1.1B | **CLOUD-GPU** | 335,626 (measured) | 0.03–0.10 GPU-h | < 8 GB |
| `wikitext` test PPL | M7 SmolLM2-1.7B | **CLOUD-GPU** | ≈310 k (est.) | 0.04–0.15 GPU-h | < 12 GB |
| 6-task full suite, declared few-shot | M13 SmolLM2-135M | **CLOUD-COLAB**/GPU | ≈38.6 M | 0.4–1.4 GPU-h | < 2 GB |
| 6-task full suite | M9 GPT-2 | **CLOUD-GPU** | ≈38.6 M | 0.4–1.5 GPU-h | < 2.5 GB |
| 6-task full suite | M5 SmolLM2-360M | **CLOUD-GPU** | ≈38.6 M | 1.0–3.9 GPU-h | < 4 GB |
| 6-task full suite | M3 Qwen2.5-0.5B | **CLOUD-GPU** | ≈38.6 M | 1.3–5.3 GPU-h | < 5 GB |
| 6-task full suite | M1 TinyLlama-1.1B | **CLOUD-GPU** | ≈38.6 M | 2.9–11.8 GPU-h | < 8 GB |
| 6-task full suite | M7 SmolLM2-1.7B | **CLOUD-GPU** | ≈38.6 M | 4.6–18.3 GPU-h | < 12 GB |
| 6-task full suite **× ≥3 seeds** (confirmatory S4) | any | **CLOUD-GPU** | ×3 | ×3 of the above | — |
| Tier-1/2/3 training arms (PTQ/QLoRA/LoftQ/LR-QAT/SpectraQuant) | M13/M1/M7 | **CLOUD-GPU** | — | budgeted separately under `AGENTS.md` §2b rule 8 (no paid resource without prior authorization) | — |
| Tier-4 ViT/DeiT on CIFAR-100/ImageNet-100 | M10/M11 | **CLOUD-GPU** (optional) | small | ≈0.1–1 GPU-h | < 4 GB |
| Class-4-CPU kernel measurement (not an eval cell) | our serialized int4/int8 artifact | **LOCAL-CPU-MEASUREMENT** | — | 0 | < 2 GB |
| Class 4-GPU / class 5 claims | any | **CLOUD-GPU**, real kernel required | — | per `benchmark-protocol.md` | — |

**Consequence for the paper's scope.** No evaluation cell is stated to run on this workstation: the
only local cells are the pipeline fixture (never a result) and the class-4-CPU measurement (owned by
the benchmark protocol). Every P1/S4 number comes from the cloud substrate, and only after its
artifact bundle passes validation (`§6.5`).

### 6.5 A cloud cell is a result only after artifact validation (`AGENTS.md` §2b rules 3–5)

A run that exists only as remote logs is **not a result** and must never enter a table or the results
registry. A cell becomes a result only when its exported artifact bundle is downloaded and passes,
locally, all of:

1. `run_manifest.json` is present and validates against `artifacts/schemas/`.
2. The manifest's git commit SHA matches the branch/commit the run was declared from, and the
   dirty-state flag is recorded.
3. The expected metric files for the declared task key(s) are present and non-empty.
4. Every artifact's recorded checksum matches the downloaded bytes.
5. The frozen configuration matches (§8): substrate ≤ the declared label, dtype, batch size, seeds,
   few-shot count, limit, task keys, dataset revisions, model revision.
6. A failed validation records the run as **failed** with its reason; it is never retried into a
   success and never silently dropped (`AGENTS.md` §2b rules 4–5).

The registry is updated **only** from validated bundles. `spectraquant cloud collect` performs these
checks; a manual log reading is not evidence.

---

## 7. Contamination audit (mandatory, programmatic, run before every arm)

1. **Sources checked.** (a) the evaluated model's stated training corpora — TinyLlama: SlimPajama +
   StarCoderData (from its model card); SmolLM2: SmolLM-Corpus; Qwen2.5: undisclosed; GPT-2: WebText;
   and (b) **our** train + calibration slice (C4 `en`, §3).
2. **Method.** Normalise text (lowercase, collapse whitespace), form the set of **13-grams**
   (n = 13, the OpenAI/`lm_eval/decontaminate.py` convention verified from the pinned
   `docs/decontamination.md`), and report, per eval task, the number of *documents* sharing at least
   one 13-gram with any source.
3. **Enforcement.** (i) Overlap between an eval split and **our** train/calibration slice ⇒ **fault,
   run excluded** (`preregistration.md` §9.5). (ii) Overlap with a *model's* pretraining data is
   reported per task and the metric is **recomputed with the contaminated documents removed**, both
   numbers published (the harness's `should_decontaminate: true` intent, implemented locally because
   the pinned harness no longer takes an n-gram path at run time).
4. **Record.** The audit writes `contamination_report.json` (per task: `n_docs`,
   `n_contaminated`, `ngram`, `source`, `sha256(eval_split_text)`) next to the eval manifest, and its
   hash enters the run manifest.
5. **Substrate.** The audit is local analysis over the small eval splits (`LOCAL-FIXTURE`-scale,
   permitted under `AGENTS.md` §2.3 "result analysis"); the model-side corpus n-gram sets it consults
   come from the checksum-validated cloud run manifest, never from a fresh local corpus download.
6. **No harness evaluation question is ever used to train, calibrate, or select anything** (`AGENTS.md`
   §4.7); the audit is the mechanical proof.

---

## 8. Byte-identity rule (the freeze mechanism)

For a set of numbers to be comparable, **all** of the following must be identical across the methods
being compared:

| Frozen element | Value |
|---|---|
| Harness commit | `ddd67220430a2470529f25fd5c05a576ca1057a0` (`v0.4.13`) |
| Task keys & YAMLs | exactly the six of §5 + `wikitext`, from that commit |
| Dataset revisions | §5 table (and `EleutherAI/wikitext_document_level` @ `647234772b9554e208af6c826f23b99e3cac88c8`) |
| Model revision | the arm's own pinned SHA; the tokenizer is that revision's |
| `--num_fewshot` | 10 / 25 / 25 / 0 / 5 / 0, per §5 |
| `--batch_size`, `--device`, dtype | `1`, `<SUBSTRATE_DEVICE>` (fixed per substrate), `float32` |
| `--seed` | `0,0,0,0` |
| `--limit` | identical for every arm (unset for confirmatory) |
| Prompt/normalization | harness YAMLs at the pinned commit; no local override, no post-processing |
| Chat template | off for primary cells |
| **Execution substrate** | one of `LOCAL-FIXTURE` / `LOCAL-CPU-MEASUREMENT` / `CLOUD-COLAB` / `CLOUD-GPU`; identical label across the arms being compared |
| **Hardware / driver / kernel versions** | recorded in `run_manifest.json`; identical across the arms being compared |
| Threads | `OMP_NUM_THREADS` (fixed per substrate, recorded), `PYTHONHASHSEED=0` |

The runner computes a single `eval_config_hash` over exactly these fields and stores it in every
result manifest; two runs with different hashes are not comparable and must not be tabulated
together. `compare` tooling refuses to diff cells whose hashes differ.

**Substrate is part of the key.** The same logical cell evaluated on two different substrates
(different substrate label, or different GPU model / driver / kernel version) produces two
**separate** results that are **never averaged, never merged into one table column, and never
differences-ed**. They are reported side by side with their substrate named. Byte-identity of the
*protocol* is necessary but not sufficient for comparability across hardware; therefore all arms of a
comparison run on the **same** substrate and hardware generation, and any cross-substrate comparison
is labelled exploratory.

---

## 9. Cost model and expected cost per cell (cloud substrate)

### 9.1 Model and constants

Forward FLOPs for scoring `T` tokens through an `N`-parameter transformer ≈ **2·N·T**. Multiple-choice
tasks cost one forward pass per (context + continuation) pair, i.e. once per choice.

**Local constants** (measured on this workstation 2026-10-08; commands in the licenses doc §5) — they
now bound only `LOCAL-FIXTURE` work and class-4-CPU measurements, **not** evaluation cells:

- fp32 GEMM ceiling: **326.66 GFLOP/s** (numpy 2.5.1, 2048³ `float32`).
- Memory copy bandwidth: **21.78 GB/s**.

**Cloud constants** [INFERENCE, class-1 analytical] — effective fp32 throughput for a batch-1
eval workload, i.e. well below the vendor peak because of memory/latency limits and non-GEMM ops:

- T4-class Colab runtime: `f` = **2–8 TFLOP/s**.
- A100-class runtime: `f` = **5–15 TFLOP/s**.

`wall ≈ 2·N·T / (f · 10¹²)` seconds, reported as a range. These are estimates, not measurements: no
cell has been run yet. Free-tier Colab caps a session at ~12 h; a cell whose upper bound exceeds that
must be resumable (`AGENTS.md` §2b rule 6).

### 9.2 Per-tokenizer token counts (measured, §4.2)
WikiText-2 document-level test text: 1,288,493 characters, 62 documents →
M9 `gpt2` 286,158; M1 TinyLlama 335,626; M3 Qwen2.5-0.5B 299,085; M5 SmolLM2-360M 303,414;
M13 SmolLM2-135M 303,414 (identical tokenizer to M5).

### 9.3 Per-document cost of the downstream suite (analytical, Qwen tokenizer)

Derived by summing the string fields of each pinned evaluation split and applying the pinned prompt
templates with the §5 few-shot counts (the few-shot block dominates `hellaswag`/ARC; the estimator
is stated here so it can be recomputed):

| Task | Requests (docs × choices) | ≈ tokens / request | ≈ total tokens |
|---|---|---|---|
| `hellaswag` (10-shot) | 10,042 × 4 = 40,168 | 638 | 25.6 M |
| `arc_easy` (25-shot) | 2,376 × 4 = 9,504 | 787 | 7.5 M |
| `arc_challenge` (25-shot) | 1,172 × 4 = 4,688 | 865 | 4.1 M |
| `piqa` (0-shot) | 1,838 × 2 = 3,676 | 35 | 0.13 M |
| `winogrande` (5-shot) | 1,267 × 2 = 2,534 | 154 | 0.39 M |
| `boolq` (0-shot) | 3,270 × 2 = 6,540 | 142 | 0.93 M |
| **total (one model, one seed)** | 67,110 | — | **≈ 38.6 M** |

### 9.4 Expected cost per cell (cloud; class-1 analytical)

`GPU-h` figures exclude model download (0.3–4.4 GB) and runtime start-up; add ≈2–5 min per runtime.

**WikiText-2 document-level test perplexity (one pass per model):**

| Model | N (params) | Tokens | FLOPs ≈ 2·N·T | T4-class (2–8 TFLOP/s) | A100-class (5–15 TFLOP/s) |
|---|---|---|---|---|---|
| M13 SmolLM2-135M | 1.345e8 | ≈303 k | 8.2e13 | 10–41 s (0.003–0.011 GPU-h) | 5–16 s |
| M9 GPT-2 | 1.370e8 | 286,158 | 7.8e13 | 10–39 s | 5–16 s |
| M5 SmolLM2-360M | 3.618e8 | 303,414 | 2.2e14 | 27–110 s (0.008–0.031 GPU-h) | 15–44 s |
| M3 Qwen2.5-0.5B | 4.940e8 | 299,085 | 3.0e14 | 37–150 s | 20–60 s |
| M1 TinyLlama-1.1B | 1.100e9 | 335,626 | 7.4e14 | 92–370 s (0.026–0.103 GPU-h) | 49–147 s |
| M7 SmolLM2-1.7B | 1.711e9 | ≈310 k | 1.1e15 | 137–550 s (0.038–0.153 GPU-h) | 73–220 s |

**Six-task downstream suite (one seed, declared few-shot, ≈38.6 M tokens):**

| Model | FLOPs ≈ 2·N·T | T4-class | A100-class |
|---|---|---|---|
| M13 SmolLM2-135M | 1.04e16 | 22–86 min (0.36–1.44 GPU-h) | 12–35 min (0.19–0.58 GPU-h) |
| M9 GPT-2 | 1.06e16 | 22–88 min (0.37–1.47) | 12–35 min (0.20–0.59) |
| M5 SmolLM2-360M | 2.80e16 | 58–233 min (0.97–3.88) | 31–93 min (0.52–1.55) |
| M3 Qwen2.5-0.5B | 3.82e16 | 80–318 min (1.33–5.30) | 42–127 min (0.71–2.12) |
| M1 TinyLlama-1.1B | 8.50e16 | 177–708 min (2.95–11.80) | 94–283 min (1.57–4.72) |
| M7 SmolLM2-1.7B | 1.32e17 | 275–1,100 min (4.58–18.33) | 147–440 min (2.45–7.33) |

The confirmatory S4 requirement of ≥3 seeds multiplies the table by 3. M1/M7 rows exceed a ~12 h
free Colab session at the pessimistic end and must use the resumable protocol (`AGENTS.md` §2b
rule 6) or an A100-class runtime. `--limit N` scales linearly: **≈9,822 tokens per document across the
six tasks**, so `--limit 10` ≈ 98 k tokens (≈4 % of one task-run) and `--limit 50` ≈ 491 k tokens.

**Training arms are not evaluation cells** and carry their own budget: no paid cloud resource is
started without prior authorization and a stated maximum estimated cost (`AGENTS.md` §2b rule 8);
free tiers are the default.

### 9.5 Cost- and time-reporting rule
Every reported figure carries its **substrate label**, hardware model, driver/kernel versions,
relevant command, and either the **measured seconds** (if the cell ran) or the label **class-1
analytical estimate** (if derived). A CPU wall time measured on the workstation may be reported only
for `LOCAL-FIXTURE` and `LOCAL-CPU-MEASUREMENT` cells; it is never quoted as the cost of an evaluation
cell, because evaluation cells do not run locally. No cloud figure is ever presented as a class-4/5
claim unless it used a real supported kernel under `benchmark-protocol.md`.

---

## 10. Freeze checklist for this protocol

- [x] Harness commit reconciled with `upstream-lockfile.md` (row U6) — both pin `ddd6722…` (`v0.4.13`).
- [ ] Six task keys + few-shot counts committed as `configs/eval/suite.yaml` (stream B).
- [ ] Calibration slice hash committed (`artifacts/manifests/calibration.json`).
- [ ] `eval_config_hash` implementation merged and unit-checked.
- [ ] Contamination audit script merged with a positive and a negative fixture.
- [ ] Eval-pipeline fixture run once on `LOCAL-FIXTURE` (synthetic inputs; never a result) to validate
      request construction and the `eval_config_hash`.
- [ ] One `CLOUD-COLAB` smoke cell run end-to-end, with its **measured** GPU-hours and a
      checksum-validated artifact bundle recorded (replacing the analytical ranges for that cell).
- [ ] Cloud submission/collection adapters (`src/spectraquant/cloud/**`) exercised, including
      artifact validation (§6.5).
- [ ] This file marked FROZEN with date + commit SHA by the orchestrator.
