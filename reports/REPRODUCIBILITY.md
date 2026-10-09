# Reproducibility

Scope: this file says exactly how to reproduce the **local** evidence in this repository from a clean
checkout, what the expected outputs are, and what **cannot** be reproduced locally. It does not
reproduce cloud cells, because none has run.

Substrate labels follow `AGENTS.md` §2b; measurement classes follow `AGENTS.md` §5.

## 1. Clean-checkout commands

```bash
git clone https://github.com/OussemaHarrabi/SpectraQuant.git
cd SpectraQuant
git checkout infra/bootstrap            # or the M10 release tag once it exists

uv sync --all-extras                    # CPU-only torch + all optional extras (onnx, alloc, models, cloud)
uv run pytest -q                        # full CPU test suite
uv run ruff check . && uv run ruff format --check .
uv run spectraquant --help              # CLI surface

# Tier-0 CI smoke fixture: bit-for-bit reproducible at a fixed seed
uv run spectraquant smoke --config configs/experiment/smoke.yaml
uv run spectraquant validate-manifest artifacts/sample-results/smoke-manifest.json
```

`uv` is the only supported workflow (`pyproject.toml` + committed `uv.lock`). Python 3.11. CPU only;
no CUDA, no GPU, no cloud, no network access is required for the tests or for the generator below.
`--all-extras` is needed because the class-4-CPU regeneration uses the `onnx` extra.

## 2. Expected test result

At the M10 release commit, `uv run pytest -q` reports:

```
908 passed, 1 skipped      # clean checkout
909 passed                 # when the working tree is dirty (see below)
```

The difference is `tests/unit/test_cloud_spec.py:183` ("working tree is clean; the dirty-tree guard
has nothing to reject"): on a clean checkout that guard test has nothing to reject and skips; with
uncommitted changes present it runs, so no test is skipped. Both are green. The count grows as slices
land; the M10 release slice adds `tests/unit/test_release_artifacts.py` (3 tests). The pre-M10
baseline was `889 passed, 1 skipped`.

## 3. Regenerating the artifacts

Every committed artifact under `artifacts/sample-results/` is produced by a script; none is
hand-edited. From the repository root:

```bash
uv run python scripts/experiments/proxy_fixture_measurement.py     # proxy-fixture/proxy-fixture.json
uv run python scripts/experiments/proxy_validation_sweep.py        # proxy-validation/proxy-validation.json
uv run python scripts/experiments/allocation_frontier.py           # allocation-frontier/frontier.json + manifests/
uv run python scripts/experiments/regularizer_sweep.py             # regularizer/sweep.json
uv run python scripts/experiments/class4cpu_measurement.py         # class4cpu/class4cpu.json (needs the onnx extra)
```

Then regenerate the published tables and the Pareto figure from those artifacts:

```bash
uv run python scripts/reproduce/generate_release_artifacts.py
```

`tests/unit/test_release_artifacts.py` fails if the committed tables/figures differ from the
generator's output, so the published files cannot drift from the artifacts.

## 4. Expected digests (where stable)

The **generated** files are byte-stable (no timestamps, no RNG) and their SHA-256 digests are:

```
8aaa30fd91663ec34836ddcb0cb086200206e8145d5d2458ccf1e57a588d59f6  reports/tables/allocator-frontier.md
7e9159b793ef574b29395f24fcd6f608a14f8e87ed593b1322bed796baa01dee  reports/tables/class4cpu-bytes.md
4d467f533aee58fb046a9acc8bdd5720555756f0120375db73253d209f906f4d  reports/tables/class4cpu-latency.md
f70e12f8107a877d7454bd833bacb4c0e12399332d53a70d04d73ff733bdbd54  reports/tables/proxy-fixture.md
2a2e34b7c36fc748f80aa3dcf997e0a2eaa91f0ee693310c84e98bae4712c83a  reports/tables/proxy-ranking.md
44cbd86788d8be04546bb96601c6394c48c9d8ee5262d42cf490ce2621216bc9  reports/tables/regularizer-arms.md
d31c3f3bb0addd54a0650a8683e08bcc17238c887802d0a0414e6fd9b37d7f4c  reports/figures/pareto-damage-vs-bytes.svg
```

The **source JSON artifacts** are *not* byte-stable across regeneration, by design: they carry
documented wall-clock and timing telemetry that is provenance, not science. The varying fields are:

| artifact | fields that vary | scientific content |
|---|---|---|
| `allocation-frontier/frontier.json` | `fixture.training.wall_time_s` | deterministic (per-arm manifests are byte-identical across runs) |
| `proxy-validation/proxy-validation.json` | `runs.training.*.wall_time_s`, `runtime.wall_time_s`, `git_commit`/`git_dirty` | deterministic |
| `proxy-fixture/proxy-fixture.json` | `fixtures.*.training.wall_time_s`, `git_commit` | deterministic |
| `regularizer/sweep.json` | `wall_time_s`, per-run telemetry, `git.*` | deterministic |
| `class4cpu/class4cpu.json` | `timestamp_utc`, `latency.raw_ns`/percentiles, `host`, `git_commit`/`git_dirty` | container SHA-256s and error figures are deterministic given the pins |

As committed at M10, the artifact digests are (reference only — they change when the wall-clock/timing
fields change):

```
0c0e61224fa0b02a4204b1e24b45cb8251b625504038651e2c730b2c2e66d685  artifacts/sample-results/proxy-validation/proxy-validation.json
9c5b2fac27083dbd352aa1924bbb6df72dbeea6c30b47585df93553a08648a65  artifacts/sample-results/proxy-fixture/proxy-fixture.json
bd34f8936cc4b8932128eb5cdcfbf5eab8f78af698b95402082ee3ef994aaf79  artifacts/sample-results/allocation-frontier/frontier.json
65a847f1e38716a806196d06821763291842b95bce5810a08ffcce9e82aa2986  artifacts/sample-results/class4cpu/class4cpu.json
aa49f83fcca60b21fe4b7b450940c08e4c94d94fd052c2d07e0e2a55b3c3eefb  artifacts/sample-results/regularizer/sweep.json
```

Stable within-run guarantees that are checked:

* Each allocation manifest replays deterministically (`reproduce_allocation`; `diff -rq` clean across
  two runs) and is replayed by `test_committed_frontier_manifests_replay`.
* `accounted_bytes == measure_serialized_bytes` for every serialized format (asserted by test); 18/18
  frontiers arms reconcile at `max_relative_difference = 0.0`.
* The class-4-CPU artifact records the SHA-256 of every ONNX container; the `*.onnx` files themselves
  are git-ignored (`AGENTS.md` §7: never commit large raw outputs).

## 5. What cannot be reproduced locally

These are **NOT RUN** and must not be reported as anything else:

1. **All model training and evaluation, Tiers 1–5** — the pinned Tier-1 transformer (`L_b = 8`,
   `d = 256`), WikiText-2 perplexity, the proxy-validation confirmatory cells (H2/H4), the H3
   regularizer cell, and the Tier-2 TinyLlama campaign. They require the cloud notebook substrate
   (`CLOUD-COLAB` / `CLOUD-GPU`) and a validated `run_manifest.json`; none exists.
2. **M3 bounded reproduction** of LR-QAT and LoftQ — the plan is frozen
   (`configs/repro/lr_qat_smollm2_135m.yaml`) but **NOT RUN**.
3. **Class 4-GPU** (bitsandbytes, GPTQ/AWQ/Marlin, TorchAO CUDA, FlashAttention, vLLM CUDA) — no CUDA
   on the workstation; CUDA paths must fail loudly rather than fall back.
4. **Class 5 end-to-end service latency/throughput** — no Windows vLLM wheel and no Linux GPU host.
5. **Cloud-only dataset fetches** — WikiText-2 and C4 are fetched by pinned revision by the cloud run;
   the local machine does not download them (see `docs/cards/`).

The class-4-CPU measurement **is** reproducible locally (it is a fixture on one weight matrix, not a
model evaluation) but is **not** comparable to published GPU latency or throughput figures, and makes
no speedup claim.
