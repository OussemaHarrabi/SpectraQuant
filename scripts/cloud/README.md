# Cloud run book

The workstation has no usable CUDA GPU and may not run research training (`AGENTS.md` §2.3), so
every Tier 1–5 experiment executes on a cloud notebook substrate. `src/spectraquant/cloud/**` is the
**only sanctioned path** between this repository and that substrate (§2b). This file is the operator
procedure: credentials, free-tier limits, the exact command sequences, and how to resume after a
pre-emption.

Design contract: `docs/coordination/design-cloud-adapter.md` (frozen).

---

## 1. What each platform can actually do

| Platform | Adapter | Automation reality |
|---|---|---|
| Google Colab (consumer) | `colab` | **No submission API.** `spectraquant cloud submit` emits an execution-ready notebook and records `submitted_by: "human"` with `remote_id: null`. A human opens and runs it; `cloud status` therefore reports `unknown` and never pretends otherwise. |
| Kaggle Notebooks | `kaggle` | Fully programmatic through the official CLI (`kernels push` → `status` → `output`). The remote kernel id is persisted **before** `submit` returns. |
| Colab Enterprise | `colab_enterprise` | Programmatic, **paid**: requires an authorized GCP project, `SPECTRAQUANT_ALLOW_PAID=1`, a cost ceiling, a run container image and a GCS staging bucket. |
| `local_cpu` | `local_cpu` | The workstation itself, for Tier-0 fixtures and class 4-CPU work only. Refuses `gpu_required=True`. |

---

## 2. Credentials (environment or platform secret store only)

Credentials are read from the environment or the platform's secret store and are **never** written to
a notebook, config, log, registry line or commit (`AGENTS.md` §2b rule 7). Every log line, error
message, registry line and generated notebook passes through
`spectraquant.cloud.secrets.redact`, which replaces a credential *value* with
`[REDACTED:<VARIABLE>]`. Variable names are visible; values never are.

| Variable | Used by | How to provide it |
|---|---|---|
| `KAGGLE_USERNAME` | Kaggle CLI | `export KAGGLE_USERNAME=...` (or `~/.kaggle/kaggle.json`, which the CLI reads itself) |
| `KAGGLE_KEY` | Kaggle CLI | same as above |
| `HF_TOKEN` | dataset/model downloads inside the notebook | `export HF_TOKEN=...`; on Colab use *Secrets* (🔑) and read it via `google.colab.userdata` inside the notebook |
| `GOOGLE_APPLICATION_CREDENTIALS` | Colab Enterprise | path to an ADC/service-account JSON **outside** the repository (e.g. `~/.config/gcloud/...`) |
| `SPECTRAQUANT_ALLOW_PAID` | budget guard | `1` **only** after you have explicitly authorized a paid run |
| `GOOGLE_CLOUD_PROJECT` / `GCLOUD_PROJECT` | Colab Enterprise guard | the authorized GCP project id |

Never commit `.env` files (already git-ignored). Never paste a token into a notebook cell: the
generator strips values, and a hand-edited notebook is not accepted as evidence anyway.

## 3. Free-tier limits (documented 2026-10-08 — verify before relying on them)

Quotas change without notice; treat these as *approximate operational limits*, not guarantees.

| Limit | Consumer Colab | Kaggle Notebooks |
|---|---|---|
| Max session length | 12 h (hard ceiling enforced by `PLATFORM_MAX_TIMEOUT_MINUTES["colab"] = 720`) | 12 h (`... ["kaggle"] = 720`; the CLI is also passed `-t <timeout>`) |
| Idle disconnect | ~90 min idle, plus usage-based interruptions on the free tier | none documented |
| GPU quota | not published; T4/P100 availability varies by time of day | **30 GPU-hours/week** (T4×2 or P100), **20 TPU-hours/week**, resets weekly |
| Disk | ~78 GB ephemeral (not persisted across sessions) | ~20 GB `/kaggle/working` (persisted as the run's output) |
| Internet | available | must be enabled per notebook (`enable_internet: true`, requires phone-verified account) |
| Persistence | **none** — the VM is destroyed at the end of the session | `/kaggle/working` is downloadable via `kaggle kernels output` |

Consequences: keep `timeout_minutes` inside the ceiling (the spec validator refuses anything larger),
checkpoint or export before the session dies, and prefer Kaggle for unattended runs and Colab for
interactive debugging.

---

## 4. Developer-driven Colab run (the primary vehicle)

```bash
# 0. make sure the code you want to run is committed (a dirty tree is refused by default)
git status --short

# 1. generate the notebook + its RunSpec from the versioned experiment config
uv run spectraquant cloud notebook \
  --config configs/experiment/<experiment>.yaml \
  --platform colab \
  --out notebooks/generated/<run_id>.ipynb
# prints: run_id, notebook_digest, spec_sha256, and the list of cells

# 2. record the handoff (no job is started; submitted_by=human)
uv run spectraquant cloud submit --spec notebooks/generated/<run_id>.ipynb.spec.json

# 3. open notebooks/generated/<run_id>.ipynb in Colab (File → Upload notebook)
#    - Runtime → Change runtime type → GPU
#    - paste secrets into the Colab *Secrets* panel (HF_TOKEN) — never into a cell
#    - Runtime → Run all
#    The notebook installs the pinned lock, checks out the exact SHA (asserting HEAD == spec),
#    verifies dataset checksums, runs `spectraquant run`, writes run_manifest.json, exports the
#    bundle (mirrored to /content/drive/MyDrive/spectraquant-runs/<run_id> when Drive is mounted),
#    prints SPECTRAQUANT_RESULT_JSON={...} and finally the teardown marker.

# 4. download the bundle (Drive mirror, or the export directory from the notebook's last cell)

# 5. validate it locally — this is the only step that can produce a result
uv run spectraquant cloud collect --run-id <run_id> --source <downloaded dir>

# 6. inspect the record
uv run spectraquant cloud registry --status
```

`cloud collect` requires, in the bundle: a schema-valid `run_manifest.json`, a `git_commit` equal to
the spec's, every `expected_artifacts` entry present with a matching sha256 and ≥ `min_bytes`, and
the **executed** notebook (a file with execution counts — the generated template is not evidence).
Anything less is recorded as `rejected` with reasons.

## 5. Kaggle run (unattended)

```bash
export KAGGLE_USERNAME=... KAGGLE_KEY=...

uv run spectraquant cloud notebook --config configs/experiment/<experiment>.yaml \
  --platform kaggle --out notebooks/generated/<run_id>.ipynb

uv run spectraquant cloud submit --spec notebooks/generated/<run_id>.ipynb.spec.json
# → kaggle kernels push; the kernel id (<user>/<slug>) is persisted before returning

uv run spectraquant cloud status --run-id <run_id>          # bounded polling; never blocks forever
uv run spectraquant cloud fetch  --run-id <run_id> --dest artifacts/runs/<run_id>/bundle
uv run spectraquant cloud collect --run-id <run_id> --source artifacts/runs/<run_id>/bundle
```

## 6. Colab Enterprise run (paid — explicit authorization required)

```bash
export GOOGLE_CLOUD_PROJECT=<authorized project>
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/<sa>.json
export SPECTRAQUANT_ALLOW_PAID=1                  # only after you authorized the spend
export SPECTRAQUANT_CLOUD_IMAGE=gcr.io/<project>/spectraquant:<tag>   # run container
export SPECTRAQUANT_GCS_BUCKET=gs://<staging bucket>                  # notebook + artifact staging

uv run spectraquant cloud notebook --config configs/experiment/<experiment>.yaml \
  --platform colab_enterprise --out notebooks/generated/<run_id>.ipynb \
  --max-cost-usd 20        # required: the ceiling you authorize

uv run spectraquant cloud submit --spec notebooks/generated/<run_id>.ipynb.spec.json
```

The adapter refuses the submission unless a project is configured **and** the budget guard passes
(`SPECTRAQUANT_ALLOW_PAID=1`, a ceiling ≤ the documented maximum, and an estimated cost inside that
ceiling). The estimate is class-1 arithmetic — `timeout_minutes / 60 × rate` with the rate in
`spectraquant.cloud.budget.PLATFORM_RATES_USD_PER_HOUR` — and it is labelled an *estimate*, never a
bill. Artifacts land in `gs://<bucket>/spectraquant/runs/<run_id>/`; download that prefix and run
`cloud collect`.

## 7. Resume after a pre-emption

The remote run id is persisted in `artifacts/runs/registry.jsonl` at submission time, so an
interruption between submit and poll cannot orphan a running job.

```bash
uv run spectraquant cloud registry --run-id <run_id>   # see the last recorded state
uv run spectraquant cloud status   --run-id <run_id>   # re-attach to the existing remote run
```

* `submit` is **idempotent**: if a remote id is already persisted and the run is not terminal,
  `submit` returns that id and does **not** push again (`can_resume(run_id) == True`).
* If the session died mid-run, re-run `fetch` + `collect` on whatever was exported; a run whose
  artifacts fail validation is recorded as `rejected` with its reasons — never silently retried into
  a success.
* A truncated run is detectable: the export cell prints `SPECTRAQUANT_RESULT_JSON={...}` and the last
  cell prints `SPECTRAQUANT_TEARDOWN_OK {...}`. `collect` reports `teardown_observed` for the bundle.
* For a Colab handoff, "resume" means re-uploading the same generated notebook: it is byte-identical
  for the same spec (check `notebook_digest`), so the run id and record stay consistent.

## 8. Registry format (`artifacts/runs/registry.jsonl`, git-ignored)

One JSON object per line, one line per state transition, appended in order:

```json
{"state": "validated", "run_id": "...", "platform": "kaggle", "remote_id": "user/slug",
 "timestamp_utc": "...", "git_commit": "...", "spec_sha256": "...", "notebook_digest": "...",
 "measurement_class": 1, "artifact_checksums": {"artifacts/result.json": "sha256:..."},
 "manifest_sha256": "sha256:...", "failure_reason": null, "submitted_by": "agent",
 "detail": "collection validated", "spec": {...}, "metrics": {...}, "checksum_verified": true}
```

States: `submitted → running → finished|failed → collected → validated|rejected` (a same-state repeat
is allowed for idempotency; everything else is refused). `metrics` appears **only** on a `validated`
line, and `validated` is unreachable unless the artifact checksums were computed from the downloaded
files (`checksum_verified`). There is no API that records a metric from anything else.

## 9. Environment variables understood by the tooling

| Variable | Effect |
|---|---|
| `SPECTRAQUANT_RUNS_DIR` | root of the per-run directories and the registry (default `artifacts/runs`) |
| `SPECTRAQUANT_NOTEBOOK_DIR` | where generated notebooks go (default `notebooks/generated`) |
| `SPECTRAQUANT_REPO_URL` | clone URL recorded in a spec when it cannot be read from `git remote` |
| `SPECTRAQUANT_WORKDIR` | (in-notebook) work directory for the checkout and artifacts |
| `SPECTRAQUANT_EXPORT_DIR` | (in-notebook) where the bundle is exported before download |
| `SPECTRAQUANT_NOTEBOOK_PATH` | (in-notebook) explicit path of the executed notebook to export |
| `SPECTRAQUANT_REMOTE_RUN_ID`, `SPECTRAQUANT_SUBMITTED_BY` | recorded in the manifest for provenance |

## 10. Regenerating the committed example

```bash
uv run python scripts/cloud/generate_smoke_notebook.py            # notebooks/generated/smoke-cloud-example.ipynb
uv run python scripts/cloud/generate_smoke_notebook.py --json
```

The generator is deterministic for a given `(spec, template version)`: regenerating on the same commit
reproduces the same `notebook_digest`. A hand-edited notebook cannot be regenerated and is therefore
not accepted as evidence (`AGENTS.md` §2b rule 2).

## 11. Default pinned install

`spec_from_config` records `install_spec = "uv export --frozen --no-dev --no-hashes --extra cloud -o
requirements-cloud.txt && uv pip install --system -r requirements-cloud.txt && uv pip install --system
--no-deps -e ."`, i.e. the committed `uv.lock` is materialised into the platform interpreter and the
checked-out package is installed editable. Override it with `cloud notebook --install-spec "<cmd>"`
for a platform-specific bootstrap; whatever you pass is executed verbatim and must not silently
resolve newer versions.
