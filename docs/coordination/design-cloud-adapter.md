# Design note — cloud execution adapter (frozen)

Status: **frozen by the orchestrator** 2026-10-08, after the cloud-compute execution policy
(`AGENTS.md` §2b). Implementation may not change a signature listed here without amending this file.

## 0. Why this exists

The local workstation has no usable CUDA GPU and, under the policy, may not run research training at
all. Every Tier 1–5 experiment therefore executes on a cloud notebook substrate and returns a
checksum-verified artifact bundle. The adapter is the only sanctioned path between the repository and
that substrate; it is deliberately boring and inspectable, and it must be impossible for it to record
a result that was not actually produced.

## 1. Platform contract (honest capabilities)

| Platform | Automation reality | Adapter behaviour |
|---|---|---|
| Google Colab (developer's chosen vehicle) | consumer Colab exposes **no official submission API** | `build_notebook()` emits a ready-to-open, self-contained notebook; the developer (or a scheduled trigger) runs it; artifacts are exported to Drive/HF and pulled back with `collect()`. The adapter records `submitted_by: "human"` — it never pretends to have started the run |
| Kaggle Notebooks | fully programmatic via the official CLI | `submit()` → `kaggle kernels push`, run id persisted, `poll()` → `kaggle kernels status`, `fetch()` → `kaggle kernels output` |
| Colab Enterprise (optional) | programmatic via GCP SDK/REST | enabled only when an authorized GCP project is configured **and** `SPECTRAQUANT_ALLOW_PAID=1` |
| `local_cpu` | the workstation, for Tier-0 fixtures and class 4-CPU measurement only | no training; refuses `gpu_required=True` |

## 2. `spectraquant.cloud.spec` — the only input

```python
class RunSpec(BaseModel):            # frozen
    run_id: str                      # slug, unique, stable across retries
    experiment_config: str           # path under configs/experiment/
    overrides: list[str]             # hydra overrides, recorded verbatim
    platform: Literal["colab", "kaggle", "colab_enterprise", "local_cpu"]
    repo_url: str
    git_commit: str                  # resolved SHA; dirty tree => refuse unless allow_dirty
    allow_dirty: bool = False
    python_version: str = "3.11"
    install_spec: str                # how the env is materialised (uv sync at commit / pinned wheels)
    dataset_refs: list[DatasetRef]   # name, revision, split, checksum
    seeds: list[int]
    gpu_required: bool
    timeout_minutes: int
    max_cost_authorized_usd: float | None   # required for paid platforms
    expected_artifacts: list[ExpectedArtifact]  # name, sha256 (optional), min_bytes
    measurement_class_expected: int  # 1..4 (4 = 4-CPU or 4-GPU per the spec's platform)
```

`RunSpec` is validated before anything is emitted; an unset `max_cost_authorized_usd` on a paid
platform is a hard error, not a warning.

## 3. `spectraquant.cloud.notebook`

```python
def build_notebook(spec: RunSpec) -> nbformat.NotebookNode   # pure function of spec (+ template version)
def notebook_digest(spec: RunSpec) -> str                    # sha256 of the serialized notebook
def write_notebook(spec: RunSpec, out_path: str) -> str      # returns digest
```

Mandatory cells, in order:

1. **Environment record** — `nvidia-smi` (if any), `python -V`, `pip freeze`/`uv pip freeze`, CPU
   model, RAM, platform name, UTC start time.
2. **Install** — the pinned `install_spec`; must not silently resolve newer versions.
3. **Repo** — clone/fetch `repo_url`, `git checkout <git_commit>`, assert
   `git rev-parse HEAD == spec.git_commit`.
4. **Data** — fetch `dataset_refs`, verify checksums, fail loudly on mismatch.
5. **Run** — `spectraquant run --config <experiment_config> <overrides>`; no scientific logic inline.
6. **Manifest** — write `run_manifest.json` (schema in `artifacts/schemas/`) including hardware,
   GPU-hours, start/end, status, metrics, artifact checksums.
7. **Export** — copy artifacts + manifest + the *executed* notebook to the export target, and print a
   single machine-readable line `SPECTRAQUANT_RESULT_JSON={...}` for log-based collection.
8. **Teardown** — explicit end marker so a truncated run is detectable.

Rules: a notebook is **generated**, never hand-edited (hand edits are overwritten); `build_notebook`
is deterministic for a given spec + template version, and the digest is recorded in the manifest;
no credentials are ever emitted into cells (they are read from the platform's secret store or env).

## 4. `spectraquant.cloud.adapters`

```python
class PlatformAdapter(Protocol):
    name: str
    def submit(self, spec: RunSpec, notebook_path: str) -> str          # -> remote run id
    def status(self, run_id: str) -> RunStatus                          # bounded-retry polling
    def fetch(self, run_id: str, dest_dir: str) -> FetchReport          # logs, artifacts, executed nb
    def can_resume(self, run_id: str) -> bool
```

`submit()` MUST persist the remote run id to the registry **before** returning, so an interruption
between submit and poll cannot orphan a running job. `status()` uses bounded exponential backoff with
a documented maximum and never blocks indefinitely.

## 5. `spectraquant.cloud.registry` + `collect`

- Append-only JSONL registry at `artifacts/runs/registry.jsonl` (git-ignored except a schema sample):
  one line per state transition — `submitted`, `running`, `finished`, `failed`, `collected`,
  `validated`, `rejected` — with run id, platform, remote id, timestamps, git commit, config hash,
  measurement class, artifact checksums, failure reason.
- `collect(run_id, source_dir) -> ValidationReport` verifies: `run_manifest.json` exists and validates
  against the schema; git commit matches the spec; every `expected_artifacts` entry is present with a
  matching sha256 and ≥ `min_bytes`; the executed notebook is present. **Only** on full success may
  the registry record `validated` and expose metrics to analysis.
- **No fabrication rule:** the registry API has no method that writes metrics from anything other than
  a validated `run_manifest.json`. A missing artifact yields `rejected`, never a partial success.

## 6. `spectraquant.cloud.budget`

```python
def assert_submission_allowed(spec: RunSpec, env: Mapping[str, str]) -> None
```

Raises unless: the platform is free-tier, or (`SPECTRAQUANT_ALLOW_PAID == "1"` **and**
`spec.max_cost_authorized_usd` is set and ≤ the platform's documented ceiling). Estimated GPU-hours ×
a documented rate are logged. No paid resource is started without the user's prior authorization.

## 7. CLI surface

```
spectraquant cloud notebook --config configs/experiment/<x>.yaml --platform colab --out notebooks/generated/<x>.ipynb
spectraquant cloud submit   --spec <run_spec.json>
spectraquant cloud status   --run-id <id>
spectraquant cloud fetch    --run-id <id> --dest artifacts/runs/<id>/
spectraquant cloud collect  --run-id <id> --source artifacts/runs/<id>/
spectraquant cloud registry --status
```

## 8. Secrets

Read only from environment/platform secret stores: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `HF_TOKEN`,
`GOOGLE_APPLICATION_CREDENTIALS`, `SPECTRAQUANT_ALLOW_PAID`. A redaction helper MUST scrub these
values from every log line, error message and emitted notebook, and a unit test MUST assert that a
secret placed in the environment never appears in generated output.

## 9. Tests (owner of this slice)

- `build_notebook` determinism: same spec → byte-identical notebook; different commit → different digest.
- Generated notebook contains no secret string (injected via env in the test).
- `RunSpec` validation: paid platform without authorized cost → error; missing git commit → error.
- Registry: `validated` is unreachable without a checksum-matching artifact set (negative test).
- `collect` rejects a manifest whose git commit differs from the spec (negative test).
- Resume: an interrupted submit→poll cycle re-uses the persisted remote id and does not re-submit.
- Budget guard: free tier passes; paid platform without `SPECTRAQUANT_ALLOW_PAID` raises.
- A synthetic end-to-end test: fake adapter + fake artifact dir → `submit → fetch → collect` yields a
  `validated` registry entry and a parsed manifest (no network).

## 10. Dependencies

`nbformat>=5.10`, `kaggle>=1.7` in an optional extra `cloud`; Colab-Enterprise extras kept separate
and optional. The core package must import without either (adapters import lazily and raise a clear
error naming the missing extra).

---

## 11. Amendment 2026-10-09 — the frozen plan carries its runner (addendum, orchestrator-approved)

Status: **addendum to the frozen contract.** §2's `RunSpec` and §3's run cell gain one optional field;
nothing in §2 is removed, renamed or reinterpreted. Reason, in one sentence: the frozen artifacts the
preregistration references are **plans** (`configs/{tier1,tier2,repro}/*.yaml`, a `PlanConfig`), not
Hydra `ExperimentConfig` documents, and §3's run cell hard-coded `spectraquant run --config …`, so a
plan could not be turned into a runnable notebook without hand-building a spec — which the policy
forbids.

### 11.1 `RunSpec.runner_command: str | None = None`

* Additive and optional: an existing spec (and an existing generated notebook) keeps the frozen
  default, `spectraquant run --config <experiment_config> <overrides>`.
* Validated to be a `spectraquant` invocation **without** the interpreter: the run cell executes
  `[sys.executable, "-m", *shlex.split(runner_command)]`, so the first token must be `spectraquant`.
  A `bash -c`, a downloaded script or a heredoc is refused — logic may not move into a cell
  (`AGENTS.md` §2b rule 1).
* Recorded in the generated header and in the notebook metadata; `NOTEBOOK_TEMPLATE_VERSION` moves to
  `1.1.0` because the cell text changed (the digest of every notebook changes with it).

### 11.2 What a plan spec records, and what it does not

`plan_to_run_spec(plan, *, platform, repo_url=None, git_commit=None, allow_dirty=None, out_dir=None)`
maps the plan's identity into §2's fields. Three mappings deserve to be written down because they are
places the spec is *narrower* than the plan, and in each the plan wins:

1. `experiment_config` records the **plan path**. The notebook's manifest cell therefore resolves it
   as a plan (`spectraquant.cloud.plan_data.is_plan_document`) instead of composing it with Hydra;
   the wrapper manifest it writes is provenance only (`compression.method="none"`, no measurement
   class) and the per-arm manifests the runner wrote are the evidence. `resolved_config` carries the
   plan document verbatim.
2. `DatasetRef.checksum` has a second spelling — a **declaration digest**
   (`spectraquant.cloud.spec.dataset_declaration_digest`) over `(name, revision, split)`. A plan pins a
   dataset by an immutable Hugging Face commit; the payload cannot be hashed locally without
   downloading up to ~1 TB, and the run must not silently accept an unpinned revision. The declaration
   digest detects a tampered spec; the pin itself is verified upstream by
   `spectraquant.cloud.plan_data.verify_pinned_revision`, which *fails loudly* on a floating revision
   or on a commit mismatch. A class-3 payload figure is never derived from it.
3. `timeout_minutes` = `cost.platform_hours_max * 60`, `max_cost_authorized_usd` =
   `cost.max_cost_authorized_usd`, `measurement_class_expected` = the plan's maximum authorised class
   capped at 4, and `gpu_required` is `True` unless the plan's substrate is `local_cpu`. A
   free-tier-only plan may not target a paid platform (refused at spec-build time).

### 11.3 `spectraquant run-plan` — the remote entry point

The plan runner (`src/spectraquant/cloud/plan_runner.py`) executes the plan's arms on a real
pretrained model and writes one schema-validated `run_manifest.json` plus `metrics.json` per
(arm, seed), an invocation-level `<out>/metrics.json`, and the single `SPECTRAQUANT_RESULT_JSON=` line
of §3's export cell. This slice implements exactly the four non-trainable arms
(`fp16_reference`, `ptq_uniform`, `low_rank_only`, `rank_then_quant`); every other kind — including
all trainable arms — raises `NotImplementedError` naming its owning milestone, and an arm the plan
does not bind to a grid point (the rank arms) requires an explicit `--rank` rather than a default the
plan never authorised. Model and dataset revisions are recorded as loaded and compared against the
pin; a mismatch is a hard error. The perplexity protocol is the runner's own documented
non-overlapping-window token-level CE (recorded in `metrics["perplexity.method"]`), **not** the frozen
P1 `lm-evaluation-harness` cell of `eval-protocol.md` §6.1, which remains unwired.

### 11.4 Dependency extra

Loading a pretrained model/dataset requires `transformers` and `datasets`, declared in the new
`models` extra (`transformers>=4.44,<5`, `datasets>=2.20,<3`; the `<3` pin keeps
commit-pinned loading scripts loadable). They are imported lazily and their absence raises naming the
extra, exactly like the `cloud` extra. A plan spec therefore records
`install_spec = "uv sync --frozen --extra cloud --extra models"`.

### 11.5 One additive schema change

`artifacts/schemas/run-manifest.schema.json` gains `"svd"` in `compression.method`'s enum: a pure
rank-truncation arm (`low_rank_only`) must be recordable as what it is, and labelling it `rtn` or
`none` would misdescribe the compression actually applied. Additive only; no existing manifest
changes meaning.
