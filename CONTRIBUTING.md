# Contributing to SpectraQuant

Read [`AGENTS.md`](AGENTS.md) first. It is the operating contract: compute constraints, scientific
boundaries, the measurement taxonomy and the reporting format all live there, and it wins over any
other document when they disagree.

## Environment

```bash
uv sync --all-extras          # the only supported way to create the environment
uv run pytest -q              # CPU test suite
uv run ruff check . && uv run ruff format --check .
uv run pyright                # type check of src/spectraquant
uv run spectraquant env       # what machine/versions your run will record
```

`uv` is required (ADR-0001). `uv.lock` is committed and authoritative: change dependencies in
`pyproject.toml`, then run `uv sync` and commit the regenerated lockfile. Never hand-edit
`uv.lock`. `make check` runs the same four checks as CI.

Optional: `uv run pre-commit install` wires the hooks in `.pre-commit-config.yaml`.

## What must not be committed

Model weights, datasets, credentials, tokens, caches, large raw outputs, machine-specific absolute
paths, and anything under the ignored experiment-output directories. Manifest files under
`artifacts/sample-results/` are the deliberate exception: they are small, reviewed and part of the
reproducibility contract.

## Working rules

1. **Branch per stream.** Never develop on `main`. Prefixes: `research/`, `feat/`, `infra/`,
   `docs/`, `fix/`, `eval/`, `perf/`.
2. **Conventional commits.** `feat:`, `fix:`, `research:`, `eval:`, `infra:`, `docs:`, `test:`,
   `chore:`. One coherent unit per commit.
3. **Ownership.** Path ownership is in `docs/coordination/ownership.md`. Two agents never edit the
   same file concurrently; new shared interfaces get a design note under `docs/decisions/`.
4. **Determinism.** Any new code path that produces numbers must be seeded through
   `spectraquant.training.seeding.seed_everything` and must have a test that runs it twice and
   compares results.
5. **Manifests.** A run is not "done" until it has written a manifest that validates against
   `artifacts/schemas/run-manifest.schema.json`. Adding a field means changing the schema *and* the
   tests in `tests/unit/test_manifest.py`.
6. **Honest scope.** Tier 2-5 experiments need an external GPU that is not yet available. They may
   be described as *planned*; never as done, and never with an estimated stand-in number. Any code
   path that would need CUDA must raise `NotImplementedError` rather than approximate.
7. **No fake results.** Never present an approximation as exact (AGENTS.md section 4.13) and never
   report fake quantization as storage or speed.

## Tests

* CPU only, deterministic, no network.
* Follow the conventions in `tests/`: `unit/` for pure logic, `integration/` for the tiny end-to-end
  run, `regression/` for committed artifacts.
* A bug fix starts with a failing test that reproduces the bug; a feature starts with the observable
  behaviour you expect.
* Prefer a throwaway script for exploratory checks and a permanent test only for behaviour a
  consumer could plausibly break.

## Reporting your work

Every contribution ends with the eight-point report from `AGENTS.md` section 10 (also mirrored in
`.github/PULL_REQUEST_TEMPLATE.md`): scope, files changed, commands, tests and results, evidence,
assumptions, risks, next action. Paste raw command output as evidence — never a paraphrase. Report
negative and inconclusive results directly.
