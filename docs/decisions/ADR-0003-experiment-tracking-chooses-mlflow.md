# ADR-0003 — One experiment tracker: MLflow (skinny), local files only

* **Status:** Accepted
* **Date:** 2026-10-08
* **Deciders:** repository bootstrap (infra stream)
* **Supersedes:** none

## Context

Runs must be comparable across seeds, methods and memory budgets. The authoritative record of a run
is already the JSON manifest validated against `artifacts/schemas/run-manifest.schema.json`
(`AGENTS.md` section 7): it carries the git commit, resolved config, dataset checksums, hardware,
software versions, compression plan, measurement class and metrics.

A tracker adds value beyond the manifest — run-to-run curves, sweep comparison, filtering by
hyperparameter — but only if it stays subordinate to the manifest and does not become a second,
divergent source of truth. Additional constraints:

* The workstation is offline-friendly and CPU-only; a tracker that requires a server, a database or
  a cloud account is a liability.
* `AGENTS.md` section 7 mandates **one** tracker, not a zoo.
* `AGENTS.md` section 4.12 forbids infrastructure without a measured requirement — no Kafka, no
  Kubernetes, and equally no hosted tracking service.

Options:

1. **MLflow with `mlflow-skinny`** — the tracker without the server, UI and SQL dependencies; runs
   log to a local `mlruns/` directory (already git-ignored).
2. Weights & Biases — excellent UX, but a hosted account becomes a dependency of the core path, and
   offline mode is a second-class experience.
3. TensorBoard — great for curves, no notion of a run with parameters/metrics/tags; would need a
   bespoke parameter store anyway.
4. Aim — comparable to MLflow, smaller community, and no existing familiarity on this project.
5. No tracker; parse the JSON manifests ourselves — zero dependency, but every comparison becomes a
   bespoke script and sweep analysis is re-invented badly.

## Decision

Adopt **`mlflow-skinny` as the single optional tracker**, declared as the `track` extra:

* It is **optional**: `uv sync --all-extras` installs it, a plain `uv sync` does not. Nothing in
  `src/spectraquant` imports it at module scope; the manifest remains the mandatory record.
* Runs log locally to `mlruns/` (git-ignored). No server, no database, no remote tracking URI, no
  credentials in the repository.
* The manifest is written first and is the source of truth. A tracked run must be reconstructible
  from its manifest alone: the tracker stores a convenience copy of parameters and metrics, never
  exclusive information.
* No other tracking library may be added. If MLflow proves inadequate, that is a new ADR that
  supersedes this one.

## Consequences

**Positive**

* Sweep comparison (rank/bit budget vs quality at equal memory) gets a real UI/API without any
  hosted dependency, and without a server to run on a 16 GB workstation.
* The core path stays installable with the minimal dependency set; CI does not need the tracker.
* Because the manifest is authoritative, a lost or corrupt `mlruns/` directory costs convenience,
  not evidence.

**Negative / accepted costs**

* `mlflow-skinny` still pulls a nontrivial transitive tree (observed in `uv.lock`: `sqlparse`,
  `databricks-sdk`, `cryptography`, `uvicorn`, `starlette`); it is confined to the optional extra so
  the core install stays small.
* Two places now describe a run (manifest + MLflow record). The rule "manifest first, tracker
  copies" is the mitigation, and it must be enforced in review.
* MLflow's own schema is looser than our manifest schema; nothing may be inferred from MLflow that
  is not also in the manifest.

## Alternatives considered (summary)

* W&B — rejected: hosted account in the core path, offline mode secondary.
* TensorBoard — rejected: no parameter/metric/tag model for sweep comparison.
* Aim — rejected: parity at best, no existing usage on the project.
* Manifests only — rejected: re-implements sweep analysis badly; the manifest stays authoritative
  regardless.

## References

* `AGENTS.md` sections 4.12, 7
* `pyproject.toml` (`[project.optional-dependencies] track`)
* `docs/architecture/system.md` (run data flow)
