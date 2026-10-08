# Path ownership

Two agents MUST NOT edit the same file concurrently. Before starting work, claim the paths for your
stream; if a path you need is already claimed by an active stream, request the change from that
stream's owner or open a design note under `docs/decisions/`.

Legend: **A–M** = specialist agents from the master specification. Wave 1 = active now.

## Assignment table

| Path / pattern | Owner (stream) | Notes |
|---|---|---|
| `AGENTS.md` | orchestrator | only the orchestrator edits |
| `.gitignore`, `.gitattributes` | orchestrator | |
| `docs/research/environment.md` | orchestrator | re-audit on hardware change |
| `docs/coordination/**` | orchestrator | ownership, status, decision log |
| `docs/research/charter.md` | A | distilled scope + gates |
| `docs/research/literature-review.md` | A | |
| `docs/research/literature-matrix.csv` | A | |
| `docs/research/preregistration.md` | A | frozen before Milestone 7 |
| `docs/research/preregistration-amendments.md` | A | append-only, timestamped |
| `docs/research/novelty-risk.md` | A | |
| `docs/research/upstream-lockfile.md` | A | pinned commits + licenses |
| `docs/research/risk-register.md` | A | |
| `references/**` | A | BibTeX, method-card templates |
| `docs/protocols/measurement-taxonomy.md` | D-lite (wave 1) | expands AGENTS.md §5 |
| `docs/research/backend-capability.md` | D-lite (wave 1) | live compatibility matrix |
| `pyproject.toml`, `uv.lock`, `Makefile`, `.pre-commit-config.yaml` | B | |
| `.github/**` | B | CI, templates |
| `docker/**` | B | |
| `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `LICENSE`, `CITATION.cff` | B | Community files |
| `docs/architecture/**`, `docs/decisions/ADR-000{1,2,3}-*.md` | B | |
| `src/spectraquant/**` (initial skeleton + CLI + smoke) | B | after wave 1: split per package |
| `tests/{unit,integration,regression}/test_{config,seeds,manifest,smoke}*.py` | B | |
| `configs/**`, `scripts/**`, `artifacts/schemas/**` | B | |
| `artifacts/manifests/**` | B (writer via code), J (audit) | |
| `src/spectraquant/cloud/**`, `notebooks/generated/**`, `scripts/cloud/**`, `tests/unit/test_cloud_*.py` | K (cloud adapter slice, wave 2) | contract: `docs/coordination/design-cloud-adapter.md` |
| `docker/**` (cloud images, if any) | B | |

## Reserved for later waves (not yet claimed)

| Path | Owner |
|---|---|
| `src/spectraquant/{quantization,benchmarking}/**`, `tests/**/test_quant*` | D |
| `src/spectraquant/{factorization}/**`, spectral summaries | E |
| `src/spectraquant/{proxies}/**` | F |
| `src/spectraquant/{regularizers,training}/**` | G |
| `src/spectraquant/{allocation}/**` | H |
| `src/spectraquant/data/**`, evaluation protocol, contamination audit | I |
| `docs/results/**` registry + statistical analysis | J |
| `src/spectraquant/benchmarking/**` perf harness, `docker/` service assets | K |
| `docs/results/verification/**` | L |
| `reports/**`, `notebooks/final/**`, `CAREER_EVIDENCE.md` | M |

## Conflict rule

Wave-1 streams: **A** (research docs), **B** (repository scaffold), **D-lite** (protocol docs),
orchestrator (contract + coordination). Their path sets are disjoint by construction. If a stream
needs a file owned by another stream, it messages that stream's agent instead of editing.
