# ADR-0001 — Python 3.11 with `uv` as the only dependency workflow

* **Status:** Accepted
* **Date:** 2026-10-08
* **Deciders:** repository bootstrap (infra stream)
* **Supersedes:** none

## Context

SpectraQuant is executed on a CPU-only Windows workstation and must be reproducible on Linux CI and
in Docker (see `docs/research/environment.md`). The dependency set includes PyTorch, whose default
PyPI wheels are CUDA-enabled and large; the project is forbidden from depending on CUDA
(`AGENTS.md` section 2), so the CPU wheel index must be selected at resolution time, not by
post-install patching.

The environment of record: Python 3.11.16 (uv-managed), uv 0.12.15, Windows 11, Docker Desktop,
`gh` CLI. A system Python 3.13 also exists on the machine but has no PyTorch installed and is not
the project interpreter.

Options for the interpreter:

1. **Python 3.11** — the version uv already manages on the workstation, the version with the
   widest PyTorch/CPU wheel coverage, and the version every upstream we must reimplement
   (LR-QAT, LoftQ, LQ-LoRA) is tested against.
2. Python 3.12 / 3.13 — newer, but adds wheel-availability risk for the pinned CPU index and for
   future optional packages (bitsandbytes-style kernels, evaluation harnesses) that we will pin
   later behind the GPU gate.
3. Multiple versions with a matrix — triples CI time for no research benefit at this stage.

Options for dependency management:

1. **uv** with `pyproject.toml` + committed `uv.lock`, plus an explicit, `explicit = true` PyTorch
   CPU index.
2. `pip` + hand-maintained `requirements.txt` — no resolver guarantee, no cross-platform lock, and
   the CPU-index requirement becomes a documented manual step that will eventually be forgotten.
3. Poetry / PDM — equivalent capability, but neither is installed on the workstation of record and
   `uv` is already the documented tool in `AGENTS.md` section 7.
4. Conda — heavy, platform-specific, and unnecessary for a pure-Python + PyTorch project.

## Decision

Use **Python `>=3.11,<3.12`** and **`uv`** as the only supported dependency workflow:

* `pyproject.toml` declares the runtime dependencies exactly:
  `torch`, `numpy`, `hydra-core`, `omegaconf`, `pydantic>=2`, `typer`, `rich`, `jsonschema`.
* Optional extras: `dev` (`pytest`, `pytest-cov`, `hypothesis`, `ruff`, `pyright`) and `track`
  (`mlflow-skinny`, see ADR-0003).
* `uv.lock` is committed and regenerated only by `uv sync`/`uv lock`; it is never hand-edited.
* The CPU wheel is forced cross-platform:

  ```toml
  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true

  [tool.uv.sources]
  torch = { index = "pytorch-cpu" }
  ```

* `[project.scripts] spectraquant = "spectraquant.cli.main:app"` is the single entry point.
* Ruff (lint + format, line length 100, target `py311`), Pyright on `src`, and pytest with
  `testpaths = ["tests"]` are configured in `pyproject.toml` — one file, no scattered tool config.

## Consequences

**Positive**

* One command (`uv sync --all-extras`) produces an identical environment on Windows, Linux CI and
  inside `docker/Dockerfile` (`uv sync --frozen`), with the CPU wheel guaranteed by construction.
* CUDA-only packages cannot sneak in through a transitive dependency: the PyTorch index is
  `explicit`, so only `torch` may resolve from it, and the dependency list is deliberately short.
* `requires-python = ">=3.11,<3.12"` makes a 3.12+ interpreter fail immediately with a clear message
  instead of producing subtly different numerics.

**Negative / accepted costs**

* Contributors without uv must install it; `requirements-dev.txt` exists only as a convenience
  mirror and is explicitly not authoritative.
* Python 3.11 means no newer stdlib niceties; the code targets 3.11 syntax (`X | None`, no
  `TypeAlias` statements in favour of plain aliases).
* Bumping Python or PyTorch is a deliberate, reviewable lockfile change rather than an automatic
  update.

## Alternatives considered (summary)

* pip/requirements — rejected: no lock guarantee, CPU index becomes tribal knowledge.
* Poetry/PDM/Conda — rejected: capability parity at best, extra tooling not present on the
  workstation of record.
* Python 3.12/3.13 — rejected for now: wheel-coverage risk against the pinned CPU index for the
  upstream methods we must reproduce; revisit only with a measurement-backed reason.

## References

* `AGENTS.md` sections 2, 7 and 8
* `docs/research/environment.md` (observed toolchain)
* `pyproject.toml`, `uv.lock`, `Makefile`, `.github/workflows/ci.yml`
