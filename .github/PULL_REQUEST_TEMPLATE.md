## What this changes

<!-- One paragraph. What behaviour changes, and for which module / config / doc. -->

## Why

<!-- The question or defect this addresses. Link the issue, ADR or preregistration clause. -->

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [ ] `uv run pyright` passes on `src`.
- [ ] `uv run pytest -q` passes on CPU (no skipped CPU tests added).
- [ ] Any new number carries a measurement class (1-5) or is explicitly marked "not measured".
- [ ] No model weights, datasets, credentials or machine-specific absolute paths are committed.
- [ ] New shared interfaces got a design note under `docs/decisions/`.
- [ ] Determinism preserved: new code paths are seeded and covered by a reproducibility test.
- [ ] If this changes the research plan, `docs/research/preregistration-amendments.md` is updated.

## Agent report (AGENTS.md section 10)

1. **Scope completed:**
2. **Files changed:**
3. **Commands executed:**
4. **Tests and results:**
5. **Evidence created:**
6. **Assumptions:**
7. **Risks / unresolved questions:**
8. **Recommended next action:**

## Raw verification output

<!-- Paste the command output that proves the change works. Never a paraphrase. -->
