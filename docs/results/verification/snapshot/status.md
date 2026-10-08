# Coordination status

Last updated: 2026-10-08 by the orchestrator. Update this file at every gate; it is the single
source of truth for "what is done, what is claimed, what is blocked".

## 1. Repository state

| Item | Value |
|---|---|
| Remote | `https://github.com/OussemaHarrabi/SpectraQuant.git` (PUBLIC, created 2026-10-08T19:55:34Z) |
| Default branch | `main` (created with the first commit on this branch set; initially empty) |
| Active branch | `infra/bootstrap` |
| Branch strategy | one branch per stream: `research/`, `feat/`, `infra/`, `docs/`, `fix/`, `eval/`, `perf/` |
| Local path | `C:\Users\oussa\oussema\SpectraQuant` |
| Git identity | `Oussema Harrabi <189178358+OussemaHarrabi@users.noreply.github.com>` (local repo config only) |

## 2. Compute

CPU-only Windows workstation; **no CUDA**, AMD iGPU unusable, 16.23 GB RAM. Full record:
`docs/research/environment.md`. Consequence: Tiers 0–1 executable locally; Tiers 2–5 are
**planned/GPU-gated** and may not be reported as completed. Measurement classes 4–5 unavailable.

Compute consumed so far: negligible (bootstrap + documentation only). No GPU spend, no cloud cost.

## 3. Completed work

| # | Work | Evidence | Status |
|---|---|---|---|
| 1 | Non-destructive bootstrap: verified empty local dir + empty remote, `git init`, remote, `infra/bootstrap`, identity, line-ending policy | `git remote -v`, `gh repo view`, `git config --local --list` | done |
| 2 | Hardware/software audit | `docs/research/environment.md` (command transcript embedded) | done |
| 3 | `AGENTS.md` operating contract (compute rules, scientific boundaries, measurement taxonomy, commands, git protocol, agent report contract) | `AGENTS.md` | done |
| 4 | Path ownership map for all specialist streams | `docs/coordination/ownership.md` | done |
| 5 | Python 3.11.16 installed via uv | `uv python list` | done |
| 6 | Scientific foundation: charter, literature review + 50-row matrix, novelty/prior-art risk, preregistration with traceability, risk register, upstream lockfile with 8 verified SHAs + licenses, BibTeX (50 verified entries), method-card template | `docs/research/**`, `references/**`; SHAs re-verified independently by the orchestrator (LR-QAT `8795afe0…`, LoftQ `ae33fd4f…`, lq-lora `c2424b3a…`, pytorch/ao `cff77b46…`, lm-eval `ddd67220…` peeled from tag `v0.4.13`); CSV parses 50×19, 0 duplicate keys | delivered, pending M1 gate |
| 7 | Backend capability + measurement protocols: taxonomy operations, 23-row live compatibility matrix, benchmark protocol, byte-accounting rules with worked examples | `docs/research/backend-capability.md`, `docs/protocols/{measurement-taxonomy,benchmark-protocol,memory-accounting}.md` | delivered (revision in flight) |
| 8 | Model/dataset licence + revision inventory and frozen evaluation protocol (5-role split separation, 6-task harness suite, pinned harness commit, cost per local cell) | `docs/research/model-dataset-licenses.md`, `docs/protocols/eval-protocol.md`; revisions/licences re-verified independently (TinyLlama-Chat `fe8a4ea1…`, Qwen2.5-0.5B-Instruct `7ae55760…`, SmolLM2-360M-Instruct `a10cc151…`, deit-base `a0fc9b37…`, all apache-2.0; SlimPajama-6B licence genuinely absent → `[UNRESOLVED]`) | delivered |
| 9 | **Compute-envelope correction (scope expansion):** a real CPU low-bit kernel path exists — ONNX Runtime 1.30.0 CPU executes `MatMulNBits` (int4) and `MatMulInteger` (int8) on artifacts we serialize; torchao 0.18.0 `IntxWeightOnlyConfig(torch.int4, PerGroup(32))` forwards on CPU; bitsandbytes 0.50.2 has a Windows CPU backend; Docker Linux containers work (16 CPU / 7.318 GiB) | Orchestrator's independent reproduction: int4 artifact 1501 B = 1024 B payload + 256 B scales, `MatMulNBits` node, max abs err 2.369; int8 artifact 2627 B, err 0.216; `docker run alpine` OK. Recorded in `AGENTS.md` §5 (class 4 split: 4-CPU available / 4-GPU deferred), `docs/research/environment.md` §1, amendment `A-0003` | done |

## 4. In flight (wave 1)

| Agent | Scope | Owned paths | Expected evidence |
|---|---|---|---|
| `LitAudit` (A) | literature review, matrix, novelty risk, preregistration, upstream lockfile, risk register, BibTeX | `docs/research/{charter,literature-review,literature-matrix.csv,preregistration,preregistration-amendments,novelty-risk,upstream-lockfile,risk-register}`, `references/**` | verified commit SHAs, CSV parse check, falsification criteria |
| `Scaffold` (B) | packaging (uv/pyproject/lock), typed package skeleton, CLI, manifest schema+validation, logging/seeding, CPU CI, Docker, pre-commit, community files, ADRs, tests, tiny deterministic e2e smoke | `pyproject.toml`, `uv.lock`, `Makefile`, `.pre-commit-config.yaml`, `.github/**`, `docker/**`, `src/spectraquant/**`, `tests/**`, `configs/**`, `scripts/**`, `artifacts/schemas/**`, `docs/architecture/**`, `docs/decisions/ADR-000{1,2,3}*`, community files | raw pytest/ruff/pyright output, two identical smoke runs, validated manifest |
| `BackendCapability` (D-lite) | measurement taxonomy operations, backend compatibility matrix, benchmark protocol, memory-accounting rules with worked examples | `docs/protocols/**`, `docs/research/backend-capability.md` | live probes, cited doc URLs, CUDA-only determination |
| `LicensesEval` (I-lite) | model/dataset license + revision inventory, evaluation protocol + bounded task suite | `docs/research/model-dataset-licenses.md`, `docs/protocols/eval-protocol.md` | verified licenses/revisions, task configs, few-shot settings | delivered |
| `ReproPlan` (C) | upstream inspection, method cards, CPU-executable bounded reproduction design with predeclared tolerance | `docs/research/{reproduction-plan,upstream-notes}.md`, `references/method-cards/{lr-qat,loftq,lq-lora,torchao-qat}.md` | pinned SHAs, licence text, cost model, infeasibility statement |
| `AdversarialReview` (L) | independent attack on the M1 foundation before freeze: novelty verdicts vs real abstracts, proxy soundness, statistical plan, testability, equal-memory enforcement | `docs/results/verification/m1-novelty-review.md` | ≥5 citations independently checked, blocking-issue list |

## 5. Gates

| Gate | Requirement | Status |
|---|---|---|
| M0 | clean checkout passes lint, type checks, unit tests, tiny experiment | in progress |
| M1 | literature matrix + novelty audit + frozen preregistration + pinned upstreams; **gate** = independent review confirms the candidate extension is differentiated (or revises the question) | not started |
| M2 | math fixtures pass; allocator matches exhaustive search on tiny cases | not started |

## 6. Blockers and limits

- **No GPU.** Tiers 2–5 blocked until an external GPU host is documented (driver/CUDA version,
  supported quantization formats). No cloud GPU budget has been agreed yet.
- **Measurement classes 4–5 unavailable.** Any kernel-backed latency or service-level claim is
  deferred; nothing will be estimated to fill the gap.
- Upstream LR-QAT / LoftQ scripts rely on CUDA-only stacks; a CPU-executable bounded reproduction
  path must be designed explicitly (Milestone 3) or reported as infeasible with a reason.

## 7. Next actions

1. Verify wave-1 evidence; run the M0 gate on a clean checkout.
2. Fan out Milestone 2 slices (quantization core, factorization/spectral, proxy, allocator) once
   `Scaffold` freezes module interfaces.
3. Start the reproduction stream (LR-QAT/LoftQ) reading the pinned lockfile — no new-method claims
   before at least one reproduction succeeds.

## 8. Recorded deviation: single worktree during bootstrap

The specification calls for one git worktree and branch per independently editable stream. During
waves 1–2 all streams instead share **one working tree on `infra/bootstrap`** with strict path
ownership (`docs/coordination/ownership.md`), because subagents on a single Windows workstation share
one `uv` environment and one checkout; separate worktrees would multiply a 200 MB+ torch install and
make the CPU smoke gate non-comparable.

Consequences: (a) two agents must never touch the same file — enforced by the ownership map; (b)
per-stream branches are still used for *integration* of accepted changes (e.g. `research/*`, `feat/*`),
so the public history keeps one branch per reviewed change; (c) the M0 gate is verified on a clean
clone of the pushed branch, not on the working tree. To be revisited if a second machine or GPU host
becomes available.

## 9. Discipline reminders

- One writer per file; check `docs/coordination/ownership.md` before editing.
- Every result carries a measurement class; fake quantization is never low-bit storage.
- Equal-memory comparison required for every compression comparison.
- Amendments to the preregistration are append-only and timestamped.
