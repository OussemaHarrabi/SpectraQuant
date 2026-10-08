"""Kaggle Notebooks adapter — the fully programmatic backend (``design-cloud-adapter.md`` §1).

Kaggle is the preferred *unattended* backend: the official CLI does upload → start → poll →
download. This adapter shells out to that CLI and nothing else: it never imports the Kaggle API
client, so the CLI's own credential handling (environment variables or ``~/.kaggle``) stays the
single source of truth.

Two rules from the design note are enforced here:

* ``submit`` persists the remote run id to the registry **before** returning, so an interruption
  between submit and poll cannot orphan a running job;
* a run whose remote id is already persisted is *resumed*, never re-submitted.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spectraquant.cloud import budget
from spectraquant.cloud.adapters.base import (
    CommandResult,
    CommandRunner,
    FetchReport,
    PollPolicy,
    RunStatus,
    poll_until_terminal,
    run_command,
)
from spectraquant.cloud.notebook import notebook_digest
from spectraquant.cloud.registry import Registry, default_registry, runs_dir
from spectraquant.cloud.secrets import redact, require_credentials
from spectraquant.cloud.spec import RunSpec

__all__ = ["KAGGLE_CREDENTIALS", "KaggleAdapter", "KaggleSubmissionError", "slugify_run_id"]

logger = logging.getLogger("spectraquant.cloud.adapters.kaggle")

#: Credentials the Kaggle CLI reads from the environment.
KAGGLE_CREDENTIALS: tuple[str, ...] = ("KAGGLE_USERNAME", "KAGGLE_KEY")

#: Kaggle slug rules: lowercase alphanumerics and hyphens, at most 50 characters.
_SLUG_MAX = 50

_STATUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'has status "([A-Za-z]+)"'),
    re.compile(r"KernelWorkerStatus\.([A-Za-z]+)"),
)

_STATUS_MAP: dict[str, str] = {
    "complete": "finished",
    "completed": "finished",
    "running": "running",
    "queued": "queued",
    "starting": "queued",
    "error": "failed",
    "failed": "failed",
    "cancelled": "failed",
    "cancelacknowledged": "failed",
    "cancelauthorized": "failed",
    "unknown": "unknown",
}


class KaggleSubmissionError(RuntimeError):
    """Raised when the Kaggle CLI refuses a push (the message is credential-redacted)."""


def slugify_run_id(run_id: str) -> str:
    """Return the Kaggle slug for a SpectraQuant run id (lowercase, hyphens, ≤50 characters)."""
    slug = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug or "spectraquant-run")[:_SLUG_MAX]


class KaggleAdapter:
    """Submit, poll and fetch Kaggle notebook runs through the official CLI.

    Args:
        registry: registry used to persist the remote run id; defaults to the repository registry.
        env: environment mapping (defaults to ``os.environ``). Credentials are read from here and
            are never logged.
        runner: command runner; injectable so tests exercise the whole flow without a network.
        work_root: where the kernel upload directory is staged; defaults to
            ``artifacts/runs/<run_id>/kaggle``.
        cli: path of the ``kaggle`` executable; discovered with ``shutil.which`` when omitted.
    """

    name = "kaggle"

    def __init__(
        self,
        *,
        registry: Registry | None = None,
        env: Mapping[str, str] | None = None,
        runner: CommandRunner | None = None,
        work_root: str | Path | None = None,
        cli: str | None = None,
    ) -> None:
        self._registry = registry if registry is not None else default_registry()
        self._env = dict(os.environ if env is None else env)
        self._runner: CommandRunner = runner if runner is not None else run_command
        self._work_root = Path(work_root) if work_root is not None else None
        self._cli = cli

    # -- helpers ------------------------------------------------------------------------
    @property
    def cli(self) -> str:
        """Path of the ``kaggle`` executable (falls back to the bare name for PATH lookup)."""
        return self._cli or shutil.which("kaggle") or "kaggle"

    def credentials(self) -> dict[str, str]:
        """Return the Kaggle credentials from the environment.

        Raises:
            MissingCredentials: ``KAGGLE_USERNAME``/``KAGGLE_KEY`` are not set.
        """
        return require_credentials(KAGGLE_CREDENTIALS, self._env)

    def remote_id_for(self, run_id: str) -> str:
        """Return the Kaggle kernel id (``<username>/<slug>``) for a run id."""
        username = self.credentials()["KAGGLE_USERNAME"]
        return f"{username}/{slugify_run_id(run_id)}"

    def kernel_dir(self, run_id: str) -> Path:
        """Directory staged for ``kaggle kernels push``."""
        root = self._work_root if self._work_root is not None else runs_dir() / run_id
        return root / "kaggle"

    def _invoke(self, argv: list[str], *, cwd: Path | None = None) -> CommandResult:
        result = self._runner(argv, self._env, cwd)
        logger.info(
            "kaggle cli argv=%s rc=%d",
            [redact(item) for item in argv],
            result.returncode,
        )
        return result

    # -- protocol -----------------------------------------------------------------------
    def can_resume(self, run_id: str) -> bool:
        """True when a remote id is persisted and the run has not reached a terminal state."""
        if self._registry.remote_id(run_id) is None:
            return False
        return self._registry.state(run_id) in {"submitted", "running"}

    def submit(self, spec: RunSpec, notebook_path: str) -> str:
        """Push the generated notebook and start the run.

        Args:
            spec: the frozen spec.
            notebook_path: path of the generated notebook (regenerated bytes are the same for the
                same spec).

        Returns:
            The Kaggle kernel id (``<username>/<slug>``).

        Raises:
            MissingCredentials: credentials are absent.
            budget.BudgetError: the submission is not authorized.
            KaggleSubmissionError: the CLI refused the push.
        """
        budget.assert_submission_allowed(spec, self._env)

        persisted = self._registry.remote_id(spec.run_id)
        if persisted is not None and self.can_resume(spec.run_id):
            logger.info(
                "kaggle: resuming run_id=%s remote_id=%s (not re-submitting)",
                spec.run_id,
                persisted,
            )
            return persisted

        remote_id = self.remote_id_for(spec.run_id)
        staging = self.kernel_dir(spec.run_id)
        staging.mkdir(parents=True, exist_ok=True)
        notebook_source = Path(notebook_path)
        if not notebook_source.is_file():
            raise FileNotFoundError(f"generated notebook not found: {notebook_source}")
        shutil.copy2(notebook_source, staging / "notebook.ipynb")
        metadata = self.kernel_metadata(spec, remote_id)
        (staging / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        argv = [
            self.cli,
            "kernels",
            "push",
            "-p",
            str(staging),
            "-t",
            str(int(spec.timeout_minutes) * 60),
        ]
        result = self._invoke(argv, cwd=staging)
        if not result.ok:
            reason = (
                f"kaggle kernels push failed (exit {result.returncode}): {result.combined[-600:]}"
            )
            self._registry.record_failure(
                spec.run_id, reason, platform=self.name, remote_id=remote_id
            )
            raise KaggleSubmissionError(reason)

        # Persist the remote id BEFORE returning: an interruption must not orphan the job.
        self._registry.record_submitted(
            spec,
            remote_id=remote_id,
            submitted_by="agent",
            notebook_digest=notebook_digest(spec),
            detail=f"kaggle kernels push -p {staging}",
        )
        return remote_id

    def kernel_metadata(self, spec: RunSpec, remote_id: str) -> dict[str, Any]:
        """Return the ``kernel-metadata.json`` document for a run."""
        return {
            "id": remote_id,
            "title": f"SpectraQuant {spec.run_id}"[:50],
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": bool(spec.gpu_required),
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }

    def status(self, run_id: str) -> RunStatus:
        """Query ``kaggle kernels status`` and map its output onto :data:`REMOTE_STATES`."""
        remote_id = self._registry.remote_id(run_id)
        if remote_id is None:
            return RunStatus(
                run_id=run_id,
                state="unknown",
                detail="no remote run id persisted for this run (was it submitted?)",
            )
        result = self._invoke([self.cli, "kernels", "status", remote_id])
        if not result.ok:
            return RunStatus(
                run_id=run_id,
                state="unknown",
                remote_id=remote_id,
                detail=f"kaggle kernels status failed (exit {result.returncode})",
                raw={"output": result.combined[-600:]},
            )
        state = parse_status_output(result.combined)
        return RunStatus(
            run_id=run_id,
            state=state,
            remote_id=remote_id,
            detail=None if state != "unknown" else result.combined[-300:],
            raw={"output": result.combined[-600:]},
        )

    def poll(
        self,
        run_id: str,
        *,
        policy: PollPolicy | None = None,
        sleep: Any = time.sleep,
    ) -> RunStatus:
        """Poll with bounded exponential backoff until terminal or out of attempts."""
        return poll_until_terminal(self, run_id, policy=policy, sleep=sleep)

    def fetch(self, run_id: str, dest_dir: str) -> FetchReport:
        """Download the run's output files with ``kaggle kernels output``."""
        remote_id = self._registry.remote_id(run_id)
        if remote_id is None:
            return FetchReport(
                run_id=run_id,
                dest_dir=str(dest_dir),
                ok=False,
                detail="no remote run id persisted for this run",
            )
        target = Path(dest_dir)
        target.mkdir(parents=True, exist_ok=True)
        result = self._invoke([self.cli, "kernels", "output", remote_id, "-p", str(target)])
        files = sorted(
            path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
        )
        if not result.ok:
            return FetchReport(
                run_id=run_id,
                dest_dir=str(target),
                files=files,
                ok=False,
                detail=f"kaggle kernels output failed (exit {result.returncode}): "
                f"{result.combined[-600:]}",
            )
        notebooks = [name for name in files if name.endswith(".ipynb")]
        logs = [name for name in files if name.endswith(".log")]
        return FetchReport(
            run_id=run_id,
            dest_dir=str(target),
            files=files,
            executed_notebook=notebooks[0] if notebooks else None,
            log=logs[0] if logs else None,
        )


def parse_status_output(output: str) -> str:
    """Map ``kaggle kernels status`` output onto a state in :data:`REMOTE_STATES`.

    The CLI prints either ``"<id>" has status "complete"`` or an enum such as
    ``KernelWorkerStatus.RUNNING``; an unrecognised payload maps to ``"unknown"`` rather than
    guessing.
    """
    for pattern in _STATUS_PATTERNS:
        for match in pattern.finditer(output):
            mapped = _STATUS_MAP.get(match.group(1).lower())
            if mapped is not None:
                return mapped
    return "unknown"


def default_kaggle_adapter() -> KaggleAdapter:
    """Return a :class:`KaggleAdapter` bound to the default registry and environment."""
    return KaggleAdapter()
