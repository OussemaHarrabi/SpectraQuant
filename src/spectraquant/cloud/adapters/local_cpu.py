"""Local-CPU adapter — the workstation, for Tier-0 fixtures and class 4-CPU work only.

``AGENTS.md`` §2.3 restricts local execution to repository management, CPU tests, tiny synthetic
fixtures, static analysis, configuration validation, notebook generation, analysis and reporting:
**no research training, no large-scale inference, no GPU evaluation**. This adapter is the substrate
for exactly that scope, and it refuses anything that needs CUDA rather than silently degrading to a
different numeric result (``AGENTS.md`` §2.2).

Execution is synchronous and local: the declared command runs in a subprocess and its manifest lands
in ``artifacts/runs/<run_id>/``, where :func:`spectraquant.cloud.collect.collect` can validate it like
any other bundle.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from spectraquant.cloud import budget
from spectraquant.cloud.adapters.base import (
    CommandRunner,
    FetchReport,
    RunStatus,
    run_command,
)
from spectraquant.cloud.notebook import notebook_digest
from spectraquant.cloud.registry import Registry, default_registry, runs_dir
from spectraquant.cloud.spec import RunSpec

__all__ = ["LOCAL_RUN_COMMAND", "LocalCpuAdapter", "LocalExecutionRefused"]

logger = logging.getLogger("spectraquant.cloud.adapters.local_cpu")

#: The frozen CLI entry point every execution substrate uses (design note §3, cell 5).
LOCAL_RUN_COMMAND: tuple[str, ...] = ("-m", "spectraquant", "run")


class LocalExecutionRefused(ValueError):
    """Raised when a spec asks the workstation to do something it must not do."""


class LocalCpuAdapter:
    """Execute a spec locally (Tier-0 scope) and expose it through the adapter protocol.

    Args:
        registry: registry that records the local run; defaults to the repository registry.
        env: environment mapping (defaults to ``os.environ``).
        runner: command runner; injectable so tests exercise the flow without running the CLI.
        runs_root: root of the per-run output directories; defaults to ``artifacts/runs``.
    """

    name = "local_cpu"

    def __init__(
        self,
        *,
        registry: Registry | None = None,
        env: Mapping[str, str] | None = None,
        runner: CommandRunner | None = None,
        runs_root: str | Path | None = None,
    ) -> None:
        self._registry = registry if registry is not None else default_registry()
        self._env = dict(os.environ if env is None else env)
        self._runner: CommandRunner = runner if runner is not None else run_command
        self._runs_root = Path(runs_root) if runs_root is not None else None

    # -- helpers ------------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        """Directory holding this run's outputs and manifest."""
        root = self._runs_root if self._runs_root is not None else runs_dir()
        return root / run_id

    def command(self, spec: RunSpec) -> list[str]:
        """Return the argv executed locally for ``spec``."""
        return [
            sys.executable,
            *LOCAL_RUN_COMMAND,
            "--config",
            spec.experiment_config,
            *spec.overrides,
        ]

    def check_allowed(self, spec: RunSpec) -> None:
        """Refuse specs the workstation must not execute.

        Raises:
            LocalExecutionRefused: the spec targets another platform, needs a GPU, or exceeds the
                Tier-0 scope.
        """
        if spec.platform != "local_cpu":
            raise LocalExecutionRefused(
                f"spec.platform={spec.platform!r} cannot run on the local_cpu adapter"
            )
        if spec.gpu_required:
            raise LocalExecutionRefused(
                "local_cpu refuses gpu_required=True: this workstation has no CUDA device and every "
                "GPU-dependent experiment runs on the cloud substrate (AGENTS.md §2, §2b)"
            )

    # -- protocol -----------------------------------------------------------------------
    def submit(self, spec: RunSpec, notebook_path: str) -> str:
        """Record the submission and execute the spec locally (synchronously).

        Args:
            spec: the frozen spec.
            notebook_path: the generated notebook (kept as the record of what was declared).

        Returns:
            The run id (local execution has no platform-side id).

        Raises:
            LocalExecutionRefused: the spec is out of the local scope.
        """
        self.check_allowed(spec)
        budget.assert_submission_allowed(spec, self._env)
        target = self.run_dir(spec.run_id)
        target.mkdir(parents=True, exist_ok=True)
        self._registry.record_submitted(
            spec,
            remote_id=spec.run_id,
            submitted_by="agent",
            notebook_digest=notebook_digest(spec),
            detail=f"local Tier-0 execution in {target} (notebook: {notebook_path})",
        )
        self._registry.record("running", spec.run_id, platform=self.name, remote_id=spec.run_id)

        argv = self.command(spec)
        result = self._runner(argv, self._env, target)
        (target / "run.log").write_text(result.combined)
        if result.ok:
            self._registry.record(
                "finished", spec.run_id, platform=self.name, remote_id=spec.run_id
            )
        else:
            self._registry.record_failure(
                spec.run_id,
                f"local run exited {result.returncode}: {result.combined[-400:]}",
                platform=self.name,
                remote_id=spec.run_id,
            )
        return spec.run_id

    def status(self, run_id: str) -> RunStatus:
        """Report the local state: ``finished`` when a manifest exists, else the recorded state."""
        target = self.run_dir(run_id)
        manifest = target / "run_manifest.json"
        if manifest.is_file():
            return RunStatus(
                run_id=run_id,
                state="finished",
                remote_id=run_id,
                detail=str(manifest),
            )
        recorded = self._registry.state(run_id)
        mapped = {
            "submitted": "queued",
            "running": "running",
            "finished": "finished",
            "failed": "failed",
        }.get(recorded or "", "unknown")
        return RunStatus(
            run_id=run_id,
            state=mapped,
            remote_id=run_id,
            detail=f"registry state: {recorded}" if recorded else "no local run recorded",
        )

    def fetch(self, run_id: str, dest_dir: str) -> FetchReport:
        """Copy the local run directory into ``dest_dir`` (a no-op download)."""
        source = self.run_dir(run_id)
        if not source.is_dir():
            return FetchReport(
                run_id=run_id,
                dest_dir=str(dest_dir),
                ok=False,
                detail=f"no local run directory at {source}",
            )
        target = Path(dest_dir)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
        files = sorted(
            path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
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

    def can_resume(self, run_id: str) -> bool:
        """True when a local run was recorded (nothing is in flight, so re-running is safe)."""
        return self._registry.state(run_id) is not None


def default_local_adapter(*, runner: CommandRunner | None = None) -> LocalCpuAdapter:
    """Return a :class:`LocalCpuAdapter` bound to the default registry and environment."""
    return LocalCpuAdapter(runner=runner)
