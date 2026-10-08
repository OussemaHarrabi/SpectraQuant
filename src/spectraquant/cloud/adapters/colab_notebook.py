"""Consumer-Colab adapter — emits an execution-ready notebook for a human (design note §1).

Consumer Colab exposes **no official submission API**. This adapter therefore cannot start a run,
and it says so: it records ``submitted_by: "human"`` with ``remote_id: null``, and its
:meth:`ColabNotebookAdapter.status` reports ``unknown`` — it never claims to have started anything.
Artifacts come back through the developer's export (Drive/HF) and are validated locally by
:func:`spectraquant.cloud.collect.collect`.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from spectraquant.cloud import budget
from spectraquant.cloud.adapters.base import FetchReport, RunStatus
from spectraquant.cloud.notebook import notebook_digest, write_notebook
from spectraquant.cloud.registry import Registry, default_registry, runs_dir
from spectraquant.cloud.spec import RunSpec

__all__ = ["HANDOFF_DETAIL", "ColabNotebookAdapter"]

logger = logging.getLogger("spectraquant.cloud.adapters.colab_notebook")

#: Recorded verbatim in the registry so the research record never overstates what happened.
HANDOFF_DETAIL = (
    "notebook emitted for human execution; consumer Colab has no submission API, "
    "so no run was started by this adapter"
)


class ColabNotebookAdapter:
    """Prepare a Colab notebook handoff and record it as a human submission.

    Args:
        registry: registry to record the handoff in; defaults to the repository registry.
        env: environment mapping (defaults to ``os.environ``).
        out_dir: directory the handoff notebook is written to; defaults to
            ``artifacts/runs/<run_id>/handoff``.
    """

    name = "colab"

    def __init__(
        self,
        *,
        registry: Registry | None = None,
        env: Mapping[str, str] | None = None,
        out_dir: str | Path | None = None,
    ) -> None:
        self._registry = registry if registry is not None else default_registry()
        self._env = dict(os.environ if env is None else env)
        self._out_dir = Path(out_dir) if out_dir is not None else None

    def handoff_dir(self, run_id: str) -> Path:
        """Directory that holds the emitted handoff notebook."""
        root = self._out_dir if self._out_dir is not None else runs_dir() / run_id
        return root / "handoff"

    def submit(self, spec: RunSpec, notebook_path: str) -> str:
        """Emit the notebook and record the human handoff; no remote job is created.

        Args:
            spec: the frozen spec.
            notebook_path: where the generated notebook is (or should be) written. When the path is
                empty or missing, the notebook is regenerated there.

        Returns:
            The *handoff* id — the run id. This is deliberately not a platform job id: no job
            exists until a human opens and runs the notebook.
        """
        budget.assert_submission_allowed(spec, self._env)
        default_target = self.handoff_dir(spec.run_id) / f"{spec.run_id}.ipynb"
        target = Path(notebook_path) if str(notebook_path).strip() else default_target
        digest = notebook_digest(spec) if target.is_file() else write_notebook(spec, target)

        if target.parent != self.handoff_dir(spec.run_id):
            handoff = self.handoff_dir(spec.run_id) / target.name
            handoff.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, handoff)

        self._registry.record_submitted(
            spec,
            remote_id=None,
            submitted_by="human",
            notebook_digest=digest,
            detail=HANDOFF_DETAIL,
        )
        logger.info(
            "colab: notebook emitted for %s at %s (no run started; submitted_by=human)",
            spec.run_id,
            target,
        )
        return spec.run_id

    def status(self, run_id: str) -> RunStatus:
        """Report that the platform cannot be queried: no submission API exists.

        The state is ``unknown`` — never ``running`` — because this adapter has no way to know
        whether the developer has opened the notebook.
        """
        return RunStatus(
            run_id=run_id,
            state="unknown",
            remote_id=None,
            detail=(
                "consumer Colab has no submission API: the run is executed by a human, so its state "
                "cannot be queried. Validate the exported bundle with `spectraquant cloud collect`."
            ),
        )

    def fetch(self, run_id: str, dest_dir: str) -> FetchReport:
        """Refuse: consumer Colab has no artifact API.

        The developer exports the bundle (Drive/HF) and points ``collect`` at the downloaded
        directory.
        """
        raise NotImplementedError(
            "consumer Colab has no artifact-download API: export the bundle from the notebook "
            "(the export cell writes it to the run's export directory, mirroring to "
            "/content/drive/MyDrive/spectraquant-runs when Drive is mounted), download it, and run "
            "`spectraquant cloud collect --run-id "
            f"{run_id} --source <downloaded dir>`"
        )

    def can_resume(self, run_id: str) -> bool:
        """True when a handoff was emitted and no terminal state has been recorded."""
        return self._registry.state(run_id) == "submitted"


def default_colab_adapter() -> ColabNotebookAdapter:
    """Return a :class:`ColabNotebookAdapter` bound to the default registry and environment."""
    return ColabNotebookAdapter()
