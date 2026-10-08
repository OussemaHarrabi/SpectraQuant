"""Colab Enterprise adapter — programmatic paid execution behind an explicit guard (design §1).

Colab Enterprise bills a GCP project, so it is the only platform this repository may not start
without the user's prior authorization. Two guards run **before** any SDK call, so a misconfigured
submission fails as a permission/config error rather than as a mysterious SDK exception:

1. an authorized GCP project must be configured (argument, ``GOOGLE_CLOUD_PROJECT`` or
   ``GCLOUD_PROJECT``) — this is the "authorized GCP project" of the design note;
2. :func:`spectraquant.cloud.budget.assert_submission_allowed` must pass, i.e.
   ``SPECTRAQUANT_ALLOW_PAID=1`` **and** a spend ceiling within the platform's documented maximum.

The SDK is optional and imported lazily: without ``google-cloud-aiplatform`` the adapter raises
``NotImplementedError`` naming what to install, after the guards have already run.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from spectraquant.cloud import budget
from spectraquant.cloud.adapters.base import FetchReport, RunStatus
from spectraquant.cloud.notebook import notebook_digest
from spectraquant.cloud.registry import Registry, default_registry
from spectraquant.cloud.spec import RunSpec

__all__ = [
    "GCP_PROJECT_ENV_VARS",
    "REQUIRED_SDK_MODULES",
    "ColabEnterpriseAdapter",
    "ColabEnterpriseConfigError",
    "MissingGcpProject",
]

logger = logging.getLogger("spectraquant.cloud.adapters.colab_enterprise")

#: Environment variables that may name the authorized GCP project (first match wins).
GCP_PROJECT_ENV_VARS: tuple[str, ...] = ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT")

#: Distribution the adapter needs before it can submit anything.
REQUIRED_SDK_MODULES: tuple[str, ...] = ("google.cloud.aiplatform",)

#: Deployment configuration the adapter needs (the image and staging bucket of the run container).
IMAGE_ENV = "SPECTRAQUANT_CLOUD_IMAGE"
BUCKET_ENV = "SPECTRAQUANT_GCS_BUCKET"
LOCATION_ENV = "SPECTRAQUANT_GCP_LOCATION"
DEFAULT_LOCATION = "us-central1"


class MissingGcpProject(ValueError):
    """Raised when no authorized GCP project is configured."""


class ColabEnterpriseConfigError(ValueError):
    """Raised when the deployment configuration for a Colab Enterprise run is incomplete."""


class ColabEnterpriseAdapter:
    """Submit a notebook as a Colab Enterprise / Vertex AI custom job, guarded by authorization.

    Args:
        project: authorized GCP project; falls back to the environment.
        location: GCP region.
        registry: registry that persists the remote job name; defaults to the repository registry.
        env: environment mapping (defaults to ``os.environ``).
        sdk_loader: callable returning the SDK namespace; injectable for tests (the real loader
            imports ``google.cloud`` lazily).
    """

    name = "colab_enterprise"

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
        registry: Registry | None = None,
        env: Mapping[str, str] | None = None,
        sdk_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._project = project
        self._location = location
        self._registry = registry if registry is not None else default_registry()
        self._env = dict(os.environ if env is None else env)
        self._injected_loader = sdk_loader is not None
        self._sdk_loader = sdk_loader if sdk_loader is not None else load_sdk

    # -- guard --------------------------------------------------------------------------
    def authorized_project(self) -> str | None:
        """Return the configured GCP project, or ``None`` when none is authorized."""
        if self._project:
            return self._project
        for name in GCP_PROJECT_ENV_VARS:
            value = str(self._env.get(name) or "").strip()
            if value:
                return value
        return None

    def authorize(self, spec: RunSpec) -> dict[str, str]:
        """Run every guard and return the resolved deployment configuration.

        Args:
            spec: the spec about to be submitted.

        Returns:
            ``{"project", "location", "image", "bucket"}``.

        Raises:
            MissingGcpProject: no authorized project is configured.
            budget.BudgetError: the paid submission is not authorized / over ceiling.
            ColabEnterpriseConfigError: the run image or staging bucket is not configured.
        """
        project = self.authorized_project()
        if project is None:
            raise MissingGcpProject(
                "Colab Enterprise requires an authorized GCP project: pass project=... or set "
                f"{' / '.join(GCP_PROJECT_ENV_VARS)} (AGENTS.md §2b — no paid resource without "
                "prior authorization)"
            )
        budget.assert_submission_allowed(spec, self._env)
        image = str(self._env.get(IMAGE_ENV) or "").strip()
        bucket = str(self._env.get(BUCKET_ENV) or "").strip()
        if not image or not bucket:
            missing = [
                name for name, value in ((IMAGE_ENV, image), (BUCKET_ENV, bucket)) if not value
            ]
            raise ColabEnterpriseConfigError(
                "Colab Enterprise submission needs the run container image and a GCS staging "
                f"bucket: set {', '.join(missing)} (see scripts/cloud/README.md)"
            )
        location = (
            self._location or str(self._env.get(LOCATION_ENV) or "").strip() or DEFAULT_LOCATION
        )
        return {"project": project, "location": location, "image": image, "bucket": bucket}

    # -- SDK ----------------------------------------------------------------------------
    def sdk_available(self) -> bool:
        """True when the SDK can be loaded (an injected loader is authoritative)."""
        if self._injected_loader:
            return True
        for module in REQUIRED_SDK_MODULES:
            try:
                if importlib.util.find_spec(module) is None:
                    return False
            except (ImportError, ValueError):  # a missing parent package
                return False
        return True

    def require_sdk(self) -> Any:
        """Return the SDK namespace.

        Raises:
            NotImplementedError: the SDK is not installed; the message names what to install.
        """
        if not self.sdk_available():
            raise NotImplementedError(
                "the Colab Enterprise SDK is not installed: install the optional dependency with "
                "`uv pip install 'google-cloud-aiplatform>=1.60'` (extra 'cloud-enterprise'). The "
                "authorization guards already passed, so only the SDK is missing."
            )
        return self._sdk_loader()

    # -- protocol -----------------------------------------------------------------------
    def submission_request(
        self, spec: RunSpec, notebook_path: str, *, authorized: Mapping[str, str]
    ) -> dict[str, Any]:
        """Return the Vertex AI custom-job payload that executes the notebook.

        The container runs the generated notebook with ``nbconvert --execute`` against the copy
        uploaded to ``gs://<bucket>/spectraquant/notebooks/<run_id>.ipynb``; the executed notebook
        and the artifact bundle land in ``gs://<bucket>/spectraquant/runs/<run_id>/``.
        """
        bucket = authorized["bucket"].removeprefix("gs://").rstrip("/")
        notebook_uri = f"gs://{bucket}/spectraquant/notebooks/{spec.run_id}.ipynb"
        output_uri = f"gs://{bucket}/spectraquant/runs/{spec.run_id}"
        machine_spec: dict[str, Any] = {
            "machine_type": "a2-highgpu-1g" if spec.gpu_required else "n1-standard-8"
        }
        if spec.gpu_required:
            machine_spec["accelerator_type"] = "NVIDIA_TESLA_A100"
            machine_spec["accelerator_count"] = 1
        return {
            "display_name": f"spectraquant-{spec.run_id}"[:128],
            "project": authorized["project"],
            "location": authorized["location"],
            "staging_bucket": f"gs://{bucket}",
            "worker_pool_specs": [
                {
                    "machine_spec": machine_spec,
                    "replica_count": 1,
                    "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 200},
                    "container_spec": {
                        "image_uri": authorized["image"],
                        "command": ["bash", "-lc"],
                        "args": [
                            "python -m pip install -q 'google-cloud-storage>=2.18' && "
                            f"gsutil cp {notebook_uri} /tmp/run.ipynb && "
                            "jupyter nbconvert --to notebook --execute --allow-errors "
                            "--ExecutePreprocessor.timeout=-1 "
                            f"--output-dir /tmp/executed /tmp/run.ipynb && "
                            f"gsutil -m cp -r /tmp/executed/* {output_uri}/notebook/ && "
                            f"gsutil -m cp -r /tmp/spectraquant-export/* {output_uri}/bundle/ || true"
                        ],
                    },
                }
            ],
            "notebook_uri": notebook_uri,
            "output_uri": output_uri,
            "notebook_path": str(notebook_path),
        }

    def upload_notebook(self, sdk: Any, *, bucket: str, notebook_path: str, run_id: str) -> str:
        """Upload the notebook to the staging bucket and return its ``gs://`` URI."""
        name = bucket.removeprefix("gs://").rstrip("/")
        uri = f"gs://{name}/spectraquant/notebooks/{run_id}.ipynb"
        client = sdk.storage.Client(project=self.authorized_project())
        blob = client.bucket(name).blob(f"spectraquant/notebooks/{run_id}.ipynb")
        blob.upload_from_filename(str(notebook_path))
        return uri

    def submit(self, spec: RunSpec, notebook_path: str) -> str:
        """Authorize, upload the notebook and submit a custom job; return its resource name.

        Raises:
            MissingGcpProject: no authorized project.
            budget.BudgetError: paid submission not authorized / over ceiling.
            ColabEnterpriseConfigError: image or bucket not configured.
            NotImplementedError: the SDK is not installed.
            FileNotFoundError: the generated notebook is missing.
        """
        authorized = self.authorize(spec)
        source = Path(notebook_path)
        if not source.is_file():
            raise FileNotFoundError(f"generated notebook not found: {source}")
        sdk = self.require_sdk()
        self.upload_notebook(
            sdk, bucket=authorized["bucket"], notebook_path=str(source), run_id=spec.run_id
        )
        request = self.submission_request(spec, str(source), authorized=authorized)
        aiplatform = sdk.aiplatform
        aiplatform.init(project=authorized["project"], location=authorized["location"])
        job = aiplatform.CustomJob(
            display_name=request["display_name"],
            worker_pool_specs=request["worker_pool_specs"],
            staging_bucket=request["staging_bucket"],
        )
        job.run(sync=False)
        remote_id = str(getattr(job, "resource_name", None) or request["display_name"])
        self._registry.record_submitted(
            spec,
            remote_id=remote_id,
            submitted_by="agent",
            notebook_digest=notebook_digest(spec),
            detail=f"vertex custom job submitted to project={authorized['project']}",
        )
        return remote_id

    def status(self, run_id: str) -> RunStatus:
        """Return the state of the submitted custom job, when the SDK is available."""
        remote_id = self._registry.remote_id(run_id)
        if remote_id is None:
            return RunStatus(
                run_id=run_id, state="unknown", detail="no remote job persisted for this run"
            )
        if not self.sdk_available():
            return RunStatus(
                run_id=run_id,
                state="unknown",
                remote_id=remote_id,
                detail="the Colab Enterprise SDK is not installed in this environment",
            )
        sdk = self._sdk_loader()
        job = sdk.aiplatform.CustomJob.get(resource_name=remote_id)
        state = str(getattr(job, "state", "unknown")).lower()
        mapped = {
            "job_state_succeeded": "finished",
            "job_state_failed": "failed",
            "job_state_running": "running",
            "job_state_queued": "queued",
            "job_state_pending": "queued",
            "job_state_cancelled": "failed",
            "job_state_cancelling": "running",
        }.get(state, "unknown")
        return RunStatus(run_id=run_id, state=mapped, remote_id=remote_id, detail=state)

    def fetch(self, run_id: str, dest_dir: str) -> FetchReport:
        """Refuse: artifacts are read from the staging bucket by the collection step.

        Downloading is a ``gsutil``/storage-client operation on the GCS staging bucket, which
        ``collect`` performs against a local directory; this adapter does not duplicate it.
        """
        raise NotImplementedError(
            "Colab Enterprise artifacts are written to the GCS staging bucket "
            f"(${BUCKET_ENV}/spectraquant/runs/{run_id}/); download that prefix with "
            "`gsutil -m cp -r` (or the storage client) and run `spectraquant cloud collect "
            f"--run-id {run_id} --source <downloaded dir>`"
        )

    def can_resume(self, run_id: str) -> bool:
        """True when a job name is persisted and the run has not reached a terminal state."""
        if self._registry.remote_id(run_id) is None:
            return False
        return self._registry.state(run_id) in {"submitted", "running"}


def load_sdk() -> Any:
    """Import the Google Cloud SDKs lazily and return a namespace with ``aiplatform`` and ``storage``.

    The import is dynamic so that a checkout without the optional dependencies imports this module
    cleanly and only fails at the moment a paid submission is actually attempted.

    Raises:
        NotImplementedError: either SDK module is missing.
    """
    modules: dict[str, Any] = {}
    for name in ("aiplatform", "storage"):
        try:
            modules[name] = importlib.import_module(f"google.cloud.{name}")
        except ImportError as exc:  # pragma: no cover - guarded by require_sdk()
            raise NotImplementedError(
                "install the optional Colab Enterprise dependencies with "
                "`uv pip install 'google-cloud-aiplatform>=1.60' 'google-cloud-storage>=2.18'`"
            ) from exc
    return SimpleNamespace(**modules)


def default_colab_enterprise_adapter() -> ColabEnterpriseAdapter:
    """Return a :class:`ColabEnterpriseAdapter` bound to the default registry and environment."""
    return ColabEnterpriseAdapter()
