"""Platform adapters: honest capability, resume semantics and the paid guard."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from _cloud_fixtures import make_spec

from spectraquant.cloud.adapters.base import (
    CommandResult,
    PollPolicy,
    RunStatus,
    poll_until_terminal,
)
from spectraquant.cloud.adapters.colab_enterprise import (
    BUCKET_ENV,
    IMAGE_ENV,
    ColabEnterpriseAdapter,
    ColabEnterpriseConfigError,
    MissingGcpProject,
)
from spectraquant.cloud.adapters.colab_notebook import HANDOFF_DETAIL, ColabNotebookAdapter
from spectraquant.cloud.adapters.kaggle import KaggleAdapter, KaggleSubmissionError, slugify_run_id
from spectraquant.cloud.adapters.local_cpu import LocalCpuAdapter, LocalExecutionRefused
from spectraquant.cloud.registry import REGISTRY_FILENAME, Registry
from spectraquant.cloud.secrets import MissingCredentials

KAGGLE_ENV = {"KAGGLE_USERNAME": "someone", "KAGGLE_KEY": "kaggle_key_value_long_enough"}


def _registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "runs" / REGISTRY_FILENAME)


class FakeRunner:
    """Records argv and returns scripted results (never touches the network)."""

    def __init__(self, results: dict[str, CommandResult] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._results = results or {}

    def __call__(self, argv, env, cwd):  # type: ignore[no-untyped-def]
        self.calls.append([str(item) for item in argv])
        joined = " ".join(str(item) for item in argv)
        for key, result in self._results.items():
            if key in joined:
                return result
        return CommandResult(argv=[str(item) for item in argv], returncode=0, stdout="ok")

    @property
    def flat(self) -> list[str]:
        return [" ".join(call) for call in self.calls]


# --------------------------------------------------------------------------------------
# Kaggle
# --------------------------------------------------------------------------------------
def test_slugify_run_id_follows_kaggle_rules() -> None:
    assert slugify_run_id("Smoke_Run.v2/2026") == "smoke-run-v2-2026"
    assert len(slugify_run_id("x" * 120)) == 50


def test_kaggle_submit_pushes_and_persists_the_remote_id(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runner = FakeRunner()
    adapter = KaggleAdapter(registry=registry, env=KAGGLE_ENV, runner=runner, work_root=tmp_path)
    spec = make_spec(platform="kaggle", gpu_required=True, timeout_minutes=60)
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    remote_id = adapter.submit(spec, str(notebook))

    assert remote_id == "someone/cloud-unit-run"
    assert registry.state(spec.run_id) == "submitted"
    assert registry.remote_id(spec.run_id) == remote_id
    assert registry.latest(spec.run_id)["submitted_by"] == "agent"
    assert "kernels push" in runner.flat[0]
    metadata = json.loads((adapter.kernel_dir(spec.run_id) / "kernel-metadata.json").read_text())
    assert metadata["id"] == remote_id
    assert metadata["enable_gpu"] is True
    assert metadata["kernel_type"] == "notebook"


def test_kaggle_submit_requires_credentials(tmp_path: Path) -> None:
    adapter = KaggleAdapter(registry=_registry(tmp_path), env={}, runner=FakeRunner())
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    with pytest.raises(MissingCredentials, match="KAGGLE_USERNAME"):
        adapter.submit(make_spec(platform="kaggle", gpu_required=True), str(notebook))


def test_kaggle_resume_reuses_the_persisted_remote_id_without_resubmitting(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runner = FakeRunner()
    adapter = KaggleAdapter(registry=registry, env=KAGGLE_ENV, runner=runner, work_root=tmp_path)
    spec = make_spec(platform="kaggle", gpu_required=True)
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    first = adapter.submit(spec, str(notebook))
    assert adapter.can_resume(spec.run_id) is True
    second = adapter.submit(spec, str(notebook))

    assert first == second
    assert len(runner.calls) == 1  # the second submit did not push again
    assert len(registry.transitions(spec.run_id)) == 1


def test_kaggle_does_not_resume_a_terminal_run(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runner = FakeRunner()
    adapter = KaggleAdapter(registry=registry, env=KAGGLE_ENV, runner=runner, work_root=tmp_path)
    spec = make_spec(platform="kaggle", gpu_required=True)
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    adapter.submit(spec, str(notebook))
    registry.record("running", spec.run_id)
    registry.record("finished", spec.run_id)

    assert adapter.can_resume(spec.run_id) is False


def test_kaggle_push_failure_is_recorded_and_raised(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runner = FakeRunner(
        {"kernels push": CommandResult(argv=["kaggle"], returncode=1, stderr="quota exceeded")}
    )
    adapter = KaggleAdapter(registry=registry, env=KAGGLE_ENV, runner=runner, work_root=tmp_path)
    spec = make_spec(platform="kaggle", gpu_required=True)
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    with pytest.raises(KaggleSubmissionError, match="quota exceeded"):
        adapter.submit(spec, str(notebook))

    assert registry.state(spec.run_id) == "failed"
    assert registry.remote_id(spec.run_id) == "someone/cloud-unit-run"


def test_kaggle_status_parses_the_cli_output(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1", remote_id="someone/run-1", platform="kaggle")
    runner = FakeRunner(
        {
            "kernels status": CommandResult(
                argv=["kaggle"], returncode=0, stdout='"someone/run-1" has status "complete"'
            )
        }
    )
    adapter = KaggleAdapter(registry=registry, env=KAGGLE_ENV, runner=runner)

    status = adapter.status("run-1")

    assert status.state == "finished"
    assert status.remote_id == "someone/run-1"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('"u/k" has status "running"', "running"),
        ('"u/k" has status "queued"', "queued"),
        ("KernelWorkerStatus.RUNNING", "running"),
        ("KernelWorkerStatus.ERROR", "failed"),
        ("something unexpected", "unknown"),
    ],
)
def test_kaggle_status_mapping(tmp_path: Path, output: str, expected: str) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1", remote_id="someone/run-1", platform="kaggle")
    runner = FakeRunner(
        {"kernels status": CommandResult(argv=["kaggle"], returncode=0, stdout=output)}
    )

    assert (
        KaggleAdapter(registry=registry, env=KAGGLE_ENV, runner=runner).status("run-1").state
        == expected
    )


def test_kaggle_status_without_a_remote_id_is_unknown(tmp_path: Path) -> None:
    adapter = KaggleAdapter(registry=_registry(tmp_path), env=KAGGLE_ENV, runner=FakeRunner())

    status = adapter.status("never-submitted")

    assert status.state == "unknown"
    assert "no remote run id" in (status.detail or "")


def test_kaggle_fetch_downloads_outputs(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1", remote_id="someone/run-1", platform="kaggle")
    dest = tmp_path / "dest"

    def runner(argv, env, cwd):  # type: ignore[no-untyped-def]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "run.log").write_text("done", encoding="utf-8")
        (dest / "run.ipynb").write_text("{}", encoding="utf-8")
        return CommandResult(argv=[str(item) for item in argv], returncode=0)

    adapter = KaggleAdapter(registry=registry, env=KAGGLE_ENV, runner=runner)

    report = adapter.fetch("run-1", str(dest))

    assert report.ok is True
    assert sorted(report.files) == ["run.ipynb", "run.log"]
    assert report.executed_notebook == "run.ipynb"


def test_kaggle_fetch_without_a_remote_id_fails(tmp_path: Path) -> None:
    adapter = KaggleAdapter(registry=_registry(tmp_path), env=KAGGLE_ENV, runner=FakeRunner())

    report = adapter.fetch("run-1", str(tmp_path / "dest"))

    assert report.ok is False
    assert "no remote run id" in (report.detail or "")


def test_kaggle_credentials_are_never_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    registry = _registry(tmp_path)
    adapter = KaggleAdapter(
        registry=registry, env=KAGGLE_ENV, runner=FakeRunner(), work_root=tmp_path
    )
    spec = make_spec(platform="kaggle", gpu_required=True)
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="spectraquant.cloud.adapters.kaggle"):
        adapter.submit(spec, str(notebook))

    assert KAGGLE_ENV["KAGGLE_KEY"] not in caplog.text


# --------------------------------------------------------------------------------------
# Consumer Colab
# --------------------------------------------------------------------------------------
def test_colab_submit_never_claims_to_have_started_a_run(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    adapter = ColabNotebookAdapter(registry=registry, env={}, out_dir=tmp_path / "handoff")
    spec = make_spec(platform="colab", gpu_required=True)
    notebook = tmp_path / "generated.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    submission_id = adapter.submit(spec, str(notebook))

    assert submission_id == spec.run_id
    line = registry.latest(spec.run_id)
    assert line is not None
    assert line["state"] == "submitted"
    assert line["remote_id"] is None
    assert line["submitted_by"] == "human"
    assert line["detail"] == HANDOFF_DETAIL
    assert adapter.status(spec.run_id).state == "unknown"
    assert adapter.can_resume(spec.run_id) is True


def test_colab_fetch_refuses_with_an_actionable_message(tmp_path: Path) -> None:
    adapter = ColabNotebookAdapter(registry=_registry(tmp_path), env={}, out_dir=tmp_path)

    with pytest.raises(NotImplementedError, match="no artifact-download API"):
        adapter.fetch("run-1", str(tmp_path / "dest"))


def test_colab_submit_generates_the_notebook_when_missing(tmp_path: Path) -> None:
    pytest.importorskip("nbformat")
    registry = _registry(tmp_path)
    adapter = ColabNotebookAdapter(registry=registry, env={}, out_dir=tmp_path / "handoff")
    spec = make_spec(platform="colab", gpu_required=True)

    adapter.submit(spec, "")

    generated = tmp_path / "handoff" / "handoff" / f"{spec.run_id}.ipynb"
    assert generated.is_file()
    assert registry.latest(spec.run_id)["notebook_digest"].startswith("sha256:")


# --------------------------------------------------------------------------------------
# Colab Enterprise (guards are real; the SDK is optional)
# --------------------------------------------------------------------------------------
def test_colab_enterprise_requires_a_project(tmp_path: Path) -> None:
    adapter = ColabEnterpriseAdapter(registry=_registry(tmp_path), env={})
    spec = make_spec(platform="colab_enterprise", gpu_required=True, max_cost_authorized_usd=5.0)

    with pytest.raises(MissingGcpProject, match="GOOGLE_CLOUD_PROJECT"):
        adapter.submit(spec, str(tmp_path / "run.ipynb"))


def test_colab_enterprise_requires_paid_authorization(tmp_path: Path) -> None:
    adapter = ColabEnterpriseAdapter(
        registry=_registry(tmp_path), env={"GOOGLE_CLOUD_PROJECT": "my-project"}
    )
    spec = make_spec(platform="colab_enterprise", gpu_required=True, max_cost_authorized_usd=5.0)

    from spectraquant.cloud.budget import PaidSubmissionNotAuthorized

    with pytest.raises(PaidSubmissionNotAuthorized):
        adapter.submit(spec, str(tmp_path / "run.ipynb"))


def test_colab_enterprise_requires_the_run_image_and_bucket(tmp_path: Path) -> None:
    adapter = ColabEnterpriseAdapter(
        registry=_registry(tmp_path),
        env={"GOOGLE_CLOUD_PROJECT": "my-project", "SPECTRAQUANT_ALLOW_PAID": "1"},
    )
    spec = make_spec(platform="colab_enterprise", gpu_required=True, max_cost_authorized_usd=5.0)

    with pytest.raises(ColabEnterpriseConfigError, match=IMAGE_ENV):
        adapter.submit(spec, str(tmp_path / "run.ipynb"))


def test_colab_enterprise_without_the_sdk_raises_a_named_error(tmp_path: Path) -> None:
    adapter = ColabEnterpriseAdapter(
        registry=_registry(tmp_path),
        env={
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "SPECTRAQUANT_ALLOW_PAID": "1",
            IMAGE_ENV: "gcr.io/my-project/spectraquant:1",
            BUCKET_ENV: "gs://my-staging-bucket",
        },
    )
    spec = make_spec(platform="colab_enterprise", gpu_required=True, max_cost_authorized_usd=5.0)
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    if adapter.sdk_available():  # pragma: no cover - the workstation has no GCP SDK
        pytest.skip("the Google Cloud SDK is installed; the missing-SDK branch cannot be exercised")

    with pytest.raises(NotImplementedError, match="google-cloud-aiplatform"):
        adapter.submit(spec, str(notebook))


def test_colab_enterprise_guard_passes_and_submits_through_an_injected_sdk(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    submitted: dict[str, object] = {}

    class FakeJob:
        resource_name = "projects/my-project/locations/us-central1/customJobs/123"

        def run(self, *, sync: bool) -> None:
            submitted["sync"] = sync

    class FakeAiplatform:
        @staticmethod
        def init(**kwargs: object) -> None:
            submitted["init"] = kwargs

        @staticmethod
        def CustomJob(**kwargs: object) -> FakeJob:
            submitted["job"] = kwargs
            return FakeJob()

    uploaded: dict[str, object] = {}

    class FakeBlob:
        def upload_from_filename(self, name: str) -> None:
            uploaded["file"] = name

    class FakeBucket:
        def blob(self, name: str) -> FakeBlob:
            uploaded["blob"] = name
            return FakeBlob()

    class FakeStorage:
        class Client:
            def __init__(self, project: str | None = None) -> None:
                uploaded["project"] = project

            def bucket(self, name: str) -> FakeBucket:
                uploaded["bucket"] = name
                return FakeBucket()

    sdk = SimpleNamespace(aiplatform=FakeAiplatform, storage=FakeStorage)
    adapter = ColabEnterpriseAdapter(
        registry=registry,
        env={
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "SPECTRAQUANT_ALLOW_PAID": "1",
            IMAGE_ENV: "gcr.io/my-project/spectraquant:1",
            BUCKET_ENV: "gs://my-staging-bucket",
        },
        sdk_loader=lambda: sdk,
    )
    spec = make_spec(platform="colab_enterprise", gpu_required=True, max_cost_authorized_usd=5.0)
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    remote_id = adapter.submit(spec, str(notebook))

    assert remote_id == "projects/my-project/locations/us-central1/customJobs/123"
    assert registry.remote_id(spec.run_id) == remote_id
    assert submitted["sync"] is False
    assert uploaded["bucket"] == "my-staging-bucket"
    job = submitted["job"]
    assert isinstance(job, dict)
    worker = job["worker_pool_specs"][0]
    assert worker["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_A100"


def test_colab_enterprise_authorize_reports_the_resolved_config(tmp_path: Path) -> None:
    adapter = ColabEnterpriseAdapter(
        registry=_registry(tmp_path),
        project="explicit-project",
        location="europe-west4",
        env={
            "SPECTRAQUANT_ALLOW_PAID": "1",
            IMAGE_ENV: "img",
            BUCKET_ENV: "gs://b",
        },
    )
    spec = make_spec(platform="colab_enterprise", gpu_required=True, max_cost_authorized_usd=5.0)

    assert adapter.authorize(spec) == {
        "project": "explicit-project",
        "location": "europe-west4",
        "image": "img",
        "bucket": "gs://b",
    }


# --------------------------------------------------------------------------------------
# local_cpu
# --------------------------------------------------------------------------------------
def test_local_cpu_refuses_gpu_required(tmp_path: Path) -> None:
    adapter = LocalCpuAdapter(registry=_registry(tmp_path), env={}, runs_root=tmp_path / "runs")
    spec = make_spec(platform="colab", gpu_required=True)  # platform mismatch is checked first

    with pytest.raises(LocalExecutionRefused, match="cannot run on the local_cpu adapter"):
        adapter.submit(spec, "x.ipynb")

    local_gpu = make_spec(platform="local_cpu", gpu_required=False)
    adapter.check_allowed(local_gpu)  # must not raise


def test_local_cpu_refuses_a_gpu_spec_declared_as_local(tmp_path: Path) -> None:
    adapter = LocalCpuAdapter(registry=_registry(tmp_path), env={}, runs_root=tmp_path / "runs")

    # A RunSpec cannot be built with local_cpu+gpu_required, so the guard is exercised directly.
    spec = make_spec(platform="local_cpu", gpu_required=False)
    object.__setattr__(spec, "gpu_required", True)

    with pytest.raises(LocalExecutionRefused, match="refuses gpu_required=True"):
        adapter.check_allowed(spec)


def test_local_cpu_submit_executes_and_records(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runs_root = tmp_path / "runs"
    seen: dict[str, object] = {}

    def runner(argv, env, cwd):  # type: ignore[no-untyped-def]
        seen["argv"] = [str(item) for item in argv]
        seen["cwd"] = cwd
        (Path(cwd) / "run.log").write_text("ran", encoding="utf-8")
        return CommandResult(argv=[str(item) for item in argv], returncode=0, stdout="done")

    adapter = LocalCpuAdapter(registry=registry, env={}, runner=runner, runs_root=runs_root)
    spec = make_spec(platform="local_cpu")

    run_id = adapter.submit(spec, "notebook.ipynb")

    assert run_id == spec.run_id
    assert "spectraquant" in seen["argv"]
    assert "--config" in seen["argv"]
    assert [record["state"] for record in registry.transitions(spec.run_id)] == [
        "submitted",
        "running",
        "finished",
    ]


def test_local_cpu_records_a_failure(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    adapter = LocalCpuAdapter(
        registry=registry,
        env={},
        runner=lambda argv, env, cwd: CommandResult(
            argv=[str(item) for item in argv], returncode=2, stderr="No such command 'run'"
        ),
        runs_root=tmp_path / "runs",
    )
    spec = make_spec(platform="local_cpu")

    adapter.submit(spec, "notebook.ipynb")

    assert registry.state(spec.run_id) == "failed"
    assert "exited 2" in (registry.latest(spec.run_id)["failure_reason"] or "")


def test_local_cpu_status_and_fetch(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runs_root = tmp_path / "runs"
    adapter = LocalCpuAdapter(
        registry=registry,
        env={},
        runner=lambda argv, env, cwd: CommandResult(
            argv=[str(item) for item in argv], returncode=0
        ),
        runs_root=runs_root,
    )
    spec = make_spec(platform="local_cpu")
    adapter.submit(spec, "notebook.ipynb")

    assert adapter.status(spec.run_id).state == "finished"
    report = adapter.fetch(spec.run_id, str(tmp_path / "copy"))
    assert report.ok is True
    assert "run.log" in report.files
    assert adapter.fetch("missing-run", str(tmp_path / "copy2")).ok is False


# --------------------------------------------------------------------------------------
# Polling
# --------------------------------------------------------------------------------------
def test_poll_until_terminal_stops_at_a_terminal_state() -> None:
    class Adapter:
        name = "scripted"

        def __init__(self) -> None:
            self.calls = 0

        def status(self, run_id: str) -> RunStatus:
            self.calls += 1
            return RunStatus(run_id=run_id, state="running" if self.calls < 3 else "finished")

    adapter = Adapter()
    sleeps: list[float] = []

    status = poll_until_terminal(
        adapter, "run-1", policy=PollPolicy(max_attempts=10), sleep=sleeps.append
    )

    assert status.state == "finished"
    assert adapter.calls == 3
    # The first query is immediate; the two retries wait 5 s then 10 s (bounded backoff).
    assert sleeps == [5.0, 10.0]


def test_poll_policy_is_bounded() -> None:
    policy = PollPolicy(max_attempts=5, base_delay_s=1.0, max_delay_s=2.0)

    assert [policy.delay_for(i) for i in range(1, 6)] == [0.0, 1.0, 2.0, 2.0, 2.0]
    assert policy.max_total_wait_s() == 7.0


def test_poll_returns_unknown_when_the_budget_runs_out() -> None:
    class Stuck:
        name = "stuck"

        def status(self, run_id: str) -> RunStatus:
            return RunStatus(run_id=run_id, state="running")

    status = poll_until_terminal(
        Stuck(), "run-1", policy=PollPolicy(max_attempts=2), sleep=lambda _: None
    )

    assert status.state == "running"


def test_command_result_redacts_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_command_secret_value")

    result = CommandResult(
        argv=["tool"], returncode=1, stderr="failed with hf_command_secret_value"
    )

    assert "hf_command_secret_value" not in result.combined
    assert "[REDACTED:HF_TOKEN]" in result.combined
    assert result.ok is False
