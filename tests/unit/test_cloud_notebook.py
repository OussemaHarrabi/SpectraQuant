"""Notebook generation: determinism, mandatory cells and credential safety."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from _cloud_fixtures import COMMIT, make_spec

from spectraquant.cloud.notebook import (
    NOTEBOOK_TEMPLATE_VERSION,
    PENDING_DIGEST,
    STAGES,
    build_notebook,
    cell_stages,
    notebook_digest,
    notebook_text,
    write_notebook,
)

nbformat = pytest.importorskip("nbformat", reason="the optional 'cloud' extra provides nbformat")


def _mandatory_cells(spec: object) -> list[str]:
    notebook = build_notebook(spec)
    return [str(cell.source) for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_is_byte_identical_for_the_same_spec() -> None:
    spec = make_spec()

    first = notebook_text(spec)
    second = notebook_text(spec)

    assert first == second
    assert notebook_digest(spec) == notebook_digest(spec)


def test_different_commit_changes_the_digest() -> None:
    first = make_spec(git_commit=COMMIT)
    second = make_spec(git_commit="1" * 40)

    assert notebook_digest(first) != notebook_digest(second)


def test_different_run_id_changes_the_digest() -> None:
    assert notebook_digest(make_spec(run_id="run-a")) != notebook_digest(make_spec(run_id="run-b"))


def test_digest_matches_the_canonical_serialization() -> None:
    spec = make_spec()
    canonical = notebook_text(spec).replace(notebook_digest(spec), PENDING_DIGEST)
    # The canonical form (digest literal zeroed) is what the digest hashes.
    assert (
        notebook_digest(spec) == "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )


def test_digest_is_embedded_in_the_generated_notebook() -> None:
    spec = make_spec()

    assert notebook_digest(spec) in notebook_text(spec)
    assert spec.sha256() in notebook_text(spec)
    assert NOTEBOOK_TEMPLATE_VERSION in notebook_text(spec)


def test_mandatory_cells_are_present_in_the_frozen_order() -> None:
    spec = make_spec()
    stages = cell_stages(spec)

    assert [entry["stage"] for entry in stages] == ["header", *STAGES]
    assert [entry["index"] for entry in stages] == list(range(len(stages)))


def test_environment_cell_records_hardware_and_start_time() -> None:
    source = _mandatory_cells(make_spec())[0]

    for token in ("nvidia-smi", "sys.version", "pip", "freeze", "STARTED_UTC", "ram_bytes"):
        assert token in source


def test_install_cell_runs_the_pinned_spec() -> None:
    source = _mandatory_cells(make_spec(install_spec="uv sync --frozen --extra cloud"))[1]

    assert "INSTALL_SPEC" in source
    assert 'SPEC["install_spec"]' in source
    assert "bash" in source


def test_repo_cell_asserts_the_exact_commit() -> None:
    source = _mandatory_cells(make_spec())[2]

    assert '"rev-parse"' in source
    assert '"HEAD"' in source
    assert 'assert HEAD_SHA == SPEC["git_commit"]' in source
    assert "git checkout" in source


def test_teardown_cell_prints_the_teardown_marker() -> None:
    source = _mandatory_cells(make_spec())[7]

    assert "SPECTRAQUANT_TEARDOWN_OK" in source


def test_write_notebook_returns_the_digest_and_writes_the_same_bytes(tmp_path: Path) -> None:
    spec = make_spec()
    target = tmp_path / "generated" / f"{spec.run_id}.ipynb"

    digest = write_notebook(spec, target)

    assert digest == notebook_digest(spec)
    assert target.read_text(encoding="utf-8") == notebook_text(spec)
    parsed = nbformat.read(str(target), as_version=4)
    assert parsed.nbformat == 4
    assert len(parsed.cells) == len(STAGES) + 1


def test_a_secret_in_the_environment_never_reaches_the_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_KEY", "kaggle_super_secret_value_123")
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_token_value")
    spec = make_spec(
        repo_url="https://kaggle_super_secret_value_123@example.invalid/repo.git",
        overrides=["data.token=hf_super_secret_token_value"],
    )

    text = notebook_text(spec)

    assert "kaggle_super_secret_value_123" not in text
    assert "hf_super_secret_token_value" not in text
    assert "[REDACTED:KAGGLE_KEY]" in text
    assert "[REDACTED:HF_TOKEN]" in text
    assert "KAGGLE_KEY" in text  # the variable *name* is allowed; the value never is


def test_notebook_generation_requires_the_cloud_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nbformat", None)

    with pytest.raises(ImportError, match="'cloud' extra"):
        notebook_text(make_spec())


def test_notebook_metadata_records_the_generator_and_spec_digest() -> None:
    spec = make_spec()
    notebook = build_notebook(spec)

    metadata = notebook.metadata["spectraquant"]

    assert metadata["generated"] is True
    assert metadata["run_id"] == spec.run_id
    assert metadata["spec_sha256"] == spec.sha256()
    assert metadata["template_version"] == NOTEBOOK_TEMPLATE_VERSION


def test_cell_ids_are_deterministic_and_positional() -> None:
    spec = make_spec()

    first = [cell["id"] for cell in build_notebook(spec).cells]
    second = [cell["id"] for cell in build_notebook(spec).cells]

    assert first == second
    assert first[0] == "sq-00-header"
    assert first[1] == "sq-01-environment"
    assert first[-1] == f"sq-0{len(STAGES)}-teardown"


def test_generated_notebook_is_valid_json_without_the_extra(tmp_path: Path) -> None:
    """The written file must parse as plain JSON (nbformat is only needed to build it)."""
    spec = make_spec()
    target = tmp_path / f"{spec.run_id}.ipynb"

    write_notebook(spec, target)

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["nbformat"] == 4
    assert len(document["cells"]) == len(STAGES) + 1


def test_scratch_workdir_keeps_the_environment_out_of_the_exported_tree() -> None:
    """The run's checkout and virtual environment must not land in the platform's output directory.

    Regression for the first real Kaggle run: the workdir defaulted to the notebook's cwd, Kaggle
    exports everything under it, and a fetch pulled 148 MB of ``.venv`` before being abandoned.
    """
    from _cloud_fixtures import make_spec

    from spectraquant.cloud.notebook import notebook_text

    text = notebook_text(make_spec(platform="kaggle", gpu_required=False))
    assert "/kaggle/temp" in text, "the scratch workdir must prefer the non-exported temp dir"
    assert "SPECTRAQUANT_WORKDIR" in text, "an explicit override must still win"
    assert "spectraquant-export" in text, "the export dir must stay inside the exported tree"


def _code_cells(spec: object) -> list[str]:
    import json as _json

    from spectraquant.cloud.notebook import notebook_text

    document = _json.loads(notebook_text(spec))
    return [
        "".join(cell["source"]) for cell in document["cells"] if cell.get("cell_type") == "code"
    ]


def test_every_generated_code_cell_compiles() -> None:
    """A generated notebook is executed by papermill, so a syntax error is a failed run.

    Regression for a template edit that referenced names (``_time``, ``HEAD_SHA``) the notebook never
    defined and left a docstring inside the template string: neither showed up until a platform run
    failed. Compiling each cell catches the second class locally.
    """
    from _cloud_fixtures import make_spec

    for index, source in enumerate(_code_cells(make_spec(platform="kaggle", gpu_required=False))):
        compile(source, f"<cell {index}>", "exec")


def test_no_cell_imports_repository_code_into_the_kernel() -> None:
    """The kernel runs the platform's Python; repository code must only run through `uv run`.

    Regression for the first real Kaggle run: the kernel was python 3.13 while the project pins 3.11
    and a CPU torch build, so importing repository code in the kernel could never work.
    """
    from _cloud_fixtures import make_spec

    for index, source in enumerate(_code_cells(make_spec(platform="kaggle", gpu_required=False))):
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import spectraquant"), (index, line)
            assert not stripped.startswith("from spectraquant"), (index, line)


def test_the_run_cell_executes_the_frozen_command_in_the_pinned_environment() -> None:
    """The runner is the spec's command, run through `uv run`, and its exit is recorded not raised."""
    from _cloud_fixtures import make_spec

    cells = "\n".join(_code_cells(make_spec(platform="kaggle", gpu_required=False)))
    assert 'RUNNER_COMMAND = SPEC.get("runner_command")' in cells
    assert "_repo_run(RUN_ARGV)" in cells
    assert '"run_rc": RUN_RC' in cells
    assert "run.log" in cells


def test_the_notebook_delegates_every_repository_stage() -> None:
    """datasets, manifest and export are stage calls, so the kernel imports no repository code."""
    from _cloud_fixtures import make_spec

    cells = "\n".join(_code_cells(make_spec(platform="kaggle", gpu_required=False)))
    for stage in ("datasets", "manifest", "export"):
        assert f'_stage("{stage}")' in cells, stage
    assert "notebook-stage" in cells
    assert "SPECTRAQUANT_WORKDIR" in cells


def test_the_gpu_check_is_in_the_install_cell_and_stages_never_re_sync() -> None:
    """Two defects from the first GPU run, both silent until the training stage.

    1. `uv run` re-syncs the environment from the lock, which resolves torch from the CPU index, so
       every stage invocation put the CPU build back and the run died with "Torch not compiled with
       CUDA enabled" after the install had already replaced torch with the CUDA build. `--no-sync` is
       therefore required, not an optimisation.
    2. The CUDA check must run *after* the install (it asserts the CUDA build is the one installed)
       and before the model download, so a GPU plan fails fast rather than minutes later.
    """
    cells = _mandatory_cells(make_spec(install_spec="uv sync --frozen --extra cloud"))
    environment, install = cells[0], cells[1]

    assert "--no-sync" in install
    assert "CUDA check" in install
    assert "torch.cuda.is_available()" in install
    assert 'SPEC["gpu_required"]' in install
    assert "CUDA check" not in environment, "the check must not run before torch is installed"
    # The check must come after the pinned install and before the dependency freeze.
    assert install.index('_require_ok(_install_rc') < install.index("CUDA check")
    assert install.index("CUDA check") < install.index('"-m", "pip", "freeze"')
