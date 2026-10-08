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


def test_data_cell_verifies_checksums_through_repo_code() -> None:
    source = _mandatory_cells(make_spec())[3]

    assert "materialise_datasets" in source
    assert "dataset_refs" in source


def test_run_cell_invokes_the_frozen_cli_command() -> None:
    source = _mandatory_cells(make_spec())[4]

    assert "spectraquant" in source
    assert '"run"' in source
    assert "RUN_OVERRIDES" in source


def test_manifest_cell_writes_a_schema_validated_manifest() -> None:
    source = _mandatory_cells(make_spec())[5]

    assert "build_run_manifest" in source
    assert "write_run_manifest" in source
    assert "GPU_HOURS" in source


def test_export_cell_prints_the_machine_readable_result_line() -> None:
    source = _mandatory_cells(make_spec())[6]

    assert 'print("SPECTRAQUANT_RESULT_JSON=" + json.dumps(' in source
    assert "EXECUTED_NOTEBOOK" in source


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
