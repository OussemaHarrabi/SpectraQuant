"""Pinned-asset access for frozen cloud plans (``configs/{tier1,tier2,repro}/*.yaml``).

A plan pins its models and datasets by **immutable repository revision**, never by a locally
computable payload hash: a dataset of that size cannot be hashed without downloading it, and a run
must not silently fall back to an unpinned revision. This module is the single place where that
guarantee is enforced, so both the notebook's data cell and :mod:`spectraquant.cloud.plan_runner`
verify the same thing:

* :func:`is_plan_document` — is a path a plan (rather than a Hydra composition config)?
* :func:`verify_pinned_revision` — the revision must be a full 40-hex commit and the upstream
  repository must resolve to exactly that commit. Anything else is a loud error.
* :func:`materialise_plan_datasets` — the notebook data cell's entry point: verify every declared
  dataset pin and record it. It deliberately does **not** bulk-download C4 (1 TB): the runner streams
  the splits it consumes, and the manifest records both the requested and the resolved revision.
* :func:`load_plan_texts` — one document-level split of a pinned dataset, for the perplexity cell.

Dependencies (``transformers``/``datasets``/``huggingface_hub``) are imported lazily and their absence
raises a message naming the ``models`` extra, mirroring the ``cloud`` extra convention.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spectraquant.cloud.spec import dataset_declaration_digest
from spectraquant.experiment_plan import DatasetRole, PlanConfig, load_plan, plan_path

__all__ = [
    "PinnedRevisionError",
    "is_plan_document",
    "load_plan_texts",
    "materialise_plan_datasets",
    "plan_dataset_ref",
    "require_datasets",
    "require_transformers",
    "verify_pinned_revision",
]

#: A revision is immutable only when it is a full commit sha; tags and branches move.
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")

_MISSING_EXTRA_HINT = (
    "install it with `uv sync --frozen --extra models` (or `uv pip install 'transformers>=4.44,<5' "
    "'datasets>=2.20,<3'`)"
)


class PinnedRevisionError(ValueError):
    """Raised when a plan's asset pin is not an immutable revision, or does not resolve to it.

    A run whose model or dataset revision cannot be proven is not evidence: the failure is loud and
    the run does not continue with a substituted asset (``AGENTS.md`` §4.11, §2b rule 3).
    """


def require_transformers() -> Any:
    """Import ``transformers`` lazily.

    Returns:
        The imported ``transformers`` module.

    Raises:
        ImportError: ``transformers`` is not installed; the message names the ``models`` extra.
    """
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised through the missing-extra test
        raise ImportError(
            f"loading a pretrained model requires the optional 'models' extra: {_MISSING_EXTRA_HINT}"
        ) from exc
    return transformers


def require_datasets() -> Any:
    """Import ``datasets`` lazily.

    Returns:
        The imported ``datasets`` module.

    Raises:
        ImportError: ``datasets`` is not installed; the message names the ``models`` extra.
    """
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - exercised through the missing-extra test
        raise ImportError(
            f"loading a pinned dataset requires the optional 'models' extra: {_MISSING_EXTRA_HINT}"
        ) from exc
    return datasets


def is_plan_document(path: str | Path) -> bool:
    """True when ``path`` holds a cloud **plan** rather than a Hydra composition config.

    Detection is by document shape (``substrate`` + ``arms`` + ``tier`` + ``measurement_classes``),
    not by directory: a plan copied elsewhere is still a plan, and a Hydra config is never one.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a hard dependency via hydra
        return False
    target = Path(path)
    if not target.is_file():
        return False
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(loaded, dict) and all(
        key in loaded for key in ("substrate", "arms", "tier", "measurement_classes")
    )


def verify_pinned_revision(repo_id: str, revision: str, *, repo_type: str) -> str:
    """Prove that ``revision`` is an immutable commit and that the upstream repo resolves to it.

    Args:
        repo_id: Hugging Face repository id (``owner/name``).
        revision: the revision the plan pins.
        repo_type: ``"model"`` or ``"dataset"``; selects the metadata endpoint.

    Returns:
        The resolved commit sha (identical to ``revision`` by construction).

    Raises:
        PinnedRevisionError: the revision is not a full 40-hex commit, or the upstream repo's commit
            for that revision differs, or the metadata lookup fails. Never falls back to a floating
            revision.
    """
    if not _IMMUTABLE_REVISION.match(str(revision)):
        raise PinnedRevisionError(
            f"{repo_type} {repo_id!r} is pinned to revision {revision!r}, which is not an immutable "
            "40-hex commit: a tag or branch can move, so the run could not be reproduced "
            "(AGENTS.md §4.11)"
        )
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - huggingface_hub ships with the models extra
        raise ImportError(
            f"verifying a pinned revision requires 'huggingface_hub': {_MISSING_EXTRA_HINT}"
        ) from exc

    api = HfApi()
    try:
        info = (
            api.dataset_info(repo_id, revision=revision)
            if repo_type == "dataset"
            else api.model_info(repo_id, revision=revision)
        )
    except Exception as exc:  # surfaced verbatim, never swallowed
        raise PinnedRevisionError(
            f"could not resolve {repo_type} {repo_id!r} at revision {revision!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    resolved = str(info.sha)
    if resolved != revision:
        raise PinnedRevisionError(
            f"{repo_type} {repo_id!r} resolved to {resolved} != pinned revision {revision!r}: "
            "the pin does not identify this commit"
        )
    return resolved


def materialise_plan_datasets(
    config_path: str | Path | PlanConfig,
    dataset_refs: Sequence[Mapping[str, Any]],
    dest_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Verify every plan dataset pin and return the records the notebook's data cell writes.

    The verification is revision-level, not payload-level: a plan dataset is anchored by its
    immutable commit, and the declared checksum is
    :func:`~spectraquant.cloud.spec.dataset_declaration_digest` over ``(name, revision, split)``, so
    a tampered spec is detected here. Bulk materialisation is deliberately out of scope: the runner
    streams the splits it consumes (C4 train is ~1 TB), and the per-split fingerprint a run measured
    is recorded in that run's manifest.

    Args:
        config_path: the plan document (used for the license/identity record).
        dataset_refs: ``RunSpec``-shaped refs: ``{name, revision, split, checksum}``.
        dest_dir: optional directory that receives ``datasets.json``.

    Returns:
        One record per ref: ``name``, ``split``, ``revision_requested``, ``revision_resolved``,
        ``declared_checksum``, ``verified``.

    Raises:
        FileNotFoundError: the plan does not exist.
        PinnedRevisionError: a revision is floating, unresolvable, or the ref's declared checksum
            disagrees with the plan.
        TypeError: ``dataset_refs`` does not carry the expected mapping shape.
    """
    config: PlanConfig = (
        load_plan(plan_path(config_path))
        if not isinstance(config_path, PlanConfig)
        else config_path
    )
    planned = {(str(d.name), str(d.revision)): d for d in config.datasets}

    records: list[dict[str, Any]] = []
    for ref in dataset_refs:
        if not isinstance(ref, Mapping):
            raise TypeError(f"dataset refs must be mappings, got {type(ref).__name__}")
        name = str(ref["name"])
        revision = str(ref["revision"])
        split = str(ref["split"])
        if (name, revision) not in planned:
            raise PinnedRevisionError(
                f"dataset {name!r}@{revision!r} is not declared by plan {config.name!r}: a run may "
                "only consume datasets the plan authorised"
            )
        resolved = verify_pinned_revision(name, revision, repo_type="dataset")
        declared = dataset_declaration_digest(name, revision, split)
        if str(ref.get("checksum", "")) != declared:
            raise PinnedRevisionError(
                f"dataset ref checksum for {name!r}@{revision!r} ({split}) is "
                f"{ref.get('checksum')!r}, expected the declaration digest {declared!r}: the spec "
                "does not match the frozen plan"
            )
        records.append(
            {
                "name": name,
                "split": split,
                "revision_requested": revision,
                "revision_resolved": resolved,
                "declared_checksum": declared,
                "checksum_semantics": "revision-declaration-digest",
                "verified": True,
            }
        )

    if dest_dir is not None:
        target = Path(dest_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "datasets.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return records


def plan_dataset_ref(config: PlanConfig, role: str, *, split: str | None = None) -> DatasetRole:
    """Return the plan dataset carrying ``role`` (raising when none does).

    Raises:
        KeyError: no dataset in the plan carries ``role``.
    """
    for dataset in config.datasets:
        if role in dataset.roles:
            return dataset
    raise KeyError(
        f"plan {config.name!r} declares no dataset for role {role!r} "
        f"(roles in the plan: {sorted({r for d in config.datasets for r in d.roles})})"
    )


def load_plan_texts(
    ref: DatasetRole,
    *,
    split: str,
    dataset_config: str | None = None,
    max_documents: int | None = None,
    stream: bool = False,
) -> list[str]:
    """Load one document-level split of a pinned dataset as a list of texts.

    The revision is passed through to ``datasets.load_dataset`` verbatim, so the loader either
    materialises exactly that commit or fails; there is no floating fallback.

    Args:
        ref: the plan's dataset entry (name, revision, license, roles).
        split: split to read (``test`` for the perplexity role).
        dataset_config: the dataset's *config* name (e.g. ``wikitext-2-raw-v1``). The frozen plans
            pin the repository and the revision but not the config name; when the repository exposes
            several configs, pass it explicitly — the runner never guesses.
        max_documents: optional cap, applied to the head of the split (deterministic).
        stream: read the split as a stream, taking only the first ``max_documents`` rows. Required
            for a very large split (C4's ``en`` train is hundreds of gigabytes) and refused without a
            cap, because an unbounded streaming read has no end. The pinned revision still applies.

    Returns:
        The document texts, in dataset order.

    Raises:
        PinnedRevisionError: the revision is not an immutable commit.
        ImportError: the ``models`` extra is missing.
        ValueError: no text column could be identified.
    """
    verify_pinned_revision(ref.name, ref.revision, repo_type="dataset")
    datasets = require_datasets()
    kwargs: dict[str, Any] = {"split": split, "revision": ref.revision}
    if dataset_config:
        kwargs["name"] = dataset_config
    # The pinned revision of a script dataset is the trust anchor: loading it requires running the
    # repository's own loader at that commit (`datasets` >= 2.16 asks for this explicitly).
    kwargs["trust_remote_code"] = True
    if stream:
        # A bounded read of a very large split MUST stream. `allenai/c4`'s `en` train split is
        # hundreds of gigabytes, and a non-streaming `load_dataset` materialises it: the first
        # training run sat for over an hour downloading and never reached a single optimiser step.
        # Streaming fetches only the rows the cap asks for, and the pinned revision still applies.
        if max_documents is None:
            raise ValueError(
                f"streaming {ref.name!r} ({split}) requires max_documents: an unbounded streaming "
                "read has no natural end"
            )
        kwargs["streaming"] = True

    dataset = datasets.load_dataset(ref.name, **kwargs)

    def _text_column(columns: list[str], probe: Any) -> str:
        column = "text" if "text" in columns else None
        if column is None:
            for candidate in columns:
                if isinstance(probe(candidate), str):
                    column = candidate
                    break
        if column is None:
            raise ValueError(
                f"dataset {ref.name!r} ({split}) exposes no string column to use as document text "
                f"(columns: {columns})"
            )
        return column

    if stream:
        assert max_documents is not None  # guaranteed by the check above
        cap = int(max_documents)
        texts: list[str] = []
        column: str | None = None
        for index, row in enumerate(dataset):
            if index >= cap:
                break
            if column is None:
                first = row
                column = _text_column(list(first.keys()), first.get)
            value = row[column]
            if isinstance(value, str) and value.strip():
                texts.append(value)
        if not texts:
            raise ValueError(
                f"streaming {ref.name!r} ({split}) produced no document in the first {cap} rows"
            )
        return texts

    column = _text_column(list(dataset.column_names), lambda name: dataset[0][name])
    texts = []
    for index, row in enumerate(dataset):
        if max_documents is not None and index >= max_documents:
            break
        value = row[column]
        if isinstance(value, str) and value.strip():
            texts.append(value)
    return texts
