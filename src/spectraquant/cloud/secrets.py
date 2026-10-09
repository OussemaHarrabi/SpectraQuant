"""Credential access and redaction for the cloud substrate (``design-cloud-adapter.md`` §8).

Credentials are read **only** from environment variables / platform secret stores. This module
never writes a credential to disk, never includes one in a notebook cell, and provides the single
redaction helper used on every log line, error message and emitted notebook.

The redaction set is deliberately conservative about *values*: a value shorter than
:data:`MIN_REDACTABLE_LENGTH` (or a bare flag such as ``"0"``/``"1"``) is never substituted, because
replacing the string ``"1"`` everywhere would corrupt the output instead of protecting it.

Public API:

* :data:`SECRET_ENV_VARS` — the only credential names this project reads.
* :func:`secret_values` — the (redactable) credential values currently present.
* :func:`redact` / :func:`redact_mapping` / :func:`redact_json` — scrub text and structures.
* :func:`install_redacting_handler` — attach the ``logging`` filter used by the CLI.
* :func:`assert_no_secret` — fail loudly when a credential value escaped into generated output.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "CREDENTIAL_ENV_VARS",
    "IDENTIFIER_ENV_VARS",
    "MIN_REDACTABLE_LENGTH",
    "REDACTION_PLACEHOLDER",
    "SECRET_ENV_VARS",
    "MissingCredentials",
    "assert_no_secret",
    "has_secret",
    "install_redacting_handler",
    "redact",
    "redact_json",
    "redact_mapping",
    "require_credentials",
    "scrub_lines",
    "secret_values",
]

#: Environment variables holding credentials or paid-resource authorization. Their *values* are
#: secrets: they are redacted from every log, manifest and generated cell.
SECRET_ENV_VARS: tuple[str, ...] = (
    "KAGGLE_KEY",
    "HF_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "SPECTRAQUANT_ALLOW_PAID",
)

#: Environment variables read for authentication whose values are *public identifiers*, not secrets.
#: A Kaggle username appears in every kernel URL, so scrubbing it corrupts the remote id the registry
#: must address on the next call (observed: the collected bundle was fetched against
#: ``[REDACTED:KAGGLE_USERNAME]/...`` and the platform answered "Permission 'kernels.get' was denied").
IDENTIFIER_ENV_VARS: tuple[str, ...] = ("KAGGLE_USERNAME",)

#: Every variable the cloud adapters read from the environment: secrets and identifiers.
CREDENTIAL_ENV_VARS: tuple[str, ...] = SECRET_ENV_VARS + IDENTIFIER_ENV_VARS

#: Values that are control flags rather than credentials: never redacted as substrings.
_FLAG_VALUES = frozenset({"0", "1", "true", "false", "yes", "no", "on", "off"})

#: Shortest value that may be redacted; below this a substring replacement would be destructive.
MIN_REDACTABLE_LENGTH = 6

#: Replacement format. Contains the variable name so a reviewer can see what was scrubbed.
REDACTION_PLACEHOLDER = "[REDACTED:{name}]"


class MissingCredentials(RuntimeError):
    """Raised when a platform credential required for a submission is absent from the environment.

    The message names the *variable*, never a value: credentials come from the environment or the
    platform's secret store and are never written to the repository (``AGENTS.md`` §2b rule 7).
    """


def secret_values(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the redactable credential values present in ``env`` (default: ``os.environ``).

    Args:
        env: environment mapping to inspect. ``None`` means the live process environment.

    Returns:
        ``{variable_name: value}`` for every name in :data:`SECRET_ENV_VARS` that is set, non-empty,
        not a bare control flag, and at least :data:`MIN_REDACTABLE_LENGTH` characters long.
    """
    source = os.environ if env is None else env
    values: dict[str, str] = {}
    for name in SECRET_ENV_VARS:
        raw = source.get(name)
        if not raw:
            continue
        value = str(raw).strip()
        if len(value) < MIN_REDACTABLE_LENGTH or value.lower() in _FLAG_VALUES:
            continue
        values[name] = value
    return values


def redact(text: str, env: Mapping[str, str] | None = None) -> str:
    """Return ``text`` with every known credential value replaced by a labelled placeholder.

    Args:
        text: text to scrub (a log line, an error message, or a serialized notebook).
        env: environment to read credentials from; ``None`` means the live process environment.

    Returns:
        The scrubbed text. Redaction is idempotent: placeholders contain no credential value.
    """
    scrubbed = text
    for name, value in secret_values(env).items():
        if value in scrubbed:
            scrubbed = scrubbed.replace(value, REDACTION_PLACEHOLDER.format(name=name))
    return scrubbed


def redact_mapping(value: Any, env: Mapping[str, str] | None = None) -> Any:
    """Recursively redact strings inside mappings, sequences and scalars."""
    if isinstance(value, str):
        return redact(value, env)
    if isinstance(value, Mapping):
        return {key: redact_mapping(item, env) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return type(value)(redact_mapping(item, env) for item in value)
    return value


def redact_json(value: Any, env: Mapping[str, str] | None = None, *, indent: int | None = 2) -> str:
    """Serialize ``value`` to JSON with every credential value redacted."""
    return redact(json.dumps(redact_mapping(value, env), indent=indent, sort_keys=True), env)


def has_secret(text: str, env: Mapping[str, str] | None = None) -> bool:
    """True when any known credential value appears verbatim in ``text``."""
    return any(value in text for value in secret_values(env).values())


def require_credentials(
    names: Iterable[str], env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return the named credentials, raising when any is missing or empty.

    Args:
        names: environment variable names the platform requires (e.g. ``KAGGLE_USERNAME``).
        env: environment mapping; ``None`` means the live process environment.

    Returns:
        ``{name: value}`` — callers pass these straight to a subprocess environment and never log
        them.

    Raises:
        MissingCredentials: one or more variables are unset or empty.
    """
    source = os.environ if env is None else env
    missing = [name for name in names if not str(source.get(name) or "").strip()]
    if missing:
        raise MissingCredentials(
            f"missing credential environment variable(s): {', '.join(missing)}. Credentials live in "
            "the environment or the platform secret store only — never in the repository "
            "(AGENTS.md §2b rule 7; see scripts/cloud/README.md)"
        )
    return {name: str(source[name]) for name in names}


def assert_no_secret(
    text: str, env: Mapping[str, str] | None = None, *, where: str = "generated output"
) -> None:
    """Raise :class:`PermissionError` when a credential value leaked into ``text``.

    Args:
        text: the artifact about to be written or logged.
        env: environment to read credentials from; ``None`` means the live process environment.
        where: human-readable location used in the error message.

    Raises:
        PermissionError: a credential value appears verbatim in ``text``.
    """
    leaked = [name for name, value in secret_values(env).items() if value in text]
    if leaked:
        raise PermissionError(
            f"credential value for {', '.join(sorted(leaked))} would have been written to {where}; "
            "refusing to emit it (design-cloud-adapter.md §8)"
        )


class RedactingFilter(logging.Filter):
    """``logging`` filter that scrubs credential values from every record it sees."""

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        super().__init__(name="spectraquant.cloud.redaction")
        self._env = env

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.msg), self._env)
        record.args = redact_mapping(record.args, self._env)
        if record.exc_text:
            record.exc_text = redact(record.exc_text, self._env)
        return True


def install_redacting_handler(
    logger: logging.Logger | None = None, env: Mapping[str, str] | None = None
) -> RedactingFilter:
    """Attach a :class:`RedactingFilter` to ``logger`` (default: this package's logger).

    Returns:
        The installed filter, so tests can assert on it.
    """
    target = logger if logger is not None else logging.getLogger("spectraquant.cloud")
    redacting = RedactingFilter(env)
    for handler in list(target.handlers) or [logging.NullHandler()]:
        handler.addFilter(redacting)
    if not target.handlers:
        target.addHandler(logging.NullHandler())
    return redacting


def scrub_lines(lines: Iterable[str], env: Mapping[str, str] | None = None) -> list[str]:
    """Redact every line of ``lines`` (used for captured subprocess output)."""
    return [redact(line, env) for line in lines]
