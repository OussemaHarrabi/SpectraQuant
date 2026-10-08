"""Structured logging for SpectraQuant runs.

Log level comes from the environment (``SPECTRAQUANT_LOG_LEVEL``, default ``INFO``); JSON output is
enabled with ``SPECTRAQUANT_LOG_JSON=1``. Structured events are emitted through ``log_event``,
which attaches ``event`` plus arbitrary ``key=value`` fields to a normal ``logging`` record.

Nothing here writes files: run artifacts are captured by the caller (see
:mod:`spectraquant.reporting.manifests`) so that logging never becomes an implicit output channel.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

__all__ = ["configure_logging", "get_logger", "log_event", "reset_logging"]

_LEVEL_ENV = "SPECTRAQUANT_LOG_LEVEL"
_JSON_ENV = "SPECTRAQUANT_LOG_JSON"
_DEFAULT_LEVEL = "INFO"

_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
#: Attributes the logging module itself sets while formatting a record; ``extra`` must not touch them.
_DERIVED = frozenset({"asctime", "message"})


class _TextFormatter(logging.Formatter):
    """``2026-10-08T12:00:00Z INFO spectraquant.smoke event=step step=10 loss=1.234``"""

    _DERIVED = _DERIVED | {"event"}

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and key not in self._DERIVED
        }
        event = getattr(record, "event", None)
        if event is None:
            return super().format(record)
        parts = [
            self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            record.levelname,
            record.name,
            f"event={event}",
        ]
        parts.extend(f"{key}={_render(value)}" for key, value in sorted(fields.items()))
        return " ".join(parts)


class _JsonFormatter(logging.Formatter):
    """One JSON object per line; machine-readable for CI and post-hoc analysis."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value if isinstance(value, (int, float, str, bool)) else str(value)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _render(value: object) -> str:
    if isinstance(value, str):
        return value if " " not in value else json.dumps(value)
    return str(value)


def _resolve_level(default: str = _DEFAULT_LEVEL) -> int:
    raw = os.environ.get(_LEVEL_ENV, default).strip().upper()
    if raw not in _LEVELS:
        raise ValueError(f"{_LEVEL_ENV} must be one of {sorted(_LEVELS)}, got {raw!r}")
    return _LEVELS[raw]


def configure_logging(level: str | int | None = None, *, force: bool = False) -> logging.Logger:
    """Configure the ``spectraquant`` root logger once and return it.

    Args:
        level: explicit level (name or numeric). ``None`` reads ``SPECTRAQUANT_LOG_LEVEL``.
        force: replace handlers even if the logger was already configured.
    """
    logger = logging.getLogger("spectraquant")
    if logger.handlers and not force:
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if isinstance(level, str):
        numeric = _LEVELS.get(level.strip().upper())
        if numeric is None:
            raise ValueError(f"unknown log level {level!r}")
    elif level is None:
        numeric = _resolve_level()
    else:
        numeric = int(level)

    handler = logging.StreamHandler()
    handler.setFormatter(
        _JsonFormatter()
        if os.environ.get(_JSON_ENV, "").strip() in {"1", "true", "yes"}
        else _TextFormatter()
    )
    logger.addHandler(handler)
    logger.setLevel(numeric)
    logger.propagate = False
    return logger


def reset_logging() -> None:
    """Drop handlers (test helper)."""
    logger = logging.getLogger("spectraquant")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the configured ``spectraquant`` logger."""
    if not name.startswith("spectraquant"):
        name = f"spectraquant.{name}"
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    message: str | None = None,
    fields: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured event: ``event=<name>`` plus ``fields`` as record attributes."""
    extra: dict[str, Any] = {"event": event}
    for key, value in (fields or {}).items():
        if key in _RESERVED or key in _DERIVED or key == "event":
            raise ValueError(f"log field {key!r} would shadow a reserved logging attribute")
        extra[key] = value
    logger.log(level, message if message is not None else event, extra=extra)
