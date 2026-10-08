"""Adapter protocol and shared plumbing (``design-cloud-adapter.md`` §4).

Every platform adapter implements the same four operations. The protocol is deliberately small so a
platform's honest capability gap shows up as an explicit refusal (``NotImplementedError`` with a
reason) rather than as a fabricated success.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from spectraquant.cloud.secrets import redact

__all__ = [
    "REMOTE_STATES",
    "TERMINAL_STATES",
    "CommandResult",
    "CommandRunner",
    "FetchReport",
    "PlatformAdapter",
    "PollPolicy",
    "RunStatus",
    "poll_until_terminal",
    "run_command",
]

logger = logging.getLogger("spectraquant.cloud.adapters")

#: States a remote job may be in. ``unknown`` means the platform cannot tell us.
REMOTE_STATES: tuple[str, ...] = ("unknown", "queued", "running", "finished", "failed")

#: States after which polling must stop.
TERMINAL_STATES: frozenset[str] = frozenset({"finished", "failed"})


@dataclass(frozen=True)
class RunStatus:
    """Status of one remote run.

    Attributes:
        run_id: SpectraQuant run id.
        remote_id: platform-side id (``None`` for a human-driven notebook handoff).
        state: one of :data:`REMOTE_STATES`.
        detail: human-readable explanation (never contains credentials).
        raw: raw platform payload, for the record.
    """

    run_id: str
    state: str
    remote_id: str | None = None
    detail: str | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        """True when polling must stop."""
        return self.state in TERMINAL_STATES


@dataclass(frozen=True)
class FetchReport:
    """Result of downloading a run's outputs.

    Attributes:
        run_id: SpectraQuant run id.
        dest_dir: where the files were written.
        files: relative paths of everything downloaded.
        executed_notebook: the executed notebook, when the platform returned one.
        log: captured log file, when one was downloaded.
        ok: whether the download succeeded.
        detail: failure reason when ``ok`` is False.
    """

    run_id: str
    dest_dir: str
    files: list[str] = field(default_factory=list)
    executed_notebook: str | None = None
    log: str | None = None
    ok: bool = True
    detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable report."""
        return {
            "run_id": self.run_id,
            "dest_dir": self.dest_dir,
            "files": list(self.files),
            "executed_notebook": self.executed_notebook,
            "log": self.log,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one external command invocation."""

    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """True when the command exited zero."""
        return self.returncode == 0

    @property
    def combined(self) -> str:
        """stdout followed by stderr, credential-redacted."""
        return redact(f"{self.stdout}{self.stderr}")

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable record (argv is kept, output is redacted)."""
        return {
            "argv": [redact(item) for item in self.argv],
            "returncode": self.returncode,
            "output": self.combined,
        }


#: A command runner: ``(argv, env, cwd) -> CommandResult``. Injected in tests (no network).
CommandRunner = Callable[[Sequence[str], Mapping[str, str] | None, Path | None], CommandResult]


def run_command(
    argv: Sequence[str],
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    *,
    timeout_s: float = 600.0,
) -> CommandResult:
    """Run a command, capturing output; never raises for a non-zero exit.

    Args:
        argv: argument vector (no shell).
        env: environment for the child process (credentials are passed through, never logged).
        cwd: working directory.
        timeout_s: hard timeout; a timeout is reported as exit code 124.

    Returns:
        A :class:`CommandResult` with redacted output on read.
    """
    command = [str(item) for item in argv]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=None if env is None else dict(env),
            cwd=None if cwd is None else str(cwd),
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv=command,
            returncode=124,
            stdout=_as_text(exc.stdout),
            stderr=f"timeout after {timeout_s}s",
        )
    except OSError as exc:
        return CommandResult(argv=command, returncode=127, stderr=f"{type(exc).__name__}: {exc}")
    return CommandResult(
        argv=command,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):  # pragma: no cover - text mode returns str
        return value.decode("utf-8", errors="replace")
    return str(value)


@dataclass(frozen=True)
class PollPolicy:
    """Bounded exponential backoff policy for :func:`poll_until_terminal`.

    Attributes:
        max_attempts: hard cap on status queries (the polling never blocks indefinitely).
        base_delay_s: delay before the second query.
        max_delay_s: cap on the delay between queries.
    """

    max_attempts: int = 20
    base_delay_s: float = 5.0
    max_delay_s: float = 120.0

    def delay_for(self, attempt: int) -> float:
        """Delay in seconds before query ``attempt`` (1-based, after the first)."""
        if attempt <= 1:
            return 0.0
        return float(min(self.base_delay_s * (2 ** (attempt - 2)), self.max_delay_s))

    def max_total_wait_s(self) -> float:
        """Documented upper bound on the total sleep time of a full poll cycle."""
        return float(sum(self.delay_for(attempt) for attempt in range(1, self.max_attempts + 1)))


@runtime_checkable
class PlatformAdapter(Protocol):
    """The four operations every platform adapter must provide.

    ``submit`` MUST persist the remote run id before returning, so an interruption between submit
    and poll cannot orphan a running job.
    """

    name: str

    def submit(self, spec: Any, notebook_path: str) -> str:
        """Submit (or hand off) the notebook; return the remote run id (or the handoff id)."""
        ...

    def status(self, run_id: str) -> RunStatus:
        """Return the current status using bounded-retry polling."""
        ...

    def fetch(self, run_id: str, dest_dir: str) -> FetchReport:
        """Download logs, artifacts and the executed notebook into ``dest_dir``."""
        ...

    def can_resume(self, run_id: str) -> bool:
        """True when a persisted remote id exists and the run may be resumed without re-submitting."""
        ...


def poll_until_terminal(
    adapter: PlatformAdapter,
    run_id: str,
    *,
    policy: PollPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_status: Callable[[RunStatus], None] | None = None,
) -> RunStatus:
    """Poll ``adapter`` until the run reaches a terminal state or the attempt budget is exhausted.

    Args:
        adapter: the platform adapter.
        run_id: SpectraQuant run id.
        policy: backoff policy (defaults to :class:`PollPolicy`, at most ~10 minutes of sleeping).
        sleep: injectable sleeper (tests pass a no-op).
        on_status: optional callback invoked with every observed status.

    Returns:
        The last observed :class:`RunStatus` (state ``"unknown"`` when the budget ran out).
    """
    chosen = policy if policy is not None else PollPolicy()
    status = RunStatus(run_id=run_id, state="unknown", detail="not polled")
    for attempt in range(1, chosen.max_attempts + 1):
        delay = chosen.delay_for(attempt)
        if delay:
            sleep(delay)
        status = adapter.status(run_id)
        if on_status is not None:
            on_status(status)
        logger.info(
            "poll run_id=%s attempt=%d/%d state=%s",
            run_id,
            attempt,
            chosen.max_attempts,
            status.state,
        )
        if status.terminal:
            return status
    return status
