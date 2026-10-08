"""Git provenance for run manifests.

The manifest must record the exact commit SHA and a dirty-state flag (AGENTS.md section 9.6), but
manifest writing must also work when the code is not inside a git checkout (sdist, Docker build
context, CI archive). Every function here therefore degrades to ``None`` instead of raising.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from spectraquant.paths import repo_root

__all__ = ["GitInfo", "git_info"]

_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class GitInfo:
    """Snapshot of the git state used to produce a run manifest.

    Attributes:
        commit: full 40-character SHA of ``HEAD``, or ``None`` when unavailable.
        short_commit: 12-character abbreviation, or ``None``.
        branch: checked-out branch name (or ``None`` when detached/unavailable).
        describe: ``git describe --tags --always --dirty`` output, or ``None``.
        dirty: ``True`` when the working tree has uncommitted changes, ``None`` when unknown.
        error: human-readable reason the information is missing, ``None`` on success.
    """

    commit: str | None
    short_commit: str | None
    branch: str | None
    describe: str | None
    dirty: bool | None
    error: str | None = None

    @property
    def is_available(self) -> bool:
        """True when a commit SHA was resolved."""
        return self.commit is not None


def _run_git(args: list[str], cwd: Path) -> tuple[bool, str]:
    """Run ``git`` capturing stdout; return ``(ok, stdout)`` and never raise."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # git missing, timeout, permission
        return False, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout).strip()
    return True, completed.stdout.strip()


def git_info(root: Path | str | None = None) -> GitInfo:
    """Collect commit SHA, branch, describe string and dirty flag.

    Args:
        root: repository directory to inspect. Defaults to :func:`spectraquant.paths.repo_root`.

    Returns:
        A :class:`GitInfo`; all fields are ``None`` (with ``error`` set) outside a git checkout or
        when the ``git`` executable is unavailable.
    """
    cwd = Path(root) if root is not None else repo_root()

    ok, out = _run_git(["rev-parse", "--show-toplevel"], cwd)
    if not ok:
        return GitInfo(None, None, None, None, None, error=f"not a git checkout: {out}")
    top = Path(out)

    ok, head = _run_git(["rev-parse", "HEAD"], cwd)
    if not ok:
        return GitInfo(None, None, None, None, None, error=f"no commit yet: {head}")

    ok_branch, branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    ok_desc, describe = _run_git(["describe", "--tags", "--always", "--dirty"], cwd)
    ok_status, status = _run_git(["status", "--porcelain", "--untracked-files=no"], top)

    return GitInfo(
        commit=head,
        short_commit=head[:12],
        branch=branch if ok_branch and branch != "HEAD" else None,
        describe=describe if ok_desc else None,
        dirty=(bool(status) if ok_status else None),
        error=None,
    )
