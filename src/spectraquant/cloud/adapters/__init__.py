"""Platform adapters (``design-cloud-adapter.md`` §1, §4).

Each adapter implements :class:`spectraquant.cloud.adapters.base.PlatformAdapter`. The honest
capability of each platform is visible in the class docstrings: Kaggle is fully programmatic,
consumer Colab is a human handoff, Colab Enterprise is paid and guarded, and ``local_cpu`` is the
Tier-0 workstation.
"""

from __future__ import annotations

from spectraquant.cloud.adapters.base import (
    CommandResult,
    CommandRunner,
    FetchReport,
    PlatformAdapter,
    PollPolicy,
    RunStatus,
    poll_until_terminal,
    run_command,
)
from spectraquant.cloud.adapters.colab_enterprise import ColabEnterpriseAdapter
from spectraquant.cloud.adapters.colab_notebook import ColabNotebookAdapter
from spectraquant.cloud.adapters.kaggle import KaggleAdapter
from spectraquant.cloud.adapters.local_cpu import LocalCpuAdapter

__all__ = [
    "ADAPTERS",
    "ColabEnterpriseAdapter",
    "ColabNotebookAdapter",
    "CommandResult",
    "CommandRunner",
    "FetchReport",
    "KaggleAdapter",
    "LocalCpuAdapter",
    "PlatformAdapter",
    "PollPolicy",
    "RunStatus",
    "adapter_for",
    "poll_until_terminal",
    "run_command",
]

#: Platform name -> adapter class.
ADAPTERS: dict[str, type] = {
    "colab": ColabNotebookAdapter,
    "kaggle": KaggleAdapter,
    "colab_enterprise": ColabEnterpriseAdapter,
    "local_cpu": LocalCpuAdapter,
}


def adapter_for(platform: str, **kwargs: object) -> PlatformAdapter:
    """Return the adapter for ``platform``.

    Args:
        platform: one of the keys of :data:`ADAPTERS`.
        **kwargs: forwarded to the adapter constructor (registry, env, runner, ...).

    Returns:
        The adapter instance.

    Raises:
        ValueError: the platform is unknown.
    """
    adapter_class = ADAPTERS.get(platform)
    if adapter_class is None:
        raise ValueError(f"unknown platform {platform!r}; expected one of {sorted(ADAPTERS)}")
    return adapter_class(**kwargs)  # type: ignore[no-any-return]
