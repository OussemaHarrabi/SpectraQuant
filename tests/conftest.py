"""Session-wide guards for the process-global numerical state torch exposes.

``spectraquant.training.seeding.seed_everything`` calls ``torch.set_num_threads`` and
``torch.use_deterministic_algorithms``. Both are **process-global**: a test that seeds the process
leaks those settings into every test that runs after it, so a test's result can depend on which test
happened to run before it.

That is not cosmetic. The T2-a oracle gate (``tests/unit/test_m3_exactness.py``) compares our NF2
LoftQ residual spectrum against a capture taken on this machine, at the frozen ``1e-6`` bound. On the
fixture's rank-16 subspace the measured relative error moves with the BLAS thread count:

===========================  ==========
threads                      rel. error
===========================  ==========
1                            ``1.0824e-06`` (violates)
2                            ``1.0858e-06`` (violates)
4                            ``6.3347e-07``
8 (this workstation default)  ``8.4295e-07``
===========================  ==========

so a leaked ``threads=1`` from ``tests/unit/test_seeding.py`` flips a **frozen** gate's verdict on
test order alone (amendment A-0021). The fix is to restore the state rather than to relax the bound:
the fixture below snapshots the thread count and the deterministic-algorithms flag before every test
and restores them afterwards.

Random-number-generator streams are deliberately *not* restored here: each test seeds its own
generators, and silently rewinding a stream around every test would hide a test that forgets to.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_process_global_numerical_state() -> object:
    """Restore the process-global thread count and determinism flag around every test."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
