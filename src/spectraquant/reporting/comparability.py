"""Equal-memory enforcement for compression comparisons (AGENTS.md section 4.5).

AGENTS.md section 4.5: *"Comparisons are only valid at equal memory; every compression comparison
MUST also report an equal-memory counterpart."* This module is the mechanism behind that sentence:
it turns two run manifests into a verdict, and it refuses the comparison when the stored-memory
figures differ by more than a predeclared tolerance.

Three measurement classes meet here and are never mixed inside one claim (AGENTS.md section 5):

* ``compression.accounted_bytes`` -- **class 1**, analytical: shapes and bit widths plus the
  planned scales, zero points, group metadata, padding and alignment.
* ``compression.measured_bytes`` -- **class 3**, measured: the bytes the frozen
  ``compression.serializer`` invocation actually wrote.
* ``compression.measured_bits_per_param`` -- class 3 as well (``8 * measured_bytes / params``).

Rules enforced here (all of them observable, none of them prose):

1. A run must *declare* which figure it offers, via ``compression.bytes_source``. A manifest that
   carries byte figures without the label is refused rather than guessed at.
2. Both runs must offer the **same** source. A class-3 arm is never compared against a class-1
   arm: that is exactly the "silent pass" this gate exists to prevent.
3. A class-3 comparison additionally requires the **same frozen serializer invocation** on both
   runs, because the byte count of our container depends on the invocation (container overhead).
4. The difference must fit the tolerance: an explicit one, else the tolerance both manifests
   predeclare. With neither, the comparison is refused.

Testing without ``measured_bytes`` (a class-1-only arm) is therefore still possible -- but only
with ``bytes_source: "accounted"`` and only against another accounted arm, and the verdict says so
(``measurement_class == 1``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from spectraquant.reporting.manifests import (
    RunManifest,
    load_schema,
    validate_manifest_file,
)

__all__ = [
    "CLASS_OF_SOURCE",
    "BytesSource",
    "ComparisonVerdict",
    "UnequalMemoryComparison",
    "assert_equal_memory",
    "compare_manifest_files",
    "load_comparable_manifests",
    "tolerance_to_bytes",
]

BytesSource = Literal["accounted", "measured"]

#: Measurement class (AGENTS.md section 5) of each bytes source.
CLASS_OF_SOURCE: dict[BytesSource, int] = {"accounted": 1, "measured": 3}


class UnequalMemoryComparison(ValueError):
    """Raised when two runs may not be compared at equal memory.

    Carries the figures the decision was made from, so a caller can report *why* the comparison
    was refused without re-deriving them:

    Attributes:
        reason: short human-readable reason.
        run_a: run id of the reference run (or the single offending run).
        run_b: run id of the candidate run, when the failure is about a pair.
        bytes_a: comparable byte count of ``run_a`` (``None`` when never resolved).
        bytes_b: comparable byte count of ``run_b`` (``None`` when never resolved).
        difference_bytes: ``|bytes_a - bytes_b|`` (``None`` when either is unknown).
        tolerance_bytes: the absolute tolerance that was applied (``None`` when not reached).
        bytes_source: which byte figure the comparison was made on.
    """

    def __init__(
        self,
        reason: str,
        *,
        run_a: str,
        run_b: str | None = None,
        bytes_a: int | None = None,
        bytes_b: int | None = None,
        tolerance_bytes: int | None = None,
        bytes_source: str | None = None,
    ) -> None:
        self.reason = reason
        self.run_a = run_a
        self.run_b = run_b
        self.bytes_a = bytes_a
        self.bytes_b = bytes_b
        self.tolerance_bytes = tolerance_bytes
        self.bytes_source = bytes_source
        self.difference_bytes = (
            None if bytes_a is None or bytes_b is None else abs(bytes_a - bytes_b)
        )

        context = [f"run={run_a!r}"] if run_b is None else [f"run_a={run_a!r}", f"run_b={run_b!r}"]
        if bytes_a is not None:
            context.append(f"bytes_a={bytes_a}")
        if bytes_b is not None:
            context.append(f"bytes_b={bytes_b}")
        if self.difference_bytes is not None:
            context.append(f"difference_bytes={self.difference_bytes}")
        if tolerance_bytes is not None:
            context.append(f"tolerance_bytes={tolerance_bytes}")
        if bytes_source is not None:
            context.append(f"bytes_source={bytes_source!r}")

        super().__init__(f"unequal-memory comparison rejected: {reason} ({', '.join(context)})")


@dataclass(frozen=True)
class ComparisonVerdict:
    """The record of a passing equal-memory comparison (AGENTS.md section 4.5)."""

    run_a: str
    run_b: str
    bytes_a: int
    bytes_b: int
    tolerance: int | float
    tolerance_bytes: int
    bytes_source: BytesSource

    @property
    def difference_bytes(self) -> int:
        """Absolute byte difference between the two arms."""
        return abs(self.bytes_a - self.bytes_b)

    @property
    def measurement_class(self) -> int:
        """Class of the compared number: 3 for measured, 1 for analytical (never mixed)."""
        return CLASS_OF_SOURCE[self.bytes_source]

    @property
    def equal(self) -> bool:
        """True when the difference is inside the applied tolerance."""
        return self.difference_bytes <= self.tolerance_bytes

    def summary(self) -> str:
        """One-line human-readable verdict (used by the CLI).

        No square brackets: the CLI prints this through Rich, which would read them as markup.
        """
        kind = "measured, class 3" if self.bytes_source == "measured" else "accounted, class 1"
        return (
            f"EQUAL MEMORY ({kind}): {self.run_a} = {self.bytes_a} B vs {self.run_b} = "
            f"{self.bytes_b} B; |difference| = {self.difference_bytes} B <= "
            f"{self.tolerance_bytes} B (declared tolerance {self.tolerance!r})"
        )

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serializable verdict."""
        return {
            "equal": self.equal,
            "run_a": self.run_a,
            "run_b": self.run_b,
            "bytes_a": self.bytes_a,
            "bytes_b": self.bytes_b,
            "difference_bytes": self.difference_bytes,
            "tolerance": self.tolerance,
            "tolerance_bytes": self.tolerance_bytes,
            "bytes_source": self.bytes_source,
            "measurement_class": self.measurement_class,
        }


def tolerance_to_bytes(value: int | float, reference_bytes: int) -> int:
    """Convert a declared tolerance into an absolute byte budget.

    An integral value is an absolute byte count; a non-integral value in ``(0, 1)`` is a fraction
    of ``reference_bytes`` (rounded up). Anything else is a schema violation and is rejected.

    Args:
        value: the declared tolerance (``compression.measured_bytes_tolerance`` or a CLI flag).
        reference_bytes: the byte total the fraction is taken of (the larger of the two arms).

    Returns:
        The tolerance in bytes.

    Raises:
        ValueError: when ``value`` is neither a byte count ``>= 0`` nor a fraction in ``(0, 1)``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"tolerance must be a number, got {type(value).__name__}")
    if reference_bytes < 0:
        raise ValueError(f"reference_bytes must be >= 0, got {reference_bytes}")
    if isinstance(value, int) or float(value).is_integer():
        if value < 0:
            raise ValueError(f"tolerance must be >= 0 bytes, got {value!r}")
        return int(value)
    if 0.0 < float(value) < 1.0:
        return math.ceil(float(value) * reference_bytes)
    raise ValueError(
        "tolerance must be an absolute byte count >= 0 or a fraction strictly inside (0, 1); "
        f"got {value!r}"
    )


def _declared_bytes(manifest: RunManifest) -> tuple[int, BytesSource]:
    """Return ``(bytes, source)`` for one manifest, refusing an undeclared source."""
    compression = manifest.compression
    source = compression.bytes_source
    if source is None:
        raise UnequalMemoryComparison(
            "compression.bytes_source is not declared, so it is unknown whether the run offers "
            "a class-1 analytical figure or a class-3 measured one (declare 'accounted' or "
            "'measured' in compression.bytes_source)",
            run_a=manifest.run_id,
        )
    if source == "measured":
        if compression.measured_bytes is None:
            raise UnequalMemoryComparison(
                "compression.bytes_source is 'measured' but compression.measured_bytes is null",
                run_a=manifest.run_id,
                bytes_source=source,
            )
        return compression.measured_bytes, source
    if compression.accounted_bytes is None:
        raise UnequalMemoryComparison(
            "compression.bytes_source is 'accounted' but compression.accounted_bytes is null",
            run_a=manifest.run_id,
            bytes_source=source,
        )
    if compression.measured_bytes is not None:
        raise UnequalMemoryComparison(
            "compression.bytes_source is 'accounted' while compression.measured_bytes is "
            f"{compression.measured_bytes}; a class-3 measurement may not be downgraded to the "
            "class-1 estimate",
            run_a=manifest.run_id,
            bytes_source=source,
        )
    return compression.accounted_bytes, source


def _declared_tolerance(run_a: RunManifest, run_b: RunManifest) -> int | float:
    """Return the tolerance the two manifests predeclare (they must agree)."""
    tolerance_a = run_a.compression.measured_bytes_tolerance
    tolerance_b = run_b.compression.measured_bytes_tolerance
    if tolerance_a is None and tolerance_b is None:
        raise UnequalMemoryComparison(
            "no tolerance: pass an explicit tolerance or declare "
            "compression.measured_bytes_tolerance in the manifests",
            run_a=run_a.run_id,
            run_b=run_b.run_id,
        )
    if (
        tolerance_a is not None
        and tolerance_b is not None
        and float(tolerance_a) != float(tolerance_b)
    ):
        raise UnequalMemoryComparison(
            f"the two manifests declare different tolerances ({tolerance_a!r} vs "
            f"{tolerance_b!r}); pass an explicit tolerance to override",
            run_a=run_a.run_id,
            run_b=run_b.run_id,
        )
    declared = tolerance_a if tolerance_a is not None else tolerance_b
    assert declared is not None  # narrowed by the branch above
    return declared


def assert_equal_memory(
    run_a: RunManifest,
    run_b: RunManifest,
    tolerance_bytes: int | None = None,
) -> ComparisonVerdict:
    """Gate two runs on equal stored memory (AGENTS.md section 4.5).

    Args:
        run_a: reference run manifest.
        run_b: candidate run manifest.
        tolerance_bytes: absolute byte tolerance; overrides any tolerance the manifests declare.
            When ``None``, the manifests' ``compression.measured_bytes_tolerance`` is used (they
            must agree), and a comparison with no tolerance anywhere is refused.

    Returns:
        The :class:`ComparisonVerdict` of the accepted comparison.

    Raises:
        UnequalMemoryComparison: the two arms are not comparable at equal memory -- different
            byte sources, different frozen serializers, an undeclared source, or a difference
            larger than the tolerance.
    """
    bytes_a, source_a = _declared_bytes(run_a)
    bytes_b, source_b = _declared_bytes(run_b)

    if source_a != source_b:
        raise UnequalMemoryComparison(
            f"the two runs use different byte sources ({source_a!r} vs {source_b!r}): a class-"
            f"{CLASS_OF_SOURCE[source_a]} figure may not be compared against a class-"
            f"{CLASS_OF_SOURCE[source_b]} figure",
            run_a=run_a.run_id,
            run_b=run_b.run_id,
            bytes_a=bytes_a,
            bytes_b=bytes_b,
        )

    if source_a == "measured":
        serializer_a = run_a.compression.serializer
        serializer_b = run_b.compression.serializer
        if not serializer_a or not serializer_b:
            raise UnequalMemoryComparison(
                "a class-3 comparison requires compression.serializer (the frozen invocation "
                f"that produced the byte count) on both runs ({serializer_a!r} vs {serializer_b!r})",
                run_a=run_a.run_id,
                run_b=run_b.run_id,
                bytes_a=bytes_a,
                bytes_b=bytes_b,
                bytes_source=source_a,
            )
        if serializer_a != serializer_b:
            raise UnequalMemoryComparison(
                f"different frozen serializers ({serializer_a!r} vs {serializer_b!r}); byte "
                "counts produced by different invocations are not comparable",
                run_a=run_a.run_id,
                run_b=run_b.run_id,
                bytes_a=bytes_a,
                bytes_b=bytes_b,
                bytes_source=source_a,
            )

    declared = tolerance_bytes if tolerance_bytes is not None else _declared_tolerance(run_a, run_b)
    tolerance_bytes_value = tolerance_to_bytes(declared, max(bytes_a, bytes_b))
    difference = abs(bytes_a - bytes_b)
    if difference > tolerance_bytes_value:
        raise UnequalMemoryComparison(
            f"the two runs differ by {difference} bytes, more than the "
            f"{tolerance_bytes_value}-byte tolerance",
            run_a=run_a.run_id,
            run_b=run_b.run_id,
            bytes_a=bytes_a,
            bytes_b=bytes_b,
            tolerance_bytes=tolerance_bytes_value,
            bytes_source=source_a,
        )

    return ComparisonVerdict(
        run_a=run_a.run_id,
        run_b=run_b.run_id,
        bytes_a=bytes_a,
        bytes_b=bytes_b,
        tolerance=declared,
        tolerance_bytes=tolerance_bytes_value,
        bytes_source=source_a,
    )


def load_comparable_manifests(
    path_a: Path | str, path_b: Path | str
) -> tuple[RunManifest, RunManifest]:
    """Read two manifests and validate both against the schema before any byte is compared.

    Raises:
        ManifestValidationError: a file is missing, is not JSON, or violates the published schema
            (including the equal-memory byte-field rules).
    """
    schema = load_schema()
    document_a = validate_manifest_file(path_a, schema=schema)
    document_b = validate_manifest_file(path_b, schema=schema)
    return RunManifest.model_validate(document_a), RunManifest.model_validate(document_b)


def compare_manifest_files(
    path_a: Path | str,
    path_b: Path | str,
    tolerance_bytes: int | None = None,
) -> ComparisonVerdict:
    """Validate two manifest files against the schema, then gate them at equal memory."""
    run_a, run_b = load_comparable_manifests(path_a, path_b)
    return assert_equal_memory(run_a, run_b, tolerance_bytes)
