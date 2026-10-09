"""Measurement-integrity scanner for prose claims (Milestone 9 gate).

The gate this module automates is stated in ``AGENTS.md`` sections 4 and 5:

* **4.3** fake quantization MUST NOT be reported as low-bit storage or accelerated inference;
* **4.4** no latency claim without a real supported kernel and a controlled protocol;
* **4.9** no state-of-the-art claims from narrow or incomparable experiments;
* **5** every number carries one measurement class, and classes MUST NOT be mixed inside a claim.

Those rules are enforced in code for *manifests* by
:mod:`spectraquant.reporting.comparability` (equal memory) and the measurement-class blocks of the
manifest schema. They were not enforced for *prose* -- README lines, reports and result tables --
which is where an overclaim actually escapes. This module closes that gap: it scans a set of
markdown/text documents for forbidden claim patterns and returns, per match, a verdict.

The hard part is that this repository legitimately *quotes* the rules it is checked against: the
measurement taxonomy has a forbidden-claims table, ``AGENTS.md`` has prohibition bullet lists, the
claim audit quotes headline claims. A naive grep for ``1.8x faster`` would flag the very table that
forbids it. The scanner therefore distinguishes a **violation** (an unmarked claim) from a **quote**
(a match inside an explicitly negated or marked context). The heuristic is:

1. a match is a **quote** when the match's *context* (the line, the wrapped continuation of the line
   when it is an unpunctuated continuation, its enclosing list/block label, and its table header)
   contains a negation/prohibition marker (``never``, ``must not``, ``forbidden``, ``no``, ``not``,
   ``without``, ...) or an attribution/quotation marker (``reported``, ``paper's claim``,
   ``the word``, ...);
2. a match in a table whose **header** says ``forbidden`` / ``never`` / ``must not`` / ``wrong`` is a
   quote;
3. a match under a **block label** such as ``**Forbidden:**`` is a quote;
4. otherwise the per-pattern document gate applies: a matching performance figure is *allowed* when
   the same document also declares a real-kernel backend, and a novelty phrase is *allowed* when the
   document carries the frozen verdict wording. Those matches are counted as *skipped*, with the
   reason, never as violations.

The heuristic has known, documented failure modes (see ``docs/results/claim-guard-report.md``):

* **false negatives** -- any stray ``not``/``no`` in the context suppresses a real violation, and a
  violation hidden in a fenced code block is not scanned;
* **false positives** -- none are currently known against the committed repository, but a genuinely
  new claim that happens to sit under a ``Forbidden:`` label would be treated as a quote;
* it **cannot** judge whether a correctly-labelled number is the *right* number. A wrong value that
  already carries the right class label, the right backend statement and no forbidden wording passes.

Everything here is deterministic: no clock, no randomness, stable ordering by (document, line,
pattern). The scan counts are measurement class **1** (analytical -- derived from committed text,
nothing executed).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from spectraquant.paths import repo_root

__all__ = [
    "CLAIM_PATTERNS",
    "ClaimPattern",
    "DocumentReport",
    "Finding",
    "ScanReport",
    "default_document_paths",
    "scan_document",
    "scan_paths",
    "scan_text",
]

Verdict = Literal["quote", "violation"]

#: Categories of documents every default scan covers, relative to the repository root.
DEFAULT_GLOBS: tuple[str, ...] = ("README.md", "AGENTS.md", "docs/**/*.md", "reports/**/*.md")

# --------------------------------------------------------------------------------------
# Shared vocabulary (the regexes the patterns are built from)
# --------------------------------------------------------------------------------------

#: A statement that a real low-bit kernel executed *our own* serialized artifact. Its presence in a
#: document lifts the latency/throughput prohibition of ``AGENTS.md`` section 4.4 for that document.
BACKEND_RE = re.compile(
    r"MatMulNBits"
    r"|MatMulInteger"
    r"|IntxWeightOnly"
    r"|torchao intx"
    r"|ONNX Runtime"
    r"|onnxruntime"
    r"|CPUExecutionProvider"
    r"|class[ -]?4-CPU"
    r"|4-CPU"
    r"|real (?:CPU )?(?:supported )?(?:low-bit )?kernel"
    r"|low-bit kernel"
    r"|kernel-backed",
    re.IGNORECASE,
)

#: The frozen novelty verdicts (``docs/results/verification/m1-novelty-review.md``). A document that
#: states them qualifies any novelty language it contains.
FROZEN_VERDICT_RE = re.compile(
    r"differentiated \((?:narrow, empirical|weak[–-]moderate)"
    r"|previously-known"
    r"|candidate contribution",
    re.IGNORECASE,
)

#: Prohibition / negation markers. Their presence in the match context makes the match a quote.
NEGATION_RE = re.compile(
    r"\bnever\b"
    r"|\bmust not\b"
    r"|\bmustn't\b"
    r"|\bforbid(?:s|den)?\b"
    r"|\bprohibit(?:s|ed)?\b"
    r"|\bbanned\b"
    r"|\bno\b"
    r"|\bnot\b"
    r"|\bcannot\b"
    r"|\bcan't\b"
    r"|\bmay not\b"
    r"|\bwithout\b"
    r"|\bnothing\b"
    r"|\bnone\b"
    r"|\bneither\b"
    r"|\bnor\b"
    r"|\bavoid\b"
    r"|\brefuse[sd]?\b"
    r"|\bunavailable\b"
    r"|\bout of scope\b"
    r"|\bNOT RUN\b"
    r"|\bnot measured\b"
    r"|\bwrong\b"
    r"|\bincorrect\b"
    r"|\bmistaken\b"
    r"|\berroneous\b",
    re.IGNORECASE,
)

#: Attribution / quotation markers. A claim quoted from a paper, or the *word* rather than the
#: claim, is not a claim by this repository.
QUOTATION_RE = re.compile(
    r"paper'?s? claim"
    r"|\breported\b"
    r"|\breporting\b"
    r"|\bquoted?\b|\bquoting\b"
    r"|\bcite[ds]?\b"
    r"|\bliterature (?:figure|value|number)\b"
    r"|\bfrom the (?:paper|literature)\b"
    r"|\buses the word\b"
    r"|\bthe word\b"
    r"|`new`",
    re.IGNORECASE,
)

#: Markers of a class-1 analytical figure that never was measured.
ESTIMATE_RE = re.compile(
    r"\bINFERENCE\b"
    r"|\banalytical\b"
    r"|\bestimat(?:e|ed|es|ion)\b"
    r"|\bapproximate(?:ly)?\b"
    r"|\btheoretical\b"
    r"|\bclass[ -]1\b",
    re.IGNORECASE,
)

#: Markers of a host/environment capability constant, which is not a claim about a compressed model.
ENVIRONMENT_RE = re.compile(
    r"\bworkstation\b"
    r"|\bthis (?:machine|host)\b"
    r"|\benvironment fact\b"
    r"|\bhost constant\b"
    r"|\bnumpy\b"
    r"|\bBLAS\b"
    r"|\bbest of \d",
    re.IGNORECASE,
)

#: Deployment context: a performance figure is only a *deployment property claim* when the same
#: context also talks about a compressed artifact, a low-bit kernel, or an inference workload.
DEPLOYMENT_CUE_RE = re.compile(
    r"\bint[2348]\b"
    r"|\bint4\b|\bint8\b|\bint3\b"
    r"|\d-bit\b"
    r"|low-bit|low bit"
    r"|quantiz|quantis"
    r"|fake[- ]?quant"
    r"|compress(?:ed|ion)"
    r"|fp32 baseline|vs\.? fp32|than fp32"
    r"|our (?:own )?container"
    r"|weight-only"
    r"|prefill|\bdecode\b"
    r"|kernel"
    r"|\bmodels?\b"
    r"|\binference\b"
    r"|\bdeploy(?:ed|ment)?\b"
    r"|per-token|per-forward"
    r"|tok(?:en)?s?/s",
    re.IGNORECASE,
)

#: A latency / throughput / speed figure. Bare seconds are deliberately matched only next to an
#: inference word, so that a training wall-clock (``0.5 s``) is not mistaken for a latency claim.
_INFERENCE_WORD = (
    r"(?:latency|per[- ]token|per[- ]forward|\bforward\b|inference|\bdecode\b|\bprefill\b)"
)
PERFORMANCE_FIGURE_RE = re.compile(
    r"\d[\d.,\s]*[×x]\s*(?:faster|speed-?up)"
    r"|\d[\d.,\s]*\s*(?:ms|ns)\b"
    rf"|{_INFERENCE_WORD}[^.\n]{{0,30}}\d[\d.,\s]*\s*s\b"
    rf"|\d[\d.,\s]*\s*s\b[^.\n]{{0,30}}{_INFERENCE_WORD}"
    r"|\d[\d.,\s]*\s*(?:tok(?:en)?s?/s|GFLOP/s|TFLOP/s|GB/s|req(?:uest)?s?/s)"
    r"|\b(?:latency|throughput|speedup|speed-?up)\b[^.\n]{0,30}\d",
    re.IGNORECASE,
)

#: Fake-quantization wording.
FAKE_QUANT_RE = re.compile(
    r"\bfake[- ]?quant\w*"
    r"|\bfake quantiz\w*"
    r"|simulated quantiz\w*"
    r"|straight-through estimator",
    re.IGNORECASE,
)

#: A storage / size / compression figure that fake quantization must not stand in for.
STORAGE_RE = re.compile(
    r"\d[\d\s.,]*[×x]\s*smaller"
    r"|\bsmaller\b"
    r"|low-bit storage"
    r"|stored bytes"
    r"|checkpoint size"
    r"|compression ratio"
    r"|serialized size"
    r"|bits\s*/\s*param|bits per param"
    r"|total_bytes|payload_bytes",
    re.IGNORECASE,
)

#: An analytical figure attached to a measured assertion through ``as`` / ``presented as``.
_EST = r"(?:analytical|theoretical|estimat(?:e|ed)|approximate|class[ -]1)"
ANALYTICAL_AS_MEASURED_RE = re.compile(
    rf"{_EST}\b[^.\n]{{0,70}}\b(?:presented )?as\b[^.\n]{{0,25}}\bmeasured\b",
    re.IGNORECASE,
)

#: A class-4-GPU / class-5 marker (a measurement that cannot be asserted from local execution).
GPU_SERVICE_RE = re.compile(
    r"class[ -]?4[ -]?GPU"
    r"|\b4-GPU\b"
    r"|class[ -]?5\b"
    r"|end-to-end service"
    r"|service[- ]level latency"
    r"|service latency",
    re.IGNORECASE,
)

#: The assertion side of a GPU/service claim: a measured verb or a magnitude with a unit.
GPU_SERVICE_ASSERT_RE = re.compile(
    r"\bmeasured\b"
    r"|\bachieved\b"
    r"|\bwe ran\b"
    r"|presented as|reported as|claimed as"
    r"|\d[\d.,\s]*\s*(?:ms|ns|tok(?:en)?s?/s|req(?:uest)?s?/s)",
    re.IGNORECASE,
)

#: A local/CPU subject (one side of a forbidden CPU-vs-published-GPU comparison). Hyphenated
#: substrate labels (``LOCAL-FIXTURE``, ``CLOUD-GPU``) are excluded: they name a substrate, not a
#: hardware comparison.
LOCAL_SUBJECT_RE = re.compile(
    r"(?<!-)\bCPU\b(?!-)"
    r"|(?<![A-Z-])4-CPU\b"
    r"|\blocal(?:ly)?\b(?!-)"
    r"|\bworkstation\b"
    r"|\bthis (?:machine|host)\b",
    re.IGNORECASE,
)

#: A *published* GPU subject (the other side of a forbidden comparison): a paper, a named GPU, or
#: explicitly published numbers. A bare "GPU" is not enough -- comparing local CPU-hours with cloud
#: GPU-hours is a cost statement, not a comparison of our numbers with published ones.
PUBLISHED_SUBJECT_RE = re.compile(
    r"\bpublished\b"
    r"|\bliterature\b"
    r"|\bpaper'?s?\b"
    r"|\bA100\b|\bH100\b|\bT4\b|\bRTX\b",
    re.IGNORECASE,
)

#: A comparison verb.
COMPARISON_RE = re.compile(
    r"\bcompare[sd]?\b|\bcomparison\b|\bcomparable\b|\bvs\.?\b|\bversus\b|\bagainst\b"
    r"|faster than|\bmatches\b",
    re.IGNORECASE,
)

#: An explicit measurement-class label. A sentence that names its classes is not mixing them.
CLASS_LABEL_RE = re.compile(r"\bclass[ -]?[1-5](?:[ -]?(?:CPU|GPU))?\b", re.IGNORECASE)

#: An unqualified novelty phrase.
NOVELTY_RE = re.compile(
    r"\bnovel\b"
    r"|\bfirst to\b"
    r"|state[ -]of[ -]the[ -]art"
    r"|\bunprecedented\b"
    r"|\bSOTA\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimPattern:
    """One forbidden claim pattern plus the co-occurrence and document-gate rules it needs.

    Attributes:
        name: stable identifier used in the report and in tests.
        description: one-line statement of what the pattern forbids.
        trigger: the forbidden phrasing itself.
        requires: further regexes that must all appear in the match context for the trigger to
            count (e.g. a deployment cue, or the two sides of a comparison).
        doc_gate: ``"real_kernel_backend"`` -- a match is *allowed* (skipped) when the document
            declares a real-kernel backend; ``"frozen_verdict"`` -- allowed when the document
            carries the frozen verdict wording. ``None`` means no document-level allowance.
        skip_on_estimate: skip (not a violation) when the context marks the figure an estimate.
        skip_on_environment: skip when the context marks the figure a host/environment constant.
        skip_on_class_label: skip when the context labels its measurement classes explicitly, so
            the sentence is separating classes rather than mixing them.
    """

    name: str
    description: str
    trigger: re.Pattern[str]
    requires: tuple[re.Pattern[str], ...] = ()
    doc_gate: str | None = None
    skip_on_estimate: bool = False
    skip_on_environment: bool = False
    skip_on_class_label: bool = False


#: The forbidden-claim catalogue. Order is stable and is the report's ordering.
CLAIM_PATTERNS: tuple[ClaimPattern, ...] = (
    ClaimPattern(
        name="latency_without_kernel_backend",
        description=(
            "a latency/throughput/speed figure for a compressed artifact in a document that never "
            "names a real low-bit kernel backend (AGENTS.md section 4.4)"
        ),
        trigger=PERFORMANCE_FIGURE_RE,
        requires=(DEPLOYMENT_CUE_RE,),
        doc_gate="real_kernel_backend",
        skip_on_estimate=True,
        skip_on_environment=True,
    ),
    ClaimPattern(
        name="fake_quant_as_storage",
        description=(
            "a fake-quantization result presented as low-bit storage, a size, or measured bytes "
            "(AGENTS.md section 4.3)"
        ),
        trigger=FAKE_QUANT_RE,
        requires=(STORAGE_RE,),
        skip_on_class_label=True,
    ),
    ClaimPattern(
        name="analytical_as_measured",
        description=(
            "a class-1 analytical figure presented as a measured value (AGENTS.md section 5)"
        ),
        trigger=ANALYTICAL_AS_MEASURED_RE,
    ),
    ClaimPattern(
        name="gpu_service_asserted_locally",
        description=(
            "a class-4-GPU or class-5 number asserted from local execution (AGENTS.md sections 2.4, 5)"
        ),
        trigger=GPU_SERVICE_RE,
        requires=(GPU_SERVICE_ASSERT_RE,),
    ),
    ClaimPattern(
        name="cpu_vs_published_gpu",
        description=(
            "a comparison of our CPU numbers with published GPU numbers (AGENTS.md sections 4.4, 4.9)"
        ),
        trigger=COMPARISON_RE,
        requires=(LOCAL_SUBJECT_RE, PUBLISHED_SUBJECT_RE),
    ),
    ClaimPattern(
        name="unqualified_novelty",
        description=(
            "an unqualified novelty claim not backed by the frozen verdict wording "
            "(AGENTS.md section 4.9)"
        ),
        trigger=NOVELTY_RE,
        doc_gate="frozen_verdict",
    ),
)


# --------------------------------------------------------------------------------------
# Result document model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One pattern match and the verdict the heuristic assigned to it."""

    document: str
    line: int
    pattern: str
    verdict: Verdict
    context: str
    reason: str

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serializable finding."""
        return {
            "document": self.document,
            "line": self.line,
            "pattern": self.pattern,
            "verdict": self.verdict,
            "context": self.context,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DocumentReport:
    """Findings and skip counts for one scanned document."""

    path: str
    findings: tuple[Finding, ...] = ()
    skipped: Mapping[str, int] = field(default_factory=dict)

    @property
    def quotes(self) -> tuple[Finding, ...]:
        """Matches classified as quotes of the rules, not violations."""
        return tuple(f for f in self.findings if f.verdict == "quote")

    @property
    def violations(self) -> tuple[Finding, ...]:
        """Matches classified as unmarked violations."""
        return tuple(f for f in self.findings if f.verdict == "violation")

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serializable document report."""
        return {
            "path": self.path,
            "quotes": len(self.quotes),
            "violations": len(self.violations),
            "skipped": dict(sorted(self.skipped.items())),
            "findings": [f.to_json_dict() for f in self.findings],
        }


@dataclass(frozen=True)
class ScanReport:
    """The scan of a set of documents."""

    documents: tuple[DocumentReport, ...] = ()

    @property
    def findings(self) -> tuple[Finding, ...]:
        """All findings across all documents, in scan order."""
        return tuple(f for d in self.documents for f in d.findings)

    @property
    def quotes(self) -> tuple[Finding, ...]:
        """All quoted matches."""
        return tuple(f for f in self.findings if f.verdict == "quote")

    @property
    def violations(self) -> tuple[Finding, ...]:
        """All violations (the gate fails when this is non-empty)."""
        return tuple(f for f in self.findings if f.verdict == "violation")

    @property
    def skipped(self) -> Mapping[str, int]:
        """Skip counts by reason, summed across documents."""
        counter: Counter[str] = Counter()
        for document in self.documents:
            counter.update(document.skipped)
        return dict(sorted(counter.items()))

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serializable scan report."""
        return {
            "documents_scanned": len(self.documents),
            "documents": [d.to_json_dict() for d in self.documents],
            "totals": {
                "quotes": len(self.quotes),
                "violations": len(self.violations),
                "skipped": dict(self.skipped),
            },
        }

    def summary(self) -> str:
        """One-line human-readable verdict (used by the CLI and the report)."""
        return (
            f"{len(self.documents)} documents scanned: {len(self.violations)} violation(s), "
            f"{len(self.quotes)} quote(s), {sum(self.skipped.values())} skipped"
        )


# --------------------------------------------------------------------------------------
# Context construction
# --------------------------------------------------------------------------------------

_TABLE_ROW_RE = re.compile(r"^\s*\|")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[*+-]|\d+\.)\s")
_HEADING_RE = re.compile(r"^\s*#")
_PUNCT_END_RE = re.compile(r"[.!?:;]\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_HTML_COMMENT_RE = re.compile(r"^\s*<!--")


def _first_table_row_index(lines: Sequence[str], index: int) -> int | None:
    """Return the index of the top row of the table that contains ``index``, or ``None``."""
    if not _TABLE_ROW_RE.match(lines[index]):
        return None
    top = index
    while top > 0 and _TABLE_ROW_RE.match(lines[top - 1]):
        top -= 1
    return top


def _table_header(lines: Sequence[str], index: int) -> str:
    """Return the header row of the table containing ``index`` (empty when there is none)."""
    top = _first_table_row_index(lines, index)
    if top is None or top + 1 >= len(lines) or not _TABLE_SEPARATOR_RE.match(lines[top + 1]):
        return ""
    return lines[top].strip()


def _block_label(lines: Sequence[str], index: int) -> str:
    """Return the nearest governing label above ``index``.

    Labels are headings and lines ending in ``:``; list items, table rows and blank lines are
    skipped so that a bullet under ``**Forbidden:**`` resolves to its label.
    """
    for offset in range(1, 13):
        position = index - offset
        if position < 0:
            return ""
        stripped = lines[position].strip()
        if (
            not stripped
            or _LIST_ITEM_RE.match(lines[position])
            or _TABLE_ROW_RE.match(lines[position])
        ):
            continue
        if _HEADING_RE.match(lines[position]):
            return stripped
        if stripped.endswith(":") or re.fullmatch(r"\*\*[^*]+\*\*:?", stripped):
            return stripped
        return ""
    return ""


def _context(lines: Sequence[str], index: int) -> tuple[str, bool, str]:
    """Return ``(context, is_table_header, header)`` for the line at ``index``.

    The context is the line itself, its wrapped continuation (the previous line only when it has no
    terminal punctuation and the current line starts like a continuation), its block label and, for
    table rows, the table header.
    """
    parts = [lines[index].strip()]
    if index > 0:
        previous = lines[index - 1].strip()
        current = lines[index].strip()
        if (
            previous
            and not _PUNCT_END_RE.search(previous)
            and not _HEADING_RE.match(current)
            and not _TABLE_ROW_RE.match(current)
            and not _LIST_ITEM_RE.match(current)
        ):
            # Only the final sentence of the wrapped line: an earlier clause is a different claim.
            parts.append(re.split(r"(?<=[.!?:;])\s+", previous)[-1])
    label = _block_label(lines, index)
    if label:
        parts.append(label)
    header = _table_header(lines, index)
    if header and header not in parts:
        parts.append(header)
    context = "\n".join(parts)
    top = _first_table_row_index(lines, index)
    is_table_header = top is not None and top == index
    return context, is_table_header, header


def _is_forbidden_table_header(header: str) -> bool:
    """True when a table header marks its body as forbidden examples."""
    return NEGATION_RE.search(header) is not None


# --------------------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------------------


def _classify(
    pattern: ClaimPattern,
    context: str,
    *,
    is_table_header: bool,
    header: str,
    has_backend: bool,
    has_frozen_verdict: bool,
) -> tuple[Verdict, str] | str:
    """Return ``(verdict, reason)`` for a match, or a skip-reason string."""
    if NEGATION_RE.search(context):
        return "quote", "negated or prohibited context"
    if QUOTATION_RE.search(context):
        return "quote", "attributed or quoted wording"
    if header and not is_table_header and _is_forbidden_table_header(header):
        return "quote", "inside a forbidden-claims table"
    if is_table_header and _is_forbidden_table_header(context):
        return "quote", "forbidden-claims table header"
    if pattern.skip_on_estimate and ESTIMATE_RE.search(context):
        return "analytical_estimate_not_measured"
    if pattern.skip_on_environment and ENVIRONMENT_RE.search(context):
        return "environment_constant"
    if pattern.skip_on_class_label and CLASS_LABEL_RE.search(context):
        return "classes_explicitly_labelled"
    if pattern.doc_gate == "real_kernel_backend" and has_backend:
        return "document_declares_real_kernel_backend"
    if pattern.doc_gate == "frozen_verdict" and has_frozen_verdict:
        return "frozen_verdict_wording_present"
    return "violation", "unmarked forbidden claim"


def scan_text(text: str, document: str = "<text>") -> DocumentReport:
    """Scan one document's text and return its :class:`DocumentReport`.

    Args:
        text: the document contents.
        document: the document name used in findings (a path, or ``<text>`` for in-memory input).

    Returns:
        The findings (violations and quotes) plus per-reason skip counts for this document.
    """
    lines = text.splitlines()
    has_backend = BACKEND_RE.search(text) is not None
    has_frozen_verdict = FROZEN_VERDICT_RE.search(text) is not None

    findings: list[Finding] = []
    skipped: Counter[str] = Counter()
    in_code_fence = False

    for index, line in enumerate(lines):
        if _CODE_FENCE_RE.match(line):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence or not line.strip() or _HTML_COMMENT_RE.match(line):
            continue

        context, is_table_header, header = _context(lines, index)
        for pattern in CLAIM_PATTERNS:
            if not pattern.trigger.search(context):
                continue
            if any(requirement.search(context) is None for requirement in pattern.requires):
                continue
            outcome = _classify(
                pattern,
                context,
                is_table_header=is_table_header,
                header=header,
                has_backend=has_backend,
                has_frozen_verdict=has_frozen_verdict,
            )
            if isinstance(outcome, str):
                skipped[outcome] += 1
                continue
            verdict, reason = outcome
            findings.append(
                Finding(
                    document=document,
                    line=index + 1,
                    pattern=pattern.name,
                    verdict=verdict,
                    context=context,
                    reason=reason,
                )
            )

    findings.sort(key=lambda f: (f.document, f.line, f.pattern, f.context))
    return DocumentReport(path=document, findings=tuple(findings), skipped=dict(skipped))


def scan_document(path: Path | str) -> DocumentReport:
    """Read one document (UTF-8) and scan it."""
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8", errors="replace")
    try:
        name = resolved.relative_to(repo_root()).as_posix()
    except ValueError:
        name = resolved.as_posix()
    return scan_text(text, name)


def default_document_paths(root: Path | str | None = None) -> tuple[Path, ...]:
    """Return the default scan set: ``README.md``, ``AGENTS.md``, ``docs/**/*.md``, ``reports/**/*.md``.

    Args:
        root: repository root; defaults to :func:`spectraquant.paths.repo_root`.

    Returns:
        Sorted, de-duplicated document paths that exist.
    """
    base = Path(root) if root is not None else repo_root()
    found: set[Path] = set()
    for glob in DEFAULT_GLOBS:
        found.update(p for p in base.glob(glob) if p.is_file())
    return tuple(sorted(found))


def scan_paths(paths: Iterable[Path | str]) -> ScanReport:
    """Scan a set of files (and ``*.md`` files under directories) in stable order.

    Args:
        paths: files or directories; directories are expanded to their ``*.md`` descendants.

    Returns:
        The combined :class:`ScanReport`.
    """
    documents: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            documents.extend(p for p in path.glob("**/*.md") if p.is_file())
        else:
            documents.append(path)
    unique = sorted(set(documents), key=lambda p: p.as_posix())
    return ScanReport(documents=tuple(scan_document(p) for p in unique))
