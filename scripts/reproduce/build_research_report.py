# /// script
# requires-python = ">=3.11"
# dependencies = ["reportlab>=4.2,<5"]
# ///
"""Build the publication PDF for the completed SpectraQuant bounded study."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1791763200")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
OUTPUT = ROOT / "output" / "pdf" / "SpectraQuant_Research_Report_2026.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#0B6E8E")
TEAL = colors.HexColor("#087F5B")
RED = colors.HexColor("#C92A4A")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE_BLUE = colors.HexColor("#EAF6FA")
PALE_GREEN = colors.HexColor("#EAF8F2")
PALE_RED = colors.HexColor("#FFF0F3")
GRID = colors.HexColor("#D9E2EC")
WHITE = colors.white


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(data: list[list[object]], widths: list[float], *, header: bool = True) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands: list[tuple] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [WHITE, colors.HexColor("#F7FAFC")]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    t.setStyle(TableStyle(commands))
    return t


def figure(
    path: Path, width: float, caption: str, styles: dict[str, ParagraphStyle]
) -> list[object]:
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return [
        image,
        Spacer(1, 2 * mm),
        P(caption, styles["caption"]),
        Spacer(1, 4 * mm),
    ]


def section(title: str, styles: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Spacer(1, 3 * mm),
        P(title, styles["section"]),
        HRFlowable(width="100%", thickness=0.7, color=BLUE, spaceBefore=1, spaceAfter=7),
    ]


def callout(
    heading: str,
    body: str,
    styles: dict[str, ParagraphStyle],
    *,
    background: colors.Color = PALE_BLUE,
    accent: colors.Color = BLUE,
) -> Table:
    payload = [
        [P(heading, styles["callout_title"])],
        [P(body, styles["callout_body"])],
    ]
    box = Table(payload, colWidths=[171 * mm], hAlign="LEFT")
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
            ]
        )
    )
    return box


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=29,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=MUTED,
            spaceAfter=13,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=BLUE,
        ),
        "abstract_title": ParagraphStyle(
            "AbstractTitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=NAVY,
            spaceAfter=4,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=13.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=BLUE,
            keepWithNext=True,
            spaceBefore=4,
            spaceAfter=0,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=NAVY,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.5,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.8,
            textColor=INK,
            leftIndent=12,
            firstLineIndent=-7,
            bulletIndent=0,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10.5,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "callout_title": ParagraphStyle(
            "CalloutTitle",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=NAVY,
        ),
        "callout_body": ParagraphStyle(
            "CalloutBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.5,
            textColor=INK,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.4,
            textColor=INK,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            leftIndent=12,
            firstLineIndent=-12,
            textColor=INK,
            spaceAfter=3,
        ),
    }


def first_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, A4[1] - 9 * mm, A4[0], 9 * mm, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(
        A4[0] - 18 * mm, 12 * mm, "Public technical report - bounded, non-peer-reviewed evidence"
    )
    canvas.restoreState()


def later_pages(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, A4[1] - 14 * mm, A4[0] - 18 * mm, A4[1] - 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, A4[1] - 11 * mm, "SPECTRAQUANT TECHNICAL REPORT")
    right = "Oussema Harrabi - October 2026"
    canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 11 * mm, right)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
    canvas.restoreState()


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="SpectraQuant: From Budget-Aware Allocation Failure to Verified Clipping Transfer",
        author="Oussema Harrabi",
        subject="Reproducible transformer-compression research report",
        creator="SpectraQuant reproducible report builder",
    )
    story: list[object] = []

    story += [
        Spacer(1, 11 * mm),
        P("SPECTRAQUANT", styles["meta"]),
        Spacer(1, 3 * mm),
        P(
            "From Budget-Aware Allocation Failure<br/>to Verified Clipping Transfer",
            styles["title"],
        ),
        P(
            "A reproducible study of local compression objectives and whole-model language quality",
            styles["subtitle"],
        ),
        HRFlowable(width="100%", thickness=2.1, color=BLUE, spaceBefore=0, spaceAfter=10),
        P("Oussema Harrabi &nbsp;&nbsp;|&nbsp;&nbsp; Independent Research Project", styles["meta"]),
        P(
            "12 October 2026 &nbsp;&nbsp;|&nbsp;&nbsp; "
            '<link href="https://github.com/OussemaHarrabi/SpectraQuant">'
            "github.com/OussemaHarrabi/SpectraQuant</link>",
            styles["small"],
        ),
        Spacer(1, 10 * mm),
    ]

    abstract = (
        "SpectraQuant asks whether inexpensive local compression objectives can preserve "
        "autoregressive language-model quality under a measured equal-byte checkpoint budget. "
        "The original sensitivity-aware low-rank/low-bit allocator failed against uniform INT4. "
        "A deployment-matched correction and a joint-loss diagnosis strengthened that negative "
        "result. A simpler clipping control then improved one fresh C4 holdout, motivating a "
        "frozen two-model/two-domain transfer study. The primary control improved only 2/4 cells "
        "and failed its consistency gate. Predeclared activation-weighted controls reduced C4 "
        "perplexity by 7.39-7.70% on SmolLM2-360M and 5.75-6.09% on Pythia-410M at identical "
        "checkpoint bytes, while regressing on Pythia PG19. All 52 expected Phase-4 artifacts "
        "and 42 checkpoint files were authenticated. The supported outcome is a bounded C4 "
        "improvement with a documented transfer failure, not a universal compression claim."
    )
    abstract_box = Table(
        [[P("Abstract", styles["abstract_title"])], [P(abstract, styles["abstract"])]],
        colWidths=[171 * mm],
    )
    abstract_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 11),
            ]
        )
    )
    story += [abstract_box, Spacer(1, 7 * mm)]
    story.append(
        callout(
            "Headline result",
            "Activation-weighted INT4 controls lower reserved C4 perplexity on both tested "
            "architectures at identical storage. The Pythia PG19 regression defines the boundary "
            "of that result.",
            styles,
            background=PALE_GREEN,
            accent=TEAL,
        )
    )
    story += [
        Spacer(1, 6 * mm),
        P(
            "<b>Keywords:</b> transformer compression; post-training quantization; INT4; "
            "clipping; sensitivity; low-rank approximation; reproducibility; negative results",
            styles["small"],
        ),
        PageBreak(),
    ]

    story += section("1. Introduction", styles)
    story += [
        P(
            "Compression methods often optimize local quantities: weight reconstruction error, "
            "activation-weighted error, Hessian surrogates, spectral statistics, or layer-wise "
            "calibration loss. The operational outcome is global. Every changed layer participates "
            "in one autoregressive computation, and locally favorable changes can interact poorly "
            "when composed.",
            styles["body"],
        ),
        callout(
            "Research question",
            "Can cheap local compression scores choose a low-bit representation that preserves "
            "end-to-end language quality at the same measured checkpoint budget?",
            styles,
        ),
        Spacer(1, 4 * mm),
        P(
            "The project treats a failed original method as the start of the investigation. "
            "It first tests a joint low-rank/low-bit allocator, corrects evaluation to the actual "
            "packed operator, diagnoses marginal-versus-joint loss, simplifies the intervention "
            "to conventional clipping, and finally freezes a transfer study before holdout access.",
            styles["body"],
        ),
    ]
    story += figure(
        FIGURES / "spectraquant-research-storyline.png",
        171 * mm,
        "<b>Figure 1.</b> The coherent progression. Each later phase tests an assumption exposed "
        "by the preceding failure.",
        styles,
    )

    story += section("2. Related work and research gap", styles)
    story += [
        P(
            "GPTQ, AWQ and SmoothQuant show that calibration statistics and second-order or "
            "activation-aware objectives can materially change low-bit behavior. HAWQ-style "
            "methods allocate precision from sensitivity estimates. LoftQ and LR-QAT combine "
            "low-rank structure with quantization-aware optimization. These results motivate the "
            "original SpectraQuant hypothesis, but they do not remove the need for equal-storage, "
            "deployment-matched comparisons.",
            styles["body"],
        ),
        P(
            "SpectraQuant does not claim a new state-of-the-art PTQ algorithm. Its strongest "
            "contribution is an evaluation result: local proxy improvements and lower weight "
            "reconstruction error can fail to predict whole-model perplexity, while clipping "
            "benefits can depend on model, corpus and grouping orientation.",
            styles["body"],
        ),
    ]

    story += section("3. Problem formulation", styles)
    story += [
        P(
            "For layer l with weight W(l), the original allocator chooses rank r(l) and bit width "
            "b(l) to minimize the sum of estimated local errors subject to a frozen byte budget:",
            styles["body"],
        ),
        callout(
            "Budgeted allocation objective",
            "min sum_l E_hat(l, r(l), b(l)) &nbsp;&nbsp; subject to &nbsp;&nbsp; "
            "sum_l bytes(l, r(l), b(l)) <= B",
            styles,
        ),
        Spacer(1, 3 * mm),
        P(
            "The deployed residual representation is Q(W) + Q(B)Q(A), with rank zero representing "
            "plain quantization. Later clipping experiments keep INT4/group32 fixed and select a "
            "clipping scale from a predeclared grid. The diagonal activation proxy weights each "
            "input feature's squared weight error by its calibration second moment. It omits "
            "cross-feature covariance and is not measured output error.",
            styles["body"],
        ),
    ]

    story += section("4. Research and evidence architecture", styles)
    story += [
        P(
            "Three rules govern all headline comparisons: compare quality only at equal measured "
            "bytes; reload the representation actually written to disk; and freeze models, data, "
            "arms and gates before fresh holdout access. Protocol changes are append-only. Failed "
            "gates remain visible.",
            styles["body"],
        )
    ]
    story += figure(
        FIGURES / "spectraquant-evidence-architecture.png",
        171 * mm,
        "<b>Figure 2.</b> Frozen design and typed run specifications feed cloud execution; packed "
        "checkpoints are reloaded for evaluation; manifests, hashes and CI gate publication.",
        styles,
    )

    story.append(PageBreak())
    story += section("5. Experimental progression", styles)
    progression = [
        [
            P("Stage", styles["small"]),
            P("Question", styles["small"]),
            P("Measured answer", styles["small"]),
        ],
        [
            P("Original pilot", styles["small"]),
            P("Beat uniform INT4 at equal bytes?", styles["small"]),
            P("No: 23.1698/23.3131 PPL vs 18.6193.", styles["small"]),
        ],
        [
            P("LR-QAT direction", styles["small"]),
            P("Reproduce the expected direction?", styles["small"]),
            P("21.8% gap closure; below the 25% gate.", styles["small"]),
        ],
        [
            P("Phase 2", styles["small"]),
            P("Does packed/reloaded evaluation rescue allocation?", styles["small"]),
            P("No: six arms are 5.52-14.75% worse.", styles["small"]),
        ],
        [
            P("Phase 3 diagnosis", styles["small"]),
            P("Do marginal gains predict joint loss?", styles["small"]),
            P("No: all candidates overpredict and regress.", styles["small"]),
        ],
        [
            P("Phase 3 pivot", styles["small"]),
            P("Can fixed clipping help once?", styles["small"]),
            P("Yes: 28.0056 to 27.4656 PPL (-1.9279%).", styles["small"]),
        ],
        [
            P("Phase 4", styles["small"]),
            P("Does the control transfer?", styles["small"]),
            P("Mixed: primary 2/4; strong C4 secondary result.", styles["small"]),
        ],
    ]
    story += [table(progression, [29 * mm, 62 * mm, 80 * mm]), Spacer(1, 5 * mm)]
    story += figure(
        FIGURES / "fixed-clipping-first-holdout.png",
        125 * mm,
        "<b>Figure 3.</b> The first fixed-clipping C4 holdout comparison. This bounded result "
        "triggered a frozen transfer study instead of immediate generalization.",
        styles,
    )

    story += section("6. Frozen Phase-4 protocol", styles)
    bullets = [
        "<b>Models.</b> SmolLM2-360M and Pythia-410M at immutable revisions.",
        "<b>Corpora.</b> Reserved C4 validation documents and PG19 test-book prefixes.",
        "<b>Calibration.</b> The first 200 C4 training documents; no holdout NLL tuning.",
        "<b>Arms.</b> Seven INT4/group32 checkpoints per model: RTN/MSE on axis0/axis1 and "
        "three activation-weighted axis1 calibration subsets.",
        "<b>Evaluation.</b> Exactly 51,200 predicted tokens per corpus/model with document "
        "boundaries preserved.",
        "<b>Order.</b> All checkpoints are frozen before holdout loading.",
        "<b>Uncertainty.</b> Document-cluster bootstrap intervals describe corpus uncertainty; "
        "calibration subsets are not training seeds.",
    ]
    for item in bullets:
        story.append(P("&bull; " + item, styles["bullet"]))

    primary = [
        ["Model", "Corpus", "RTN PPL", "MSE PPL", "Reduction"],
        ["SmolLM2-360M", "C4", "21.8140", "22.3901", "-2.6408%"],
        ["SmolLM2-360M", "PG19", "16.6734", "16.8819", "-1.2501%"],
        ["Pythia-410M", "C4", "32.3251", "31.3777", "+2.9311%"],
        ["Pythia-410M", "PG19", "27.9244", "27.0500", "+3.1313%"],
    ]
    story.append(
        KeepTogether(
            [
                *section("7. Primary transfer result", styles),
                table(primary, [46 * mm, 30 * mm, 30 * mm, 30 * mm, 35 * mm]),
                Spacer(1, 4 * mm),
                callout(
                    "Primary verdict: FAIL",
                    "The geometric-mean perplexity reduction is 0.5752%, above the 0.5% magnitude "
                    "threshold. Only 2/4 cells improve, below the required 3/4 consistency gate.",
                    styles,
                    background=PALE_RED,
                    accent=RED,
                ),
                Spacer(1, 5 * mm),
            ]
        )
    )
    story += figure(
        FIGURES / "clipping-transfer-primary.png",
        142 * mm,
        "<b>Figure 4.</b> Primary MSE-clipping transfer cells and descriptive intervals.",
        styles,
    )

    story += section("8. Secondary activation-weighted controls", styles)
    secondary = [
        ["Model", "Corpus", "PPL reduction range", "Full checkpoint"],
        ["SmolLM2-360M", "C4", "+7.3882 to +7.7028%", "271,561,031 B"],
        ["SmolLM2-360M", "PG19", "+5.8906 to +6.1849%", "271,561,031 B"],
        ["Pythia-410M", "C4", "+5.7462 to +6.0939%", "302,615,137 B"],
        ["Pythia-410M", "PG19", "-2.5608 to -1.0864%", "302,615,137 B"],
    ]
    story += [
        table(secondary, [43 * mm, 28 * mm, 55 * mm, 45 * mm]),
        Spacer(1, 4 * mm),
        P(
            "The control improves reserved C4 on both architectures and SmolLM2 PG19, but "
            "regresses on Pythia PG19. Calibration-domain sensitivity is plausible, but the "
            "study does not isolate it causally because it does not add book-domain calibration. "
            "The supported claim is a measured C4 improvement at equal bytes.",
            styles["body"],
        ),
    ]
    story += figure(
        FIGURES / "clipping-transfer-controls.png",
        154 * mm,
        "<b>Figure 5.</b> All predeclared clipping controls. Secondary outcomes do not rescue "
        "the failed primary gate.",
        styles,
    )

    story += section("9. Why the local objectives failed", styles)
    story += [
        P(
            "Weight-MSE clipping reduced aggregate weight squared error by 9.98% on SmolLM2 "
            "axis0 and 10.19% on axis1, yet worsened perplexity on both corpora. Pythia axis1 "
            "provides another counterexample: 10.69% lower weight error accompanied 7.06% worse "
            "PG19 perplexity. Lower reconstruction error is therefore insufficient evidence for "
            "better language quality.",
            styles["body"],
        )
    ]
    story += figure(
        FIGURES / "clipping-objective-vs-quality.png",
        150 * mm,
        "<b>Figure 6.</b> Improvements in local compression objectives versus measured "
        "whole-model perplexity.",
        styles,
    )
    story += [
        P(
            "The allocation diagnosis exposes the same problem. Layer effects measured in "
            "isolation were not additive after simultaneous deployment. Residual paths, "
            "normalization, activation shifts and error propagation couple the layers.",
            styles["body"],
        )
    ]
    story += figure(
        FIGURES / "allocation-prediction-vs-joint.png",
        143 * mm,
        "<b>Figure 7.</b> Marginal allocation predictions overstate jointly realized gains.",
        styles,
    )

    story += section("10. Storage, provenance and compute", styles)
    storage = [
        ["Model", "Target payload", "Full checkpoint"],
        ["SmolLM2-360M", "176,947,200 B", "271,561,031 B"],
        ["Pythia-410M", "198,844,416 B", "302,615,137 B"],
    ]
    story += [
        table(storage, [58 * mm, 56 * mm, 57 * mm]),
        Spacer(1, 4 * mm),
        P(
            "Each model's seven Phase-4 arms match these byte counts exactly. The collection "
            "process verified all 52 expected artifacts and 42 dynamically selected checkpoint "
            "files, including checksums, source revision, model/data fingerprints, token counts, "
            "and the requirement that selection precedes holdout access.",
            styles["body"],
        ),
        P(
            "Two private Kaggle runs completed on their first submission. Their runner intervals "
            "sum to approximately 1.511184 visible T4 device-hours, including installation but "
            "excluding export/platform overhead. This is execution telemetry, not exact account "
            "billing.",
            styles["body"],
        ),
    ]

    story += section("11. Reproducibility", styles)
    repro = [
        ["Layer", "Recorded evidence"],
        ["Protocol", "Frozen ADRs, append-only amendments and acceptance gates"],
        ["Execution", "Typed run specs and generated Kaggle/Colab notebooks"],
        ["Environment", "Python, PyTorch, Transformers, CUDA and hardware metadata"],
        ["Storage", "Payload/layout/full-checkpoint sizes plus checksums"],
        ["Results", "Original metrics, source scores and derived attestations"],
        ["Publication", "Artifact-generated tables/figures with recorded hashes"],
        ["Quality gates", "Formatting, types, CPU tests, docs and security CI"],
    ]
    story += [
        table(repro, [38 * mm, 133 * mm]),
        Spacer(1, 4 * mm),
        P(
            "The Phase-4 scientific source was frozen at commit "
            "<font name='Courier'>71a1ec11cea6be90a6ab49356e81d980f4667784</font>. "
            "A new scientific claim requires a new run ID, new protocol and untouched data; "
            "collection scripts do not silently resubmit compute.",
            styles["body"],
        ),
    ]

    story += section("12. Limitations", styles)
    limitations = [
        "Calibration subsets are not independent model-training seeds.",
        "Only two sub-billion-parameter model families and two reserved corpora are tested.",
        "Perplexity uses dense reconstruction, not a deployed low-bit GPU kernel.",
        "The diagonal activation proxy omits feature covariance.",
        "Full matched GPTQ/AWQ arms were not run in Phase 4.",
        "Exposed holdouts cannot support further tuning; the next study needs fresh data.",
    ]
    for item in limitations:
        story.append(P("&bull; " + item, styles["bullet"]))

    story += section("13. Conclusions", styles)
    story += [
        P(
            "SpectraQuant did not validate its original joint low-rank/low-bit allocator. "
            "Deployment-matched evaluation and joint-loss analysis made that negative result "
            "stronger. A simpler clipping pivot produced useful equal-byte gains, while the "
            "frozen transfer study showed that those gains depend on architecture, corpus and "
            "grouping.",
            styles["body"],
        ),
        callout(
            "What the study establishes",
            "Activation-weighted INT4 clipping is a strong C4 control in the tested setting, "
            "with 5.75-7.70% lower perplexity across two architectures at identical storage. "
            "Its Pythia PG19 regression rules out a universal or domain-robust claim.",
            styles,
            background=PALE_GREEN,
            accent=TEAL,
        ),
        Spacer(1, 4 * mm),
        P(
            "The broader research-engineering lesson is procedural: evaluate the deployed "
            "representation, freeze the next test, preserve negative cells, authenticate "
            "artifacts, and bound every claim by the evidence that actually ran.",
            styles["body"],
        ),
    ]

    story += section("References", styles)
    references = [
        "[1] Frantar et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arXiv:2210.17323, 2022.",
        "[2] Lin et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. arXiv:2306.00978, 2023.",
        "[3] Xiao et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arXiv:2211.10438, 2022.",
        "[4] Dong et al. HAWQ-V2: Hessian Aware Trace-Weighted Quantization of Neural Networks. NeurIPS, 2020.",
        "[5] Li et al. LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models. arXiv:2310.08659, 2023.",
        "[6] Bondarenko et al. LR-QAT: Low-Rank Quantization-Aware Training for LLMs. arXiv:2406.06385, 2024.",
        "[7] Biderman et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML, 2023.",
        "[8] Raffel et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR, 2020.",
        "[9] Rae et al. Compressive Transformers for Long-Range Sequence Modelling. arXiv:1911.05507, 2019.",
    ]
    for ref in references:
        story.append(P(ref, styles["reference"]))

    story += [
        Spacer(1, 7 * mm),
        HRFlowable(width="100%", thickness=0.7, color=GRID, spaceBefore=0, spaceAfter=5),
        P(
            "<b>Artifact availability.</b> Code, protocols, generated notebooks, bounded evidence, "
            "figures and LaTeX source are available at "
            '<link href="https://github.com/OussemaHarrabi/SpectraQuant">'
            "github.com/OussemaHarrabi/SpectraQuant</link>.",
            styles["small"],
        ),
    ]

    doc.build(
        story,
        onFirstPage=first_page,
        onLaterPages=later_pages,
        canvasmaker=partial(pdf_canvas.Canvas, invariant=1),
    )
    return OUTPUT


if __name__ == "__main__":
    result = build()
    print(result)
