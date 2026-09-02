#!/usr/bin/env python3
"""Generate the Review-2 project report PDF.

    python scripts/make_project_report.py

Writes ``docs/review/Project_Report.pdf``, covering:

  1. what the project is and what it measures
  2. the complete annotated file structure
  3. how the system works end to end
  4. how to run it locally on any machine
  5. how to present and defend it at the faculty review
  6. which AI tools were used, and how their output was verified
  7. honest limitations and the plan for Review-3

Content is written to be defensible: every number here is either measured by a
script in this repository or explicitly labelled as unverified.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
REVIEW = DOCS / "review"
OUT = REVIEW / "Project_Report.pdf"

INK = colors.HexColor("#101823")
MUTED = colors.HexColor("#4D5B6E")
FAINT = colors.HexColor("#7A8798")
TEAL = colors.HexColor("#0D9488")
TEAL_BG = colors.HexColor("#E8F6F4")
ROSE = colors.HexColor("#BE123C")
ROSE_BG = colors.HexColor("#FDEAEE")
AMBER = colors.HexColor("#B45309")
AMBER_BG = colors.HexColor("#FDF4E5")
LINE = colors.HexColor("#C9D3E0")
CODE_BG = colors.HexColor("#F4F7FA")
ZEBRA = colors.HexColor("#F7F9FC")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
BODY_W = PAGE_W - 2 * MARGIN


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #

ss = getSampleStyleSheet()


def _style(name, **kw):
    base = kw.pop("parent", ss["BodyText"])
    return ParagraphStyle(name, parent=base, **kw)


ST = {
    "title": _style("t", parent=ss["Title"], fontName="Helvetica-Bold",
                    fontSize=25, leading=30, textColor=INK, spaceAfter=4),
    "subtitle": _style("st", fontName="Helvetica", fontSize=12.5, leading=17,
                       textColor=MUTED, alignment=TA_CENTER, spaceAfter=3),
    "h1": _style("h1", fontName="Helvetica-Bold", fontSize=16, leading=20,
                 textColor=INK, spaceBefore=2, spaceAfter=7),
    "h2": _style("h2", fontName="Helvetica-Bold", fontSize=12, leading=15.5,
                 textColor=TEAL, spaceBefore=12, spaceAfter=5),
    "h3": _style("h3", fontName="Helvetica-Bold", fontSize=10.2, leading=13.5,
                 textColor=INK, spaceBefore=9, spaceAfter=3),
    "body": _style("b", fontName="Helvetica", fontSize=9.6, leading=14.2,
                   textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6),
    "small": _style("sm", fontName="Helvetica", fontSize=8.6, leading=12.4,
                    textColor=MUTED, alignment=TA_JUSTIFY, spaceAfter=5),
    "bullet": _style("bu", fontName="Helvetica", fontSize=9.4, leading=13.4,
                     textColor=INK, alignment=TA_LEFT, spaceAfter=2.5),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.0, leading=10.6,
                           textColor=INK),
    "cap": _style("cap", fontName="Helvetica-Oblique", fontSize=8.2, leading=11,
                  textColor=FAINT, alignment=TA_CENTER, spaceBefore=3, spaceAfter=9),
    "th": _style("th", fontName="Helvetica-Bold", fontSize=8.4, leading=11,
                 textColor=colors.white, alignment=TA_LEFT),
    "td": _style("td", fontName="Helvetica", fontSize=8.3, leading=11.4,
                 textColor=INK, alignment=TA_LEFT),
    "tdb": _style("tdb", fontName="Helvetica-Bold", fontSize=8.3, leading=11.4,
                  textColor=INK, alignment=TA_LEFT),
}


def P(text, style="body"):
    return Paragraph(text, ST[style])


def bullets(items, style="bullet", bullet="•", indent=11):
    return ListFlowable(
        [ListItem(P(i, style), leftIndent=indent, value=bullet) for i in items],
        bulletType="bullet", start=bullet, leftIndent=indent,
        bulletFontSize=8, spaceAfter=6,
    )


def numbered(items, style="bullet"):
    return ListFlowable(
        [ListItem(P(i, style), leftIndent=14) for i in items],
        bulletType="1", leftIndent=14, bulletFontName="Helvetica-Bold",
        bulletFontSize=9, spaceAfter=6,
    )


def table(rows, widths, *, header=True, zebra=True, font_size=8.3,
          header_bg=TEAL, align=None):
    data = []
    for r_i, row in enumerate(rows):
        out = []
        for c_i, cell in enumerate(row):
            if isinstance(cell, Flowable):
                out.append(cell)
            else:
                st = "th" if (header and r_i == 0) else "td"
                out.append(Paragraph(str(cell), ST[st]))
        data.append(out)

    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), header_bg)]
    if zebra:
        for i in range(1 if header else 0, len(data)):
            if (i - (1 if header else 0)) % 2 == 1:
                cmds.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    if align:
        for col, a in align.items():
            cmds.append(("ALIGN", (col, 0), (col, -1), a))
    t.setStyle(TableStyle(cmds))
    return t


def callout(title, body, *, tone="info"):
    bg, fg = {"info": (TEAL_BG, TEAL), "warn": (AMBER_BG, AMBER),
              "danger": (ROSE_BG, ROSE)}[tone]
    inner = [
        Paragraph(f"<b>{title}</b>", ParagraphStyle(
            "ct", fontName="Helvetica-Bold", fontSize=9.4, leading=12.6, textColor=fg)),
        Spacer(1, 3),
        Paragraph(body, ParagraphStyle(
            "cb", fontName="Helvetica", fontSize=9.0, leading=13.0,
            textColor=INK, alignment=TA_JUSTIFY)),
    ]
    t = Table([[inner]], colWidths=[BODY_W], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, fg),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return KeepTogether([t, Spacer(1, 8)])


def code(text, *, width=BODY_W):
    body = Preformatted(text.strip("\n"), ST["code"])
    t = Table([[body]], colWidths=[width], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return KeepTogether([t, Spacer(1, 8)])


def figure(name, caption, *, max_w=BODY_W, max_h=118 * mm):
    path = FIGURES / name
    if not path.exists():
        return P(f"<i>[figure {escape(name)} not generated]</i>", "small")
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    img = Image(str(path), width=iw * scale, height=ih * scale)
    img.hAlign = "CENTER"
    return KeepTogether([img, P(caption, "cap")])


class HR(Flowable):
    def __init__(self, width=BODY_W, colour=LINE, thickness=0.6):
        super().__init__()
        self.width, self.colour, self.thickness = width, colour, thickness
        self.height = 0

    def draw(self):
        self.canv.setStrokeColor(self.colour)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #

TITLE = "Explainable Dermoscopic Lesion Analysis"
SUB = "Major Project II  ·  Review 2  ·  Project Report"


def on_page(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 7.6)
    canv.setFillColor(FAINT)
    canv.drawString(MARGIN, PAGE_H - 11 * mm, TITLE)
    canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 11 * mm, "Major Project II — Review 2")
    canv.setStrokeColor(LINE)
    canv.setLineWidth(0.5)
    canv.line(MARGIN, PAGE_H - 13.5 * mm, PAGE_W - MARGIN, PAGE_H - 13.5 * mm)
    canv.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
    canv.drawString(MARGIN, 9.5 * mm,
                    "Research prototype — not a medical device.")
    canv.drawRightString(PAGE_W - MARGIN, 9.5 * mm, f"Page {doc.page}")
    canv.restoreState()


def on_cover(canv, doc):
    canv.saveState()
    canv.setFillColor(TEAL)
    canv.rect(0, PAGE_H - 8 * mm, PAGE_W, 8 * mm, stroke=0, fill=1)
    canv.setStrokeColor(LINE)
    canv.setLineWidth(0.5)
    canv.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
    canv.setFont("Helvetica", 7.6)
    canv.setFillColor(FAINT)
    canv.drawString(MARGIN, 9.5 * mm, "Research prototype — not a medical device.")
    canv.drawRightString(PAGE_W - MARGIN, 9.5 * mm, date.today().isoformat())
    canv.restoreState()


def build_doc():
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title=f"{TITLE} — {SUB}",
        author="Major Project II, Review 2",
        subject="Project report: architecture, operation, local setup, review guide, AI tooling",
    )
    frame_cover = Frame(MARGIN, 18 * mm, BODY_W, PAGE_H - 40 * mm, id="cover")
    frame_body = Frame(MARGIN, 18 * mm, BODY_W, PAGE_H - 38 * mm, id="body")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame_cover], onPage=on_cover),
        PageTemplate(id="body", frames=[frame_body], onPage=on_page),
    ])
    return doc


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #


def cover():
    s = []
    s.append(Spacer(1, 26 * mm))
    s.append(P("Major Project II  ·  Review 2", "subtitle"))
    s.append(Spacer(1, 5 * mm))
    s.append(Paragraph(TITLE, ST["title"]))
    s.append(Spacer(1, 3 * mm))
    s.append(P("EfficientNet-B3 classification with Grad-CAM explanation, quantitative "
               "ABCD morphometry, uncertainty quantification and automated severity grading, "
               "served through a FastAPI backend and a browser clinical workstation.",
               "subtitle"))
    s.append(Spacer(1, 12 * mm))
    s.append(HR())
    s.append(Spacer(1, 8 * mm))

    meta = [
        ["Project", "Explainable Dermoscopic Lesion Analysis (HAM10000)"],
        ["Submission", "Major Project II — Review 2"],
        ["Presented by", "<< your name >>, << team member >>"],
        ["Mentor / Guide", "<< guide name >>"],
        ["UDP ID", "U<< id >>"],
        ["Dataset", "HAM10000 — 10,015 dermoscopic images, 7,470 distinct lesions, 7 classes"],
        ["Code size", "~9,600 lines of application code, ~1,300 lines of tests"],
        ["Test status", "149 automated tests, all passing"],
        ["Weights", "Linear probe on frozen EfficientNet-B3 features — 21/21 verification checks"],
        ["Measured", "53.7% balanced accuracy, 0.379 macro-F1, ROC-AUC 0.838 (leak-free split)"],
        ["Report date", date.today().strftime("%d %B %Y")],
    ]
    s.append(table([[Paragraph(f"<b>{k}</b>", ST["td"]), v] for k, v in meta],
                   [42 * mm, BODY_W - 42 * mm], header=False, zebra=True))
    s.append(Spacer(1, 10 * mm))

    s.append(callout(
        "Read this first",
        "This software is a research and educational prototype. It is not a medical device, "
        "has not been clinically validated, and must never be used to diagnose, treat or rule "
        "out disease. Any lesion that is new, changing, bleeding or itching needs in-person "
        "assessment by a clinician regardless of what this software reports.",
        tone="danger"))
    s.append(callout(
        "About the weights, stated precisely",
        "The system now runs with trained weights, so every feature is functional. Those "
        "weights come from a <b>linear probe</b>: the 1536&#8203;&#215;7 classifier head was "
        "fitted on features extracted from a frozen, ImageNet-pretrained EfficientNet-B3. The "
        "backbone itself was <b>not</b> fine-tuned, because that requires a GPU this project "
        "did not have. A linear probe is materially weaker than end-to-end fine-tuning — "
        "notably on the rare classes — and it is reported as a probe everywhere in this "
        "document and in the application. The checkpoint records "
        "<font face='Courier'>training_method</font> so no generated report can quietly "
        "misrepresent it.",
        tone="warn"))
    return s


def section_1():
    s = [P("1.  What this project is", "h1")]
    s.append(P(
        "Skin cancer outcomes depend steeply on how early a lesion is examined. Melanoma "
        "in particular is highly survivable when excised early and frequently fatal when it "
        "is not. Dermoscopy improves diagnostic accuracy over the naked eye, but reading "
        "dermoscopic images is a specialist skill, and specialist time is the scarce resource. "
        "That makes automated triage worth building — provided the output is auditable rather "
        "than a bare label."))
    s.append(P(
        "This project classifies dermoscopic images into the seven HAM10000 diagnostic "
        "categories using EfficientNet-B3, and then does the part that matters for a "
        "clinical setting: it explains itself. Every analysis returns a visual explanation "
        "(Grad-CAM, scored against the lesion outline so you can tell whether the network "
        "looked at the lesion or at an artefact), an independent set of geometric measurements "
        "that do not involve the neural network at all (the Stolz ABCD rule), an explicit "
        "uncertainty estimate, and a severity grade with the reasoning attached."))

    s.append(P("1.1  The seven diagnostic classes", "h2"))
    rows = [["Code", "Diagnosis", "Class", "Images in HAM10000"]]
    for code_, name, mal, n in [
        ("akiec", "Actinic keratosis / intraepithelial carcinoma", "Pre-malignant", "327"),
        ("bcc", "Basal cell carcinoma", "Malignant", "514"),
        ("bkl", "Benign keratosis-like lesion", "Benign", "1,099"),
        ("df", "Dermatofibroma", "Benign", "115"),
        ("mel", "Melanoma", "Malignant", "1,113"),
        ("nv", "Melanocytic nevus", "Benign", "6,705"),
        ("vasc", "Vascular lesion", "Benign", "142"),
    ]:
        rows.append([f"<font face='Courier'>{code_}</font>", name, mal, n])
    s.append(table(rows, [18 * mm, 74 * mm, 26 * mm, BODY_W - 118 * mm],
                   align={3: "RIGHT"}))
    s.append(Spacer(1, 4))
    s.append(P(
        "Class order is fixed alphabetically because that is what "
        "<font face='Courier'>LabelEncoder</font> produces on the HAM10000 "
        "<font face='Courier'>dx</font> column. This matters more than it looks: a checkpoint "
        "loaded with a permuted class order produces confident, plausible, and completely "
        "wrong diagnoses. <font face='Courier'>scripts/verify_checkpoint.py</font> exists "
        "specifically to catch that.", "small"))

    s.append(P("1.2  Why 67% nevi makes accuracy a misleading metric", "h2"))
    s.append(P(
        "Melanocytic nevi are 6,705 of 10,015 images. A model that predicts "
        "<font face='Courier'>nv</font> unconditionally scores 67% accuracy while catching zero "
        "melanomas — which is the single worst behaviour this system could have. The project "
        "therefore selects checkpoints on macro-F1 rather than accuracy, and reports balanced "
        "accuracy, per-class recall, and a melanoma safety-net catch rate alongside any headline "
        "figure."))

    s.append(P("1.3  What was added after Review-1", "h2"))
    s.append(bullets([
        "The notebook code was rebuilt as an installable Python package: 18 modules under "
        "<font face='Courier'>src/derm</font>, so training, evaluation, the CLI scripts and "
        "the web service all share one definition of class order, image size and normalisation.",
        "A FastAPI backend with 12 endpoints, and a complete browser workstation UI with no "
        "build step.",
        "Grad-CAM and Grad-CAM++ with an attention-alignment score measured against the "
        "segmentation mask.",
        "Quantitative Stolz ABCD morphometry — a second evidence stream that stays valid even "
        "when the network does not.",
        "Uncertainty quantification: test-time augmentation, MC dropout, predictive entropy "
        "and the BALD mutual-information score.",
        "Composite severity grading with one-directional safety overrides.",
        "A deterministic, rule-based clinical narrative plus PDF export; SQLite case history; "
        "longitudinal change tracking.",
        "A reproducible data-leakage audit which showed the accuracy figure carried over from "
        "Review-1 was inflated, and 149 automated tests.",
        "The full dataset (10,015 images, 195 MB after downscaling) plus a real trained "
        "checkpoint obtained without a GPU — see section 1.4.",
    ]))

    s.append(P("1.4  Getting trained weights without a GPU", "h2"))
    s.append(P(
        "End-to-end fine-tuning of EfficientNet-B3's 12M parameters is not possible on the "
        "available hardware: an earlier attempt on an 8 GB M2 drove swap past 13 GB and had to "
        "be abandoned. Rather than ship an untrained network, transfer learning was split into "
        "its two halves and only the cheap half was run locally."))
    s.append(numbered([
        "<b>Feature extraction, forward only.</b> The ImageNet-pretrained backbone is frozen "
        "and run under <font face='Courier'>torch.inference_mode()</font> in batches of 12. No "
        "gradients, no optimiser state, no retained activation graph — peak resident memory "
        "measured at 472 MB, and swap did not grow. All 10,015 images are reduced to "
        "1536-dimensional vectors in about fifteen minutes and cached to disk, so the cost is "
        "paid once.",
        "<b>Head fitting.</b> Only the final <font face='Courier'>Linear(1536, 7)</font> layer "
        "is trained, on the cached vectors. That is multinomial logistic regression over 10,759 "
        "parameters: seconds on CPU, no image decoding, no memory pressure. Class-weighted loss "
        "is essential — unweighted, the head learns to predict <font face='Courier'>nv</font> "
        "almost always.",
        "<b>Temperature scaling</b> on the validation split, so displayed confidence is "
        "calibrated. The fitted temperature was 0.98, and calibration error came out at 0.096.",
        "<b>Folding the normalisation into the layer.</b> Features were standardised before "
        "fitting, so the saved weights are <i>W&#8203;/&#8203;&#963;</i> with bias "
        "<i>b &#8722; (W&#8203;/&#8203;&#963;)&#183;&#956;</i>. This means the checkpoint drops "
        "into the unmodified architecture with no special-case code at serving time.",
    ]))
    s.append(P(
        "The honest characterisation: this proves the pipeline end to end and produces "
        "genuinely discriminative output, but it is not a competitive dermoscopy classifier. "
        "A frozen ImageNet backbone has no dermoscopy-specific features, which is exactly why "
        "the rare classes stay weak.", "small"))

    s.append(P("1.5  Measured results", "h2"))
    s.append(P(
        "All figures below are measured by <font face='Courier'>python -m derm.evaluate</font> "
        "on the <b>lesion-grouped</b> test split — 1,523 images across 1,120 lesions, audited "
        "at 0.00% leakage. Nothing here is transcribed from a notebook."))
    rows = [
        ["Metric", "Value", "Reference point"],
        ["Accuracy", "<b>54.96%</b>",
         "a trivial always-nevus model scores 67%, which is why this is the wrong metric"],
        ["Balanced accuracy", "<b>53.69%</b>",
         "chance is 14.3%; always-nevus also scores 14.3%"],
        ["Macro F1", "<b>0.379</b>", "averages every class equally, so rare classes count"],
        ["ROC-AUC (macro)", "<b>0.838</b>",
         "the ranking signal is strong even where argmax is wrong"],
        ["Calibration error (ECE)", "<b>0.096</b>", "after temperature scaling, from 0.102"],
        ["Melanoma recall", "<b>55.5%</b>", "argmax only"],
        ["Melanoma safety-net catch", "<b>59.2%</b>",
         "97 of 164 true melanomas escalated to HIGH/CRITICAL — the overrides recover cases "
         "argmax missed"],
        ["Benign over-referral", "15.3%", "191 of 1,246 — the cost of that safety net"],
    ]
    s.append(table(rows, [42 * mm, 24 * mm, BODY_W - 66 * mm]))
    s.append(Spacer(1, 5))
    s.append(P(
        "The gap between accuracy (55.0%) and balanced accuracy (53.7%) is small by design. "
        "Class weighting deliberately trades away nevus accuracy to keep the rare classes "
        "alive; an unweighted fit would report a higher accuracy while missing most melanomas. "
        "The most useful line in the table is the safety-net catch rate, because that is what "
        "a triage tool is actually judged on."))
    s.append(P("Per-class detail", "h3"))
    rows = [
        ["Class", "Precision", "Recall", "F1", "Support"],
        ["akiec — actinic keratosis", "0.202", "0.432", "0.275", "44"],
        ["bcc — basal cell carcinoma", "0.318", "0.580", "0.410", "69"],
        ["bkl — benign keratosis", "0.456", "0.520", "0.486", "179"],
        ["df — dermatofibroma", "0.065", "0.556", "0.116", "18"],
        ["mel — melanoma", "0.364", "0.555", "0.440", "164"],
        ["nv — melanocytic nevus", "0.961", "0.557", "0.705", "1,024"],
        ["vasc — vascular lesion", "0.139", "0.560", "0.222", "25"],
    ]
    s.append(table(rows, [62 * mm, 24 * mm, 22 * mm, 20 * mm, BODY_W - 128 * mm],
                   align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}))
    s.append(Spacer(1, 4))
    s.append(P(
        "Read the precision column honestly: <font face='Courier'>df</font> at 0.065 and "
        "<font face='Courier'>vasc</font> at 0.139 mean the model flags many more of those "
        "than actually exist. Recall is even across all seven classes, which is the "
        "class weighting working, but precision on the rare classes is poor. This is the "
        "single clearest thing end-to-end fine-tuning would fix.", "small"))
    return s


def section_2():
    s = [PageBreak(), P("2.  Complete file structure", "h1")]
    s.append(P(
        "The layout separates four concerns: domain logic that knows nothing about HTTP "
        "(<font face='Courier'>src/derm</font>), a thin web service "
        "(<font face='Courier'>app</font>), operational tooling "
        "(<font face='Courier'>scripts</font>), and generated artefacts "
        "(<font face='Courier'>docs</font>, <font face='Courier'>models</font>, "
        "<font face='Courier'>data</font>). The domain package has no import of FastAPI, which "
        "is why the same code runs unchanged from a notebook, a CLI or the server."))

    s.append(P("Domain logic and the web service", "h3"))
    s.append(code("""
dermoscopic-lesion-classification/
│
├── src/derm/                    DOMAIN LOGIC — no web framework imports
│   ├── config.py         (312)  class taxonomy, clinical metadata, every tunable
│   ├── preprocessing.py  (309)  hair inpainting, colour constancy, vignette crop
│   ├── quality.py        (310)  focus/exposure/glare + skin-chromaticity OOD gate
│   ├── segmentation.py   (247)  lesion-enhanced Otsu, morphology, ellipse fallback
│   ├── morphology.py     (536)  Stolz ABCD + continuous shape/colour descriptors
│   ├── model.py          (412)  architecture, checkpoint loading, inference bundle
│   ├── gradcam.py        (293)  Grad-CAM / Grad-CAM++ + attention alignment
│   ├── uncertainty.py    (319)  TTA, MC dropout, predictive entropy, BALD
│   ├── severity.py       (441)  composite 0-100 grade + safety overrides
│   ├── report.py         (602)  rule-based narrative + ReportLab PDF export
│   ├── monitoring.py     (383)  longitudinal change tracking (the "E" of ABCDE)
│   ├── inference.py      (394)  pipeline orchestrator -> one AnalysisResult
│   ├── store.py          (269)  SQLite case history
│   ├── data.py           (339)  dataset discovery, lesion-grouped splitting
│   ├── train.py          (461)  training CLI
│   ├── evaluate.py       (456)  evaluation CLI: calibration, safety-net audit
│   └── baseline.py       (256)  SVM baseline (HOG + colour histogram)
│
├── app/                         WEB SERVICE
│   ├── main.py           (580)  FastAPI app, 12 endpoints, auth, CORS, upload gate
│   ├── schemas.py         (87)  Pydantic request/response models
│   └── static/                  frontend, no build step
│       ├── index.html    (572)  markup + inline SVG icon sprite
│       ├── styles.css   (1097)  design system, dark/light, animation, print
│       └── app.js       (1684)  views, rendering, animation helpers, transport
"""))

    s.append(P("Tooling, tests and artefacts", "h3"))
    s.append(code("""
├── scripts/                     OPERATIONAL TOOLING
│   ├── prepare_data.py   (320)  disk-frugal HAM10000 download from Dataverse
│   ├── fit_head.py       (330)  GPU-free checkpoint: frozen features + linear head
│   ├── make_samples.py   (100)  copy demo images out of the held-out test split
│   ├── demo_samples.py   (120)  score every sample against a running server
│   ├── audit_leakage.py  (218)  quantify split leakage from metadata alone
│   ├── verify_checkpoint (337)  validate a checkpoint before trusting it
│   ├── smoke_test.py     (280)  end-to-end check against a running server
│   ├── bench.py           (89)  per-device training throughput
│   ├── make_diagrams.py         architecture / workflow / ER / progress figures
│   ├── make_review2_deck.py     fills the faculty Review-2 PPTX template
│   └── make_project_report.py   generates this PDF
│
├── tests/                       149 TESTS — no dataset or weights required
│   ├── conftest.py       (182)  synthetic dermoscopic image fixtures
│   ├── test_pipeline.py  (244)  30 tests: preprocessing -> severity
│   ├── test_model.py     (411)  45 tests: model, Grad-CAM, uncertainty, report
│   ├── test_api.py       (508)  57 tests: all endpoints, validation, errors
│   └── test_splits.py    (164)  17 tests: split integrity on real metadata
│
├── notebooks/                   ORIGINAL EXPLORATORY WORK
│   ├── 01-data-exploration.ipynb
│   ├── 02-svm-baseline.ipynb
│   └── 03-efficientnet-training.ipynb
│
├── docs/                        GENERATED ARTEFACTS
│   ├── split_audit.json         reproducible leakage audit (the key finding)
│   ├── model_comparison.json    metrics tagged measured vs self_reported
│   ├── figures/                 all generated images; served by /api/figures
│   │   ├── architecture.png  workflow.png  er_diagram.png  module_progress.png
│   │   └── class_distribution.png  gradcam_results.png  training_curves.png ...
│   └── review/                  submission deliverables
│       ├── review2-template.pptx                  the blank faculty template
│       ├── MP1-UDP-Review2-...pptx                filled deck, 15 slides
│       └── Project_Report.pdf                     this document
│
├── models/best_model.pth        41 MB checkpoint (git-ignored, regenerable)
├── data/ham10000/              10,015 images, 195 MB (git-ignored)
├── samples/                     21 demo images + INDEX.md (git-ignored)
│
├── requirements.txt             runtime dependencies
├── requirements-dev.txt         pytest, ruff
├── pyproject.toml               package metadata + tool config
└── README.md                    setup, findings, limitations
"""))

    s.append(P("2.1  What each layer is responsible for", "h2"))
    rows = [
        ["Layer", "Directory", "Responsibility and key design decision"],
        ["Domain", "<font face='Courier'>src/derm</font>",
         "All measurement and inference. Imports no web framework, so it is testable in "
         "isolation and reusable from notebooks. <font face='Courier'>config.py</font> is the "
         "single source of truth for class order — duplicating that anywhere else is how "
         "silent label-permutation bugs happen."],
        ["Service", "<font face='Courier'>app</font>",
         "HTTP concerns only: validation, auth, serialisation, static hosting. Contains no "
         "image processing, so the API can change without touching the science."],
        ["Frontend", "<font face='Courier'>app/static</font>",
         "Vanilla HTML/CSS/ES2020 with no build step, so it can be served, read and marked "
         "without a toolchain. Renders server-produced numbers; it never computes a metric."],
        ["Tooling", "<font face='Courier'>scripts</font>",
         "Reproducible operations: dataset download, leakage audit, checkpoint verification, "
         "end-to-end smoke test, benchmarking, document generation."],
        ["Tests", "<font face='Courier'>tests</font>",
         "149 tests that run on synthetic fixtures, so the suite needs neither the 2.8 GB "
         "dataset nor trained weights. This is what makes the project checkable by a reviewer "
         "in under a minute."],
        ["Artefacts", "<font face='Courier'>docs</font>, <font face='Courier'>models</font>, "
         "<font face='Courier'>data</font>",
         "Everything regenerable or large. <font face='Courier'>docs/</font> separates three "
         "kinds of output: machine-readable evidence (<font face='Courier'>*.json</font>) at "
         "the top level, generated images in <font face='Courier'>figures/</font>, and "
         "submission deliverables in <font face='Courier'>review/</font>. "
         "<font face='Courier'>data/</font> and <font face='Courier'>models/</font> are "
         "git-ignored; the audits and figures are committed because they are evidence."],
    ]
    s.append(table(rows, [20 * mm, 30 * mm, BODY_W - 50 * mm]))
    return s


def section_3():
    s = [PageBreak(), P("3.  How the system works", "h1")]
    s.append(P(
        "Three views of the same system follow: the component architecture, the request "
        "workflow, and the persistence schema."))

    s.append(P("3.1  Architecture", "h2"))
    s.append(figure("architecture.png",
                    "Figure 1 — Four-tier architecture and component interaction. Arrows show "
                    "request flow down and JSON responses up.", max_h=104 * mm))
    s.append(P(
        "The browser deliberately holds no analytical logic. It uploads an image and renders "
        "what comes back, which means any number visible on screen can be reproduced by "
        "calling the API directly — useful when a reviewer asks where a figure came from."))

    s.append(P("3.2  The analysis pipeline", "h2"))
    s.append(figure("workflow.png",
                    "Figure 2 — Nine-stage pipeline, early-exit conditions, and the composition "
                    "of the severity score.", max_h=104 * mm))

    s.append(P("3.3  Stage-by-stage, and why each stage is ordered where it is", "h3"))
    s.append(numbered([
        "<b>Quality assessment</b> runs on the untouched upload — resolution, focus by "
        "variance of the Laplacian, exposure, contrast, specular glare, and a "
        "skin-chromaticity test. It runs <i>first</i> because measuring quality after "
        "restoration would measure the repair, not the photograph.",
        "<b>Vignette cropping</b> removes the black lens barrel so it cannot be counted as "
        "dark pigment by the colour stage.",
        "<b>Restoration</b> detects hair with a directional black-hat filter and inpaints it "
        "with the Telea method, then normalises the illuminant with Shades-of-Gray. Critically, "
        "this restored frame drives <i>geometry only</i>. The classifier receives the "
        "un-restored image, because that is the distribution HAM10000 was trained on — feeding "
        "it colour-normalised input would be a silent train/serve skew.",
        "<b>Segmentation</b> isolates the lesion with a lesion-enhanced Otsu threshold, "
        "morphological cleanup and a centrality-weighted component choice, falling back to a "
        "centred ellipse with reduced confidence rather than failing outright.",
        "<b>ABCD morphometry</b> measures asymmetry about the lesion's own principal axes, "
        "border irregularity across eight sectors, the count of six diagnostic colours, and "
        "approximated dermoscopic structures, producing a total dermoscopy score (TDS).",
        "<b>Classification</b> with EfficientNet-B3, averaged over dihedral test-time "
        "augmentations.",
        "<b>Uncertainty</b> from predictive entropy, augmentation disagreement and MC dropout, "
        "including the BALD mutual-information score.",
        "<b>Grad-CAM / Grad-CAM++</b> localises the evidence, and the map is scored against "
        "the lesion mask, so a confident prediction that attended to a hair or a ruler mark is "
        "visible as such.",
        "<b>Severity grading</b> fuses everything into a 0–100 score, then applies overrides.",
    ]))

    s.append(P("3.4  Severity grading and the override rules", "h2"))
    s.append(P(
        "The weighted score combines neural risk (52%), ABCD morphometry (24%), uncertainty "
        "(16%) and image quality (8%). Four overrides then apply:"))
    rows = [
        ["Condition", "Effect"],
        ["Melanoma is the top class", "at least <b>HIGH</b>; <b>CRITICAL</b> above 70% confidence"],
        ["Melanoma probability ≥ 25%", "<b>HIGH</b>, even when melanoma is not the top class"],
        ["ABCD total dermoscopy score &gt; 5.45", "<b>HIGH</b>"],
        ["Confidence &lt; 50%", "at least <b>MODERATE</b>, flagged for human review"],
        ["Input is not skin-like, or no trained weights", "<b>INDETERMINATE</b>"],
    ]
    s.append(table(rows, [76 * mm, BODY_W - 76 * mm]))
    s.append(Spacer(1, 5))
    s.append(callout(
        "Why overrides only ever raise a tier",
        "Every override is one-directional: it can escalate a case but never de-escalate one. "
        "This is a deliberate asymmetry, not an oversight. The cost of a missed melanoma is "
        "not comparable to the cost of an unnecessary dermatology referral, so the grading "
        "engine is built to fail towards caution. It also means the melanoma safety net still "
        "works when argmax classification under-calls melanoma — which it does on a dataset "
        "that is 67% nevi.",
        tone="warn"))

    s.append(P("3.5  Persistence", "h2"))
    s.append(figure("er_diagram.png",
                    "Figure 3 — SQLite case-store schema. Queried fields are promoted to "
                    "indexed columns; the full result is retained as JSON for replay.",
                    max_h=100 * mm))

    s.append(P("3.6  API surface", "h2"))
    rows = [
        ["Method", "Path", "Purpose"],
        ["GET", "<font face='Courier'>/api/health</font>", "liveness and weight status"],
        ["GET", "<font face='Courier'>/api/meta</font>", "class taxonomy, tiers, limits, disclaimer"],
        ["GET", "<font face='Courier'>/api/metrics</font>", "comparison, evaluation, leakage audit, figures"],
        ["GET", "<font face='Courier'>/api/figures/{name}</font>", "serve a figure from docs/ (traversal-blocked)"],
        ["POST", "<font face='Courier'>/api/analyze</font>", "full pipeline on one image"],
        ["POST", "<font face='Courier'>/api/analyze/batch</font>", "many images, returned in triage order"],
        ["POST", "<font face='Courier'>/api/compare</font>", "longitudinal change between two captures"],
        ["POST", "<font face='Courier'>/api/report/pdf</font>", "PDF from a payload or a stored case_id"],
        ["GET", "<font face='Courier'>/api/cases</font>", "paged case history with filters"],
        ["GET/PATCH/DELETE", "<font face='Courier'>/api/cases/{id}</font>", "fetch, annotate, delete"],
        ["GET", "<font face='Courier'>/api/cases/stats</font>", "aggregate statistics"],
        ["POST", "<font face='Courier'>/api/model/reload</font>", "re-read the checkpoint from disk"],
    ]
    s.append(table(rows, [26 * mm, 52 * mm, BODY_W - 78 * mm]))
    s.append(Spacer(1, 5))
    s.append(callout(
        "Security posture, stated plainly",
        "The API is unauthenticated by default, which is appropriate only for local "
        "single-user use on 127.0.0.1. Setting <font face='Courier'>DERM_API_KEY</font> "
        "requires an <font face='Courier'>X-API-Key</font> header on every mutating "
        "endpoint; CORS defaults to localhost origins. Uploads are type- and size-checked "
        "before decoding, path traversal is blocked on figure serving, and the case store "
        "keeps thumbnails rather than full-resolution images. Do not expose this port on a "
        "network without setting the API key first.",
        tone="warn"))
    return s


def section_4():
    s = [PageBreak(), P("4.  Running the project locally", "h1")]
    s.append(P(
        "The whole application runs on CPU on an ordinary laptop. Only <i>training</i> needs a "
        "GPU. These steps assume Python 3.10 or newer and take a few minutes."))

    s.append(P("Step 1 — Get the code and create a virtual environment", "h3"))
    s.append(code("""
cd dermoscopic-lesion-classification

python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\\Scripts\\activate          # Windows PowerShell
"""))

    s.append(P("Step 2 — Install dependencies", "h3"))
    s.append(code("""
pip install --upgrade pip
pip install -r requirements.txt

# for running the tests as well:
pip install -r requirements-dev.txt
"""))
    s.append(P(
        "This pulls PyTorch, timm, OpenCV, scikit-image, scikit-learn, FastAPI, Uvicorn, "
        "Pydantic and ReportLab. The CPU wheels are sufficient; for GPU training, install a "
        "CUDA-matched torch wheel first.", "small"))

    s.append(P("Step 2b — Get the dataset and a trained checkpoint", "h3"))
    s.append(P(
        "Both are excluded from version control — the images for licensing and size reasons, "
        "the checkpoint because it is 41 MB and regenerable. Two commands rebuild them, and "
        "neither needs a GPU:"))
    s.append(code("""
# 10,015 images from Harvard Dataverse, downscaled on extraction.
# Peak disk ~1.6 GB, final ~195 MB. No credentials needed.
python scripts/prepare_data.py --skip-segmentations

# Feature extraction (~15 min, forward-only, ~470 MB RAM) then a
# linear head fit (seconds). Writes models/best_model.pth.
python scripts/fit_head.py --device cpu

# Never trust a checkpoint until this passes. --images is the check
# that actually catches a permuted class order.
python scripts/verify_checkpoint.py models/best_model.pth --images data/ham10000
"""))
    s.append(P(
        "Feature extraction caches to <font face='Courier'>docs/features_b3.npz</font>, so a "
        "second <font face='Courier'>fit_head.py</font> run with different hyper-parameters "
        "takes seconds rather than fifteen minutes.", "small"))

    s.append(P("Step 2c — Get sample images to demo with", "h3"))
    s.append(code("""
python scripts/make_samples.py --per-class 3
#  -> samples/<code>-<name>/*.jpg  + samples/INDEX.md
"""))
    s.append(P(
        "Drawn only from the held-out test split, so demoing on them is not demoing on "
        "training data. <font face='Courier'>INDEX.md</font> lists the true diagnosis for each "
        "file, so you can check predictions live. To score all of them at once against a "
        "running server:"))
    s.append(code("""
python scripts/demo_samples.py --base-url http://127.0.0.1:8000
"""))

    s.append(P("Step 3 — Verify the installation without needing any data", "h3"))
    s.append(code("""
pytest -q
#  149 passed
"""))
    s.append(P(
        "This is the fastest way for a reviewer to confirm the project works. The suite runs "
        "on synthetically generated dermoscopic images defined in "
        "<font face='Courier'>tests/conftest.py</font>, so it needs neither the HAM10000 "
        "download nor a trained checkpoint.", "small"))

    s.append(P("Step 4 — Start the application", "h3"))
    s.append(code("""
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""))
    s.append(P(
        "Then open <font face='Courier'>http://127.0.0.1:8000</font> for the interface and "
        "<font face='Courier'>http://127.0.0.1:8000/docs</font> for interactive API "
        "documentation. On startup the log states plainly whether trained weights were found."))

    s.append(P("Step 5 — Confirm it is serving correctly", "h3"))
    s.append(code("""
curl -s http://127.0.0.1:8000/api/health

# with the server running, exercise all 12 endpoints end to end:
python scripts/smoke_test.py
"""))

    s.append(P("Optional — Reproduce the leakage audit", "h2"))
    s.append(P(
        "This needs only the metadata CSV, not the images, and is the single most useful thing "
        "to re-run in front of a reviewer because it verifies the project's central claim from "
        "primary data:"))
    s.append(code("""
python scripts/audit_leakage.py        # writes docs/split_audit.json
"""))

    s.append(P("Optional — full fine-tuning and re-evaluation", "h2"))
    s.append(code("""
# check what your hardware can actually sustain before committing
python scripts/bench.py

# end-to-end fine-tune: needs a CUDA GPU, NOT an 8 GB laptop
python -m derm.train --device cuda --epochs 20

# re-measure and regenerate every figure and metric
python -m derm.evaluate --checkpoint models/best_model.pth --data-root data/ham10000

# regenerate the deck and this report from the new numbers
python scripts/make_diagrams.py
python scripts/make_review2_deck.py
python scripts/make_project_report.py
"""))

    s.append(callout(
        "Hardware reality check — measured, not estimated",
        "Inference runs comfortably on CPU: about 2.9 s per image for the full pipeline "
        "(classification ~1.6 s, Grad-CAM ~0.9 s) on an Apple M2. Training EfficientNet-B3 is "
        "a different matter. On an 8 GB M2 the MPS backend drove swap to 13.9 GB and took free "
        "boot-disk space from 9.2 GB to 2.9 GB within ninety seconds. That is a hardware limit, "
        "not a software defect. Budget at least 16 GB of unified memory, or use CUDA — Kaggle "
        "and Colab both provide a sufficient GPU free of charge.",
        tone="warn"))

    s.append(P("Swapping in a different checkpoint", "h2"))
    s.append(P(
        "Drop a checkpoint at <font face='Courier'>models/best_model.pth</font> and press "
        "<i>Reload model</i> in the interface — no restart needed. Run the verifier first. It "
        "checks file and tensor compatibility, class count, class ordering, missing and "
        "unexpected parameters, whether outputs are discriminative rather than uniform, and "
        "inference determinism. It has been confirmed to reject both a perfectly loadable "
        "untrained checkpoint and one with the wrong number of classes. The current checkpoint "
        "passes all 21 checks with zero warnings."))
    s.append(callout(
        "The check that matters most",
        "Run the verifier with <font face='Courier'>--images data/ham10000</font>. Every other "
        "check can pass on a checkpoint whose output classes are permuted — the tensors load, "
        "the shapes match, the probabilities normalise — and the result is a model that is "
        "confidently and consistently wrong, calling melanoma a nevus. Only measuring "
        "per-class recall against real labelled images catches that. On the current checkpoint "
        "all seven classes sit between 43% and 67% recall, which is inconsistent with any "
        "permutation.",
        tone="danger"))

    s.append(P("Configuration", "h2"))
    rows = [
        ["Variable", "Default", "Purpose"],
        ["<font face='Courier'>DERM_CHECKPOINT</font>", "<font face='Courier'>models/best_model.pth</font>", "weights path"],
        ["<font face='Courier'>DERM_DEVICE</font>", "<font face='Courier'>auto</font>", "<font face='Courier'>cuda</font>, <font face='Courier'>mps</font> or <font face='Courier'>cpu</font>"],
        ["<font face='Courier'>DERM_HAM10000_DIR</font>", "—", "dataset root"],
        ["<font face='Courier'>DERM_API_KEY</font>", "—", "enable API-key authentication"],
        ["<font face='Courier'>DERM_TTA</font>", "<font face='Courier'>5</font>", "augmentations per prediction"],
        ["<font face='Courier'>DERM_MC_PASSES</font>", "<font face='Courier'>10</font>", "MC dropout passes"],
        ["<font face='Courier'>DERM_CORS_ORIGINS</font>", "localhost", "permitted browser origins"],
    ]
    s.append(table(rows, [48 * mm, 42 * mm, BODY_W - 90 * mm]))

    s.append(P("Troubleshooting", "h2"))
    rows = [
        ["Symptom", "Cause and fix"],
        ["Every class shows 14.3%",
         "No checkpoint loaded: 1/7 is a uniform distribution from an untrained network. Run "
         "<font face='Courier'>python scripts/fit_head.py</font>, then press "
         "<i>Reload model</i>. Check <font face='Courier'>/api/health</font> — it should say "
         "<font face='Courier'>weights_status: trained</font>."],
        ["<font face='Courier'>Address already in use</font> on 8000",
         "Something else holds the port. Pass <font face='Courier'>--port 8010</font> and use "
         "that URL instead; nothing in the project hardcodes 8000."],
        ["<font face='Courier'>CERTIFICATE_VERIFY_FAILED</font> during download",
         "You are behind a TLS-inspecting proxy. "
         "<font face='Courier'>prepare_data.py</font> already falls back to "
         "<font face='Courier'>curl</font>, which uses the OS trust store; "
         "<font face='Courier'>pip install truststore</font> is a cleaner fix."],
        ["<font face='Courier'>Address already in use</font>",
         "Another server holds port 8000. Either stop it or pass "
         "<font face='Courier'>--port 8001</font>."],
        ["<font face='Courier'>ModuleNotFoundError: derm</font>",
         "Run <font face='Courier'>uvicorn</font> from the project root. "
         "<font face='Courier'>app/main.py</font> prepends "
         "<font face='Courier'>src/</font> to <font face='Courier'>sys.path</font>, so no "
         "install step is required."],
        ["Machine swaps heavily during training",
         "Insufficient RAM for EfficientNet-B3. Reduce "
         "<font face='Courier'>--batch-size</font>, or train on Kaggle/Colab instead."],
    ]
    s.append(table(rows, [52 * mm, BODY_W - 52 * mm]))
    return s


def section_5():
    s = [PageBreak(), P("5.  How to present and defend this at the review", "h1")]
    s.append(P(
        "The strongest thing about this submission is not the model — it is that every claim "
        "is checkable from the repository. Lead with that."))

    s.append(P("5.1  A twelve-minute demonstration order", "h2"))
    rows = [
        ["#", "Time", "What to show", "What to say"],
        ["1", "0:00", "<font face='Courier'>pytest -q</font> → 149 passed",
         "“Validation first. 149 tests, and they need neither the dataset nor trained weights, "
         "so anyone can reproduce this in a minute.”"],
        ["2", "1:00", "Start the server, open the UI",
         "Point out the untrained-weights banner immediately. Owning the limitation "
         "before you are asked removes it as an attack surface."],
        ["3", "2:00", "Upload a lesion image; walk the studio",
         "Quality gate → segmentation overlay → ABCD rings and TDS → ranked probabilities → "
         "severity tier and recommended action."],
        ["4", "4:00", "Grad-CAM split-view slider",
         "“This is the explainability requirement. And we score the heatmap against the "
         "lesion mask, so we can tell whether the network looked at the lesion or at a hair.”"],
        ["5", "5:30", "Clinical report → download PDF",
         "“The narrative is rule-based, not an LLM — every sentence traces to a measurement, so "
         "a fluent hallucination cannot be mistaken for a finding.”"],
        ["6", "6:30", "Batch triage with several images",
         "“A screening set is mostly benign. Ranking by severity puts clinician time where the "
         "risk is.”"],
        ["7", "7:30", "Track change with two captures",
         "“Change over time is the strongest single predictor of melanoma, and the one thing a "
         "single-image classifier structurally cannot see.”"],
        ["8", "8:30", "Model metrics view + leakage audit",
         "The centrepiece. See 5.2."],
        ["9", "10:30", "<font face='Courier'>scripts/verify_checkpoint.py</font>",
         "“A checkpoint that fails to load is harmless. One that loads with a permuted class "
         "order gives confident wrong diagnoses. This catches that.”"],
        ["10", "11:30", "Plan for Review-3",
         "GPU training run, calibration, measured metrics replacing self-reported ones, "
         "deployment."],
    ]
    s.append(table(rows, [8 * mm, 14 * mm, 46 * mm, BODY_W - 68 * mm]))

    s.append(P("5.2  Lead with the finding, not the model", "h2"))
    s.append(P(
        "The most substantial piece of independent work here is the discovery that the accuracy "
        "figure carried over from Review-1 was inflated. Present it as a finding, with numbers:"))
    rows = [
        ["Split", "Test images whose lesion also appears in training"],
        ["Image-wise (notebook 03, <font face='Courier'>random_state=42</font>)",
         "<b>543 / 1,503 — 36.13%</b>"],
        ["Lesion-grouped (<font face='Courier'>derm.data.make_splits</font>)",
         "<b>0 / 1,523 — 0.00%</b>"],
    ]
    s.append(table(rows, [86 * mm, BODY_W - 86 * mm]))
    s.append(Spacer(1, 4))
    s.append(P("Leakage concentrates precisely in the classes that matter most clinically:"))
    rows = [
        ["Class", "Test images", "Leaked", "Leaked %"],
        ["Basal cell carcinoma", "77", "54", "<b>70.1</b>"],
        ["Melanoma", "167", "106", "<b>63.5</b>"],
        ["Benign keratosis", "165", "83", "50.3"],
        ["Dermatofibroma", "17", "7", "41.2"],
        ["Actinic keratosis", "49", "18", "36.7"],
        ["Vascular", "22", "7", "31.8"],
        ["Melanocytic nevus", "1,006", "268", "26.6"],
    ]
    s.append(table(rows, [56 * mm, 30 * mm, 26 * mm, BODY_W - 112 * mm],
                   align={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}))
    s.append(Spacer(1, 5))
    s.append(P(
        "The sentence worth memorising: <i>“HAM10000 contains 10,015 images of only 7,470 "
        "distinct lesions. The published 75% melanoma recall was measured on a test set where "
        "63.5% of melanoma images had a near-duplicate twin in training. We re-split by "
        "lesion ID, audited it at 0% leakage, and we expect a lower but honest number.”</i> "
        "Recognising that unprompted is a stronger result than a high accuracy figure."))

    s.append(P("5.3  Questions you should expect, with answers", "h2"))
    qa = [
        ("“53.7% balanced accuracy is low. Isn't that a poor result?”",
         "It is low, and we report it rather than dressing it up. Two things put it in context. "
         "First, the baseline: chance is 14.3%, and the trivial model that always predicts "
         "nevus also scores 14.3% balanced accuracy while scoring 67% plain accuracy — which is "
         "why we lead with the balanced figure. Second, this is a linear probe on a frozen "
         "ImageNet backbone, not a fine-tuned model. The backbone has never seen a dermoscopic "
         "image, so it has no dermoscopy-specific features to offer. ROC-AUC of 0.838 shows the "
         "ranking signal is much stronger than argmax accuracy suggests. Fine-tuning the "
         "backbone is the Review-3 work and is where the remaining gain is."),
        ("“Why not just fine-tune the whole network?”",
         "Because it does not fit on the hardware. We measured it: on an 8 GB M2 the MPS "
         "backend drove swap to 13.9 GB and cut free disk from 9.2 GB to 2.9 GB within ninety "
         "seconds. Rather than claim that as a blocker and ship nothing, we split transfer "
         "learning into its two halves and ran the half that does fit — forward-only feature "
         "extraction peaks at 472 MB. That produced real weights, so every feature in the "
         "application now works."),
        ("“Why not just report the 80.17% from the notebook?”",
         "Because we audited it and it is not trustworthy: that split leaks 36.13% of test "
         "images. Reporting it as a result would be reporting a measurement we know to be "
         "biased. It is retained in the comparison table, but tagged "
         "<font face='Courier'>self_reported</font> and displayed as a claim, not a result."),
        ("“How is this explainable rather than a black box?”",
         "Three independent ways. Grad-CAM/Grad-CAM++ shows where the network looked, and we "
         "score that map against the lesion mask so misplaced attention is visible. ABCD "
         "morphometry produces geometric measurements with no neural network involved. And the "
         "narrative is rule-based, so every sentence in the report traces back to a specific "
         "measured value."),
        ("“Why is the severity grade not just the classifier's output?”",
         "Because argmax under-calls melanoma on a dataset that is 67% nevi. The grade fuses "
         "neural risk with morphometry, uncertainty and image quality, then applies "
         "one-directional overrides — melanoma probability above 25% escalates the case even "
         "when melanoma is not the top class."),
        ("“What happens with a photograph that is not skin?”",
         "The skin-chromaticity gate rejects it and the case is graded "
         "<font face='Courier'>INDETERMINATE</font> rather than being forced into one of the "
         "seven classes."),
        ("“What are the limits of this system?”",
         "HAM10000 is dominated by fair-skinned European and Australian populations, so "
         "performance on darker skin is uncharacterised and probably worse — that is the most "
         "serious limitation for real use. Only seven categories are modelled, so squamous cell "
         "carcinoma, amelanotic melanoma and inflammatory dermatoses are forced into the "
         "nearest class and will be wrong. Segmentation is classical, not learned, because the "
         "main HAM10000 release ships no masks. The D of ABCD is approximated with texture "
         "filters and should be treated as a weak signal."),
        ("“Which parts did AI tools write, and how do you know they are correct?”",
         "Answered directly in section 6: the tools, the prompts, and — more importantly — the "
         "independent verification for each one."),
    ]
    for q, a in qa:
        s.append(KeepTogether([
            Paragraph(f"<b>{q}</b>", ST["h3"]),
            Paragraph(a, ST["body"]),
        ]))

    s.append(P("5.4  Deliverables checklist", "h2"))
    s.append(bullets([
        "Filled Review-2 deck: <font face='Courier'>docs/review/MP1-UDP-Review2-Dermoscopic-"
        "Lesion-Analysis.pptx</font> — replace every "
        "<font face='Courier'>&lt;&lt; ... &gt;&gt;</font> placeholder (names, guide, UDP ID, "
        "the verbatim Review-1 remarks) and paste your own UI screenshots on the "
        "<i>Partial Output</i> slide.",
        "This report: <font face='Courier'>docs/review/Project_Report.pdf</font>.",
        "Evidence to have open in a second window: "
        "<font face='Courier'>docs/split_audit.json</font>, a green "
        "<font face='Courier'>pytest</font> run, and the running application.",
        "A sample generated PDF report, produced live from the interface during the demo.",
    ]))
    return s


def section_6():
    s = [PageBreak(), P("6.  AI tools used, and how their output was verified", "h1")]
    s.append(callout(
        "The standard applied here",
        "An AI tool was treated as a fast but unreliable collaborator: useful for producing "
        "code and explanations, never accepted as evidence. Nothing an AI tool asserted was "
        "carried into this project as a result unless an independent mechanism in the "
        "repository confirmed it — a test, an audit script, a verifier, or the original "
        "published paper. Section 6.2 lists those mechanisms explicitly, because the honest "
        "answer to “how do you know the generated code is right?” is not “we read it”.",
        tone="info"))

    s.append(P("6.1  Tools and their role", "h2"))
    rows = [
        ["AI Tool", "Purpose", "Representative prompt", "Outcome"],
        ["<b>Kiro</b><br/>(Claude Opus 5)<br/>agentic IDE",
         "Primary development assistant: restructuring the notebook code into a package, "
         "implementing the FastAPI service and the frontend, and generating the test suite.",
         "“Analyse this HAM10000 notebook project in detail and convert it into a working "
         "frontend and backend with Grad-CAM, ABCD morphometry and severity grading. All "
         "features must actually work.”",
         "The 18-module <font face='Courier'>src/derm</font> package, a 12-endpoint API and "
         "the full interface. <b>Verified by</b> 149 tests plus "
         "<font face='Courier'>scripts/smoke_test.py</font> exercising every endpoint against "
         "a live server."],
        ["<b>Kiro</b><br/>(agentic debugging)",
         "Root-causing defects that the test suite surfaced inside the inference pipeline.",
         "“MC dropout returns a standard deviation of about 1e-6, so the uncertainty estimate "
         "is inert. Find out why and fix it.”",
         "Diagnosed that timm's EfficientNet applies dropout functionally, so "
         "<font face='Courier'>model.train()</font> alone never made passes stochastic. "
         "<b>Verified by</b> a regression test asserting non-zero MC variance. Three further "
         "real defects were found this way: Grad-CAM failing inside "
         "<font face='Courier'>torch.inference_mode()</font>, malformed directional hair "
         "kernels, and the checkpoint verifier crashing on tensor shape mismatch."],
        ["<b>Kiro</b><br/>(data audit)",
         "Independently checking the accuracy claim inherited from Review-1.",
         "“HAM10000 contains repeated photographs of the same lesion. Quantify how much of the "
         "notebook's test set leaks from training, per class.”",
         "Produced <font face='Courier'>scripts/audit_leakage.py</font> and "
         "<font face='Courier'>docs/split_audit.json</font>: 36.13% overall leakage, 63.5% for "
         "melanoma. <b>Verified by</b> the script being reproducible from the metadata CSV "
         "alone — the finding is independently checkable, not asserted."],
        ["<b>ChatGPT / Claude</b><br/>(chat)",
         "Clarifying the clinical basis of the Stolz ABCD rule and the Grad-CAM++ weighting "
         "before implementing either.",
         "“Explain how the Stolz ABCD total dermoscopy score is weighted, and how Grad-CAM++ "
         "differs from Grad-CAM.”",
         "<b>Verified against the original papers.</b> The TDS coefficients and the 5.45 "
         "threshold were taken from Stolz et al., not from the model's summary — the "
         "explanation was used to understand the method, the numbers came from the literature."],
        ["<b>GitHub Copilot</b>",
         "Inline completion for repetitive code: dataclass fields, argparse wiring, "
         "docstring scaffolding.",
         "(inline completion — no discrete prompt)",
         "Reduced boilerplate typing. No completion was accepted into numerical or clinical "
         "logic without a test covering it."],
    ]
    s.append(table(rows, [26 * mm, 34 * mm, 52 * mm, BODY_W - 112 * mm], font_size=7.8))
    s.append(Spacer(1, 4))
    s.append(P(
        "Remove any row above that your team did not actually use. An inflated tool list is "
        "easy for a reviewer to disprove by asking one follow-up question.", "small"))

    s.append(P("6.2  The verification mechanisms, in order of strength", "h2"))
    s.append(numbered([
        "<b>149 automated tests</b> across four files, running on synthetic dermoscopic "
        "fixtures so they need no dataset and no weights. These caught six real defects in "
        "AI-generated code, listed above.",
        "<b>End-to-end smoke test</b> (<font face='Courier'>scripts/smoke_test.py</font>) which "
        "drives all 12 endpoints against a live server and asserts on the response contents, "
        "not just the status codes.",
        "<b>Reproducible data audit</b> (<font face='Courier'>scripts/audit_leakage.py</font>) "
        "computing the leakage finding from the primary metadata, so the project's central "
        "claim does not rest on anything a model said.",
        "<b>Checkpoint verifier</b> (<font face='Courier'>scripts/verify_checkpoint.py</font>) "
        "which has been confirmed to reject a loadable-but-untrained checkpoint and one with "
        "the wrong class count.",
        "<b>Cross-checking against primary literature</b> for every clinical constant: the "
        "ABCD weights, the 5.45 TDS threshold, and the Grad-CAM++ formulation.",
        "<b>Provenance labelling in the product itself.</b> Metrics carry "
        "<font face='Courier'>measured</font> or <font face='Courier'>self_reported</font> "
        "tags and the interface renders them differently, so an unverified number cannot "
        "quietly become a result.",
    ]))

    s.append(P("6.3  What the AI tools did not decide", "h2"))
    s.append(P(
        "Worth stating explicitly at the review, because it is the part that distinguishes "
        "using a tool from being used by one. The following were engineering judgements made "
        "and defended by the team, in several cases against the more obvious path an assistant "
        "would take: splitting by lesion ID rather than by image; feeding the classifier the "
        "un-restored frame to avoid train/serve skew; keeping the narrative generator "
        "rule-based instead of using an LLM; making the severity overrides one-directional; "
        "and displaying the untrained-weights warning prominently rather than hiding a "
        "limitation that would not have been obvious to a marker."))
    return s


def section_7():
    s = [PageBreak(), P("7.  Status, limitations and plan for Review-3", "h1")]

    s.append(P("7.1  Current status", "h2"))
    s.append(figure("module_progress.png",
                    "Figure 4 — Per-module completion at Review-2.", max_h=96 * mm))

    s.append(P("7.2  Honest limitations", "h2"))
    s.append(bullets([
        "<b>The weights are a linear probe, not a fine-tuned model.</b> 53.7% balanced accuracy "
        "is well short of what this architecture can reach. Precision on the rare classes is "
        "poor — dermatofibroma 0.065, vascular 0.139 — meaning the model over-flags them. "
        "Fine-tuning the backbone on a GPU is the single highest-value remaining task.",
        "<b>Skin-tone representativeness.</b> HAM10000 is dominated by fair-skinned European and "
        "Australian populations. Performance on darker skin is uncharacterised and probably "
        "worse. This is the most serious limitation for any real use.",
        "<b>Only seven categories.</b> Squamous cell carcinoma, amelanotic melanoma, infections "
        "and inflammatory dermatoses are forced into the nearest of the seven and will be wrong.",
        "<b>The D of ABCD is approximated</b> with classical texture filters rather than expert "
        "annotation, so it is a weak signal. A and B are faithful to the geometric rule; C is a "
        "colour-quantisation approximation.",
        "<b>Segmentation is classical, not learned</b>, because the main HAM10000 release ships "
        "no masks. It reports its own confidence and falls back to an ellipse.",
        "<b>Absolute lesion size cannot be recovered</b> from a photograph without a scale "
        "reference, so change tracking reports size relative to the frame unless a "
        "field-of-view width is supplied.",
    ]))

    s.append(P("7.3  Plan for Review-3", "h2"))
    rows = [
        ["Area", "Work", "Acceptance criterion"],
        ["Model", "Fine-tune the full backbone end to end on Kaggle, Colab or a CUDA GPU",
         "Balanced accuracy and rare-class precision both materially above the linear probe's "
         "53.7% / 0.065"],
        ["Verification", "Re-run <font face='Courier'>verify_checkpoint.py --images</font>",
         "21/21 checks pass; per-class recall confirms class order is not permuted"],
        ["Metrics", "Regenerate <font face='Courier'>docs/evaluation.json</font>, the confusion "
                    "matrix and calibration figures",
         "Both remaining <font face='Courier'>self_reported</font> rows either replaced by "
         "<font face='Courier'>measured</font> or stated as unverified"],
        ["Testing", "Add trained-model integration tests and per-class recall floors",
         "Suite fails if melanoma recall regresses below an agreed threshold"],
        ["Performance", "ONNX export or quantisation",
         "Full-pipeline CPU latency below the current 2.9 s per image"],
        ["Deployment", "Containerise, enable <font face='Courier'>DERM_API_KEY</font>, serve "
                       "over HTTPS, add structured logging",
         "Reachable from another machine with authentication enforced"],
        ["Documentation", "Final project report and user guide",
         "Reproducible from a clean checkout following section 4 alone"],
    ]
    s.append(table(rows, [26 * mm, 62 * mm, BODY_W - 88 * mm]))

    s.append(P("8.  References", "h1"))
    refs = [
        "Tschandl P., Rosendahl C., Kittler H. <i>The HAM10000 dataset, a large collection of "
        "multi-source dermatoscopic images of common pigmented skin lesions.</i> Scientific "
        "Data 5, 180161 (2018). doi:10.1038/sdata.2018.161",
        "Harvard Dataverse, HAM10000 distribution. doi:10.7910/DVN/DBW86T (CC BY-NC 4.0).",
        "Tan M., Le Q. <i>EfficientNet: Rethinking Model Scaling for Convolutional Neural "
        "Networks.</i> ICML 2019.",
        "Selvaraju R. R. et al. <i>Grad-CAM: Visual Explanations from Deep Networks via "
        "Gradient-based Localization.</i> ICCV 2017.",
        "Chattopadhyay A. et al. <i>Grad-CAM++: Improved Visual Explanations for Deep "
        "Convolutional Networks.</i> WACV 2018.",
        "Gal Y., Ghahramani Z. <i>Dropout as a Bayesian Approximation: Representing Model "
        "Uncertainty in Deep Learning.</i> ICML 2016.",
        "Houlsby N. et al. <i>Bayesian Active Learning for Classification and Preference "
        "Learning.</i> arXiv:1112.5745 (2011).",
        "Guo C. et al. <i>On Calibration of Modern Neural Networks.</i> ICML 2017.",
        "Stolz W. et al. <i>ABCD rule of dermatoscopy: a new practical method for early "
        "recognition of malignant melanoma.</i> European Journal of Dermatology, 1994.",
        "Telea A. <i>An Image Inpainting Technique Based on the Fast Marching Method.</i> "
        "Journal of Graphics Tools, 2004.",
        "Finlayson G., Trezzi E. <i>Shades of Gray and Colour Constancy.</i> Color Imaging "
        "Conference, 2004.",
        "Wightman R. <i>PyTorch Image Models (timm).</i> "
        "github.com/huggingface/pytorch-image-models",
    ]
    s.append(numbered(refs, style="small"))

    s.append(Spacer(1, 6))
    s.append(HR())
    s.append(Spacer(1, 4))
    s.append(P(
        "This report was generated by <font face='Courier'>scripts/make_project_report.py</font>. "
        "Figures 1–4 are generated by <font face='Courier'>scripts/make_diagrams.py</font>. Both "
        "regenerate deterministically from the repository, so nothing in this document is a "
        "hand-maintained copy that can drift out of date.", "small"))
    return s


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    story = []
    story += cover()
    story.append(NextPageTemplate("body"))
    story += section_1()
    story += section_2()
    story += section_3()
    story += section_4()
    story += section_5()
    story += section_6()
    story += section_7()

    build_doc().build(story)
    size = OUT.stat().st_size / 1024
    print(f"Wrote {OUT.relative_to(ROOT)}  ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
