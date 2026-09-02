"""Clinical narrative generation and PDF export.

The narrative is produced by a deterministic rule-based generator rather than a
language model. That is a deliberate choice for a diagnostic-support tool: every
sentence is traceable to a specific measurement, the same input always yields the
same report, and there is no possibility of a fluent hallucination being read as
a clinical finding.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import LESION_CLASSES, MEDICAL_DISCLAIMER
from .morphology import MorphologyFeatures
from .quality import QualityReport
from .severity import SeverityAssessment
from .uncertainty import UncertaintyReport

COLOR_LABELS = {
    "white": "white / depigmented areas",
    "red": "red or vascular areas",
    "light_brown": "light brown pigment",
    "dark_brown": "dark brown pigment",
    "blue_gray": "blue-gray pigment",
    "black": "black pigment",
}

STRUCTURE_LABELS = {
    "structureless_area": "a broad structureless (homogeneous) area",
    "dots_globules": "dots and globules",
    "pigment_network": "a reticular pigment network",
    "streaks": "peripheral streaks or branched extensions",
    "blue_white_veil": "a blue-white veil",
}


@dataclass
class ClinicalNarrative:
    """Structured, human-readable report sections."""

    impression: str
    summary: str
    findings: list[str] = field(default_factory=list)
    differential: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    recommendation: list[str] = field(default_factory=list)
    disclaimer: str = MEDICAL_DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "impression": self.impression,
            "summary": self.summary,
            "findings": self.findings,
            "differential": self.differential,
            "explanation": self.explanation,
            "limitations": self.limitations,
            "recommendation": self.recommendation,
            "disclaimer": self.disclaimer,
        }

    def to_text(self) -> str:
        blocks = [
            f"IMPRESSION\n{self.impression}",
            f"SUMMARY\n{self.summary}",
        ]
        for title, items in (
            ("FINDINGS", self.findings),
            ("DIFFERENTIAL", self.differential),
            ("BASIS FOR THE PREDICTION", self.explanation),
            ("LIMITATIONS", self.limitations),
            ("RECOMMENDATION", self.recommendation),
        ):
            if items:
                bullets = "\n".join(f"  - {item}" for item in items)
                blocks.append(f"{title}\n{bullets}")
        blocks.append(f"DISCLAIMER\n{self.disclaimer}")
        return "\n\n".join(blocks)


def _describe_border(morphology: MorphologyFeatures) -> str:
    score = morphology.abcd.border
    if score >= 6:
        quality_word = "markedly irregular, with notching across most of the periphery"
    elif score >= 3:
        quality_word = "irregular in several segments"
    elif score >= 1:
        quality_word = "mildly irregular in one or two segments"
    else:
        quality_word = "smooth and well demarcated"
    return (
        f"Border is {quality_word} (B={score}/8, circularity "
        f"{morphology.circularity:.2f}, solidity {morphology.solidity:.2f})."
    )


def _describe_asymmetry(morphology: MorphologyFeatures) -> str:
    score = morphology.abcd.asymmetry
    mapping = {
        0: "symmetric about both principal axes",
        1: "asymmetric about one principal axis",
        2: "asymmetric about both principal axes",
    }
    return (
        f"Lesion is {mapping.get(score, 'of indeterminate symmetry')} "
        f"(A={score}/2, shape asymmetry index {morphology.asymmetry_index:.2f}, "
        f"pigment asymmetry {morphology.color_asymmetry:.2f})."
    )


def _describe_colors(morphology: MorphologyFeatures) -> str:
    present = morphology.abcd.colors_present
    if not present:
        return "No single colour occupies more than 5% of the lesion (C=0/6)."
    labels = ", ".join(COLOR_LABELS.get(c, c) for c in present)
    tail = (
        " Five or more colours is itself a recognised melanoma-specific feature."
        if len(present) >= 5
        else ""
    )
    return f"{len(present)} of 6 diagnostic colours present (C={len(present)}/6): {labels}.{tail}"


def _describe_structures(morphology: MorphologyFeatures) -> str:
    present = morphology.abcd.structures_present
    if not present:
        return "No distinct dermoscopic structures were detected (D=0/5)."
    labels = ", ".join(STRUCTURE_LABELS.get(s, s) for s in present)
    return f"Detected structures (D={len(present)}/5): {labels}."


def generate_narrative(
    *,
    predictions: list[dict[str, Any]],
    assessment: SeverityAssessment,
    morphology: MorphologyFeatures | None,
    quality_report: QualityReport | None,
    uncertainty_report: UncertaintyReport | None,
    attention: dict[str, float] | None = None,
    model_is_trained: bool = True,
    preprocessing_steps: list[str] | None = None,
) -> ClinicalNarrative:
    """Assemble the narrative from the measurements already computed."""
    top = predictions[0]
    lesion = LESION_CLASSES[top["code"]]
    confidence = float(top["probability"])

    # ------------------------------------------------------------ impression
    non_skin_input = quality_report is not None and not quality_report.is_skin_like
    if not model_is_trained and non_skin_input:
        impression = (
            "Automated classification unavailable: this image does not appear to be "
            "a dermoscopic photograph of skin, so the classifier's output was "
            "excluded rather than shown as a diagnosis. Retake with a genuine "
            "close-up image of the lesion."
        )
    elif not model_is_trained:
        impression = (
            "Automated classification unavailable (model has no trained weights). "
            f"Morphometric assessment alone: {assessment.tier.replace('_', ' ').lower()}."
        )
    else:
        impression = (
            f"{lesion.name} favoured at {confidence * 100:.1f}% confidence. "
            f"Severity grade: {assessment.tier} ({assessment.score:.0f}/100). "
            f"{assessment.recommendation}, {assessment.timeframe}."
        )

    # --------------------------------------------------------------- summary
    malignant_pct = assessment.malignancy_probability * 100
    summary_parts = []
    if model_is_trained:
        summary_parts.append(
            f"The classifier ranks {lesion.name} highest at {confidence * 100:.1f}%, "
            f"with a combined malignant or premalignant probability of {malignant_pct:.1f}%."
        )
    elif non_skin_input:
        summary_parts.append(
            "This image does not appear to be a dermoscopic photograph of skin, so "
            "the classifier's probabilities carry no diagnostic meaning and are not "
            "shown as a diagnosis. Any geometric and colour measurements below come "
            "from whatever the segmenter found in the frame and should be treated "
            "with the same caution."
        )
    else:
        summary_parts.append(
            "No trained checkpoint was loaded, so the class probabilities below are "
            "the output of an untrained network and carry no diagnostic meaning. "
            "The geometric and colour measurements are unaffected and are reported "
            "on their own merits."
        )
    if morphology is not None:
        abcd = morphology.abcd
        summary_parts.append(
            f"ABCD morphometry gives a total dermoscopy score of {abcd.tds:.2f}, "
            f"which falls in the '{abcd.interpretation.replace('_', ' ')}' band."
        )
    if uncertainty_report is not None and model_is_trained:
        summary_parts.append(
            f"The prediction is {uncertainty_report.verdict}: "
            f"{uncertainty_report.tta_agreement * 100:.0f}% of augmented views of the "
            f"same image agree, with normalised entropy {uncertainty_report.entropy:.2f}."
        )
    summary = " ".join(summary_parts)

    # -------------------------------------------------------------- findings
    findings: list[str] = []
    if morphology is not None:
        findings.extend(
            [
                _describe_asymmetry(morphology),
                _describe_border(morphology),
                _describe_colors(morphology),
                _describe_structures(morphology),
                f"Lesion occupies {morphology.diameter_fraction * 100:.0f}% of the frame "
                f"width (equivalent diameter {morphology.diameter_px:.0f} px); "
                f"lesion-to-skin contrast {morphology.lesion_skin_contrast:.2f}.",
            ]
        )
        if morphology.blue_white_veil > 0.08:
            findings.append(
                f"Blue-white veil covers {morphology.blue_white_veil * 100:.0f}% of the "
                "lesion, a melanoma-specific feature that warrants attention."
            )
    if quality_report is not None:
        findings.append(
            f"Image quality {quality_report.verdict} ({quality_report.score:.0f}/100): "
            f"{quality_report.width}x{quality_report.height} px, focus measure "
            f"{quality_report.sharpness:.0f}, mean brightness {quality_report.brightness:.0f}."
        )
    if preprocessing_steps:
        findings.append("Preprocessing applied: " + "; ".join(preprocessing_steps) + ".")

    # ---------------------------------------------------------- differential
    differential: list[str] = []
    if model_is_trained:
        for entry in predictions[:3]:
            meta = LESION_CLASSES[entry["code"]]
            differential.append(
                f"{meta.name} - {entry['probability'] * 100:.1f}%. {meta.description}"
            )

    # ------------------------------------------------------------ explanation
    explanation: list[str] = []
    if attention:
        inside = attention.get("inside_ratio", 0.0)
        verdict = attention.get("verdict", 0.0)
        if verdict >= 0.5:
            explanation.append(
                f"Grad-CAM places {inside * 100:.0f}% of its activation inside the "
                "segmented lesion, so the prediction is anchored to the lesion itself."
            )
        else:
            explanation.append(
                f"Only {inside * 100:.0f}% of Grad-CAM activation falls inside the "
                "segmented lesion. The prediction may be responding to background "
                "skin, hair or frame artefacts rather than the lesion, and should be "
                "weighted accordingly."
            )
    for driver in assessment.drivers:
        explanation.append(f"{driver.label}: {driver.detail}")
    for override in assessment.overrides_applied:
        explanation.append(f"Grading override - {override}")

    # ------------------------------------------------------------ limitations
    limitations: list[str] = [
        "Trained on HAM10000, which is dominated by lesions from fair-skinned "
        "central-European and Australian populations. Performance on darker skin "
        "tones is not characterised and is likely worse.",
        "Only seven diagnostic categories are modelled. Any lesion outside those "
        "seven - including squamous cell carcinoma, amelanotic melanoma and "
        "infectious or inflammatory conditions - cannot be reported correctly and "
        "will be forced into the nearest of the seven classes.",
        "The D component of the ABCD score is approximated with classical texture "
        "filters, not expert annotation, and should be read as a weak signal.",
    ]
    if quality_report is not None and quality_report.blocking:
        limitations.append(
            "Critical image-quality problems were detected; see the findings above. "
            "The result should not be relied on."
        )
    if morphology is not None and not morphology.reliable:
        limitations.append(
            "Automatic lesion segmentation was unreliable for this image, so all "
            "geometric measurements are approximate."
        )
    if not model_is_trained and non_skin_input:
        limitations.append(
            "This image was judged not to be a dermoscopic photograph of skin, so "
            "classification was withheld rather than reported as a random guess."
        )
    elif not model_is_trained:
        limitations.append(
            "No trained model weights were loaded. Classification output is random."
        )

    # ---------------------------------------------------------- recommendation
    recommendation: list[str] = [
        f"{assessment.recommendation} ({assessment.timeframe}).",
    ]
    if model_is_trained:
        recommendation.append(f"Class-specific guidance: {lesion.management}")
    if assessment.requires_human_review:
        recommendation.append(
            "Flagged for human review: " + "; ".join(assessment.review_reasons)
            if assessment.review_reasons
            else "Flagged for human review because of the assigned severity tier."
        )
    recommendation.append(
        "Advise the patient on the ABCDE self-check rule and photograph the lesion "
        "with a scale marker so future change can be measured objectively."
    )

    return ClinicalNarrative(
        impression=impression,
        summary=summary,
        findings=findings,
        differential=differential,
        explanation=explanation,
        limitations=limitations,
        recommendation=recommendation,
    )


# --------------------------------------------------------------------------- #
# PDF export
# --------------------------------------------------------------------------- #


def _decode_data_uri(uri: str) -> io.BytesIO | None:
    if not uri or "," not in uri:
        return None
    try:
        return io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))
    except Exception:  # noqa: BLE001
        return None


def render_pdf(result: dict[str, Any]) -> bytes:
    """Render an analysis result dictionary into a one-file PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Image as PDFImage,
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Lesion analysis {result.get('case_id', '')}",
        author="derm - explainable lesion classification",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=4,
            textColor=colors.HexColor("#1f3a5f"),
        )
    )
    styles.add(
        ParagraphStyle("Small", parent=styles["Normal"], fontSize=7.5, leading=10,
                       textColor=colors.HexColor("#555555"))
    )

    story: list[Any] = []
    severity_data = result.get("severity", {})
    prediction = result.get("prediction", {})
    tier_color = colors.HexColor(severity_data.get("color", "#7f8c8d"))

    story.append(Paragraph("Dermoscopic Lesion Analysis Report", styles["Title"]))
    story.append(
        Paragraph(
            f"Case {result.get('case_id', 'n/a')} &nbsp;|&nbsp; generated "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
            f"source file: {result.get('filename') or 'uploaded image'}",
            styles["Small"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 8))

    # --- headline banner ---------------------------------------------------- #
    banner = Table(
        [
            [
                Paragraph(
                    f"<b>{severity_data.get('tier', 'n/a')}</b>",
                    ParagraphStyle("Tier", parent=styles["Normal"], fontSize=16,
                                   textColor=colors.white),
                ),
                Paragraph(
                    f"<b>{severity_data.get('headline', '')}</b><br/>"
                    f"{severity_data.get('recommendation', '')} "
                    f"({severity_data.get('timeframe', '')})",
                    ParagraphStyle("BannerText", parent=styles["Normal"], fontSize=9,
                                   leading=12, textColor=colors.white),
                ),
            ]
        ],
        colWidths=[32 * mm, None],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), tier_color),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(banner)
    story.append(Spacer(1, 10))

    # --- model status warning ---------------------------------------------- #
    model_info = result.get("model", {})
    neural_usable = severity_data.get("neural_usable", model_info.get("is_trained", False))
    if not neural_usable:
        non_skin = not (result.get("quality") or {}).get("is_skin_like", True)
        warning_text = (
            "<b>Warning:</b> this image does not appear to be a dermoscopic "
            "photograph of skin. The class probabilities in this report are "
            "meaningless and are not a diagnosis."
            if non_skin and model_info.get("is_trained", False)
            else "<b>Warning:</b> no trained model weights were loaded. The class "
            "probabilities in this report are the output of an untrained "
            "network and have no diagnostic meaning."
        )
        story.append(
            Table(
                [[Paragraph(
                    warning_text,
                    ParagraphStyle("Warn", parent=styles["Body"],
                                   textColor=colors.HexColor("#7a3b00")),
                )]],
                colWidths=[None],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff3cd")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0a800")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]),
            )
        )
        story.append(Spacer(1, 8))

    # --- images ------------------------------------------------------------- #
    images = result.get("images", {})
    panels = [
        ("Submitted image", images.get("original")),
        ("Grad-CAM overlay", images.get("overlay")),
        ("Lesion segmentation", images.get("segmentation")),
    ]
    cells, captions = [], []
    for caption, uri in panels:
        stream = _decode_data_uri(uri) if uri else None
        if stream is None:
            continue
        cells.append(PDFImage(stream, width=52 * mm, height=52 * mm, kind="proportional"))
        captions.append(Paragraph(caption, styles["Small"]))
    if cells:
        table = Table([cells, captions], colWidths=[56 * mm] * len(cells))
        table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    # --- probabilities ------------------------------------------------------ #
    story.append(Paragraph("Class probabilities", styles["SectionHeading"]))
    rows = [["Diagnosis", "Code", "Class", "Probability"]]
    for entry in result.get("probabilities", []):
        rows.append(
            [
                entry["name"],
                entry["code"],
                entry["malignancy"],
                f"{entry['percentage']:.2f}%",
            ]
        )
    probability_table = Table(rows, colWidths=[72 * mm, 18 * mm, 30 * mm, 26 * mm])
    probability_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f5f7fa")]),
                ("ALIGN", (3, 1), (3, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(probability_table)

    # --- ABCD table --------------------------------------------------------- #
    morphology = result.get("morphology")
    if morphology:
        abcd = morphology["abcd"]
        story.append(Paragraph("ABCD morphometry", styles["SectionHeading"]))
        abcd_rows = [
            ["Component", "Score", "Max", "Weight"],
            ["A - Asymmetry", abcd["asymmetry"], 2, "1.3"],
            ["B - Border", abcd["border"], 8, "0.1"],
            ["C - Colour", abcd["colors"], 6, "0.5"],
            ["D - Structures", abcd["structures"], 5, "0.5"],
            ["Total dermoscopy score", f"{abcd['tds']:.2f}", abcd["tds_max"],
             abcd["interpretation"].replace("_", " ")],
        ]
        abcd_table = Table(abcd_rows, colWidths=[62 * mm, 24 * mm, 24 * mm, 36 * mm])
        abcd_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eef2f7")),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(abcd_table)

    story.append(PageBreak())

    # --- narrative ---------------------------------------------------------- #
    narrative = result.get("narrative") or {}
    story.append(Paragraph("Impression", styles["SectionHeading"]))
    story.append(Paragraph(narrative.get("impression", prediction.get("name", "")), styles["Body"]))
    if narrative.get("summary"):
        story.append(Spacer(1, 4))
        story.append(Paragraph(narrative["summary"], styles["Body"]))

    for title, key in (
        ("Findings", "findings"),
        ("Differential", "differential"),
        ("Basis for the prediction", "explanation"),
        ("Recommendation", "recommendation"),
        ("Limitations", "limitations"),
    ):
        items = narrative.get(key) or []
        if not items:
            continue
        story.append(Paragraph(title, styles["SectionHeading"]))
        story.append(
            ListFlowable(
                [ListItem(Paragraph(item, styles["Body"]), leftIndent=10) for item in items],
                bulletType="bullet",
                bulletFontSize=6,
                leftIndent=10,
            )
        )

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"<b>Disclaimer.</b> {narrative.get('disclaimer', MEDICAL_DISCLAIMER)}",
            styles["Small"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Model: {model_info.get('architecture', 'n/a')} at "
            f"{model_info.get('image_size', '?')}px, weights "
            f"'{model_info.get('weights_status', 'unknown')}', device "
            f"{model_info.get('device', 'n/a')}. Total analysis time "
            f"{result.get('timings_ms', {}).get('total', 0):.0f} ms.",
            styles["Small"],
        )
    )

    document.build(story)
    return buffer.getvalue()


__all__ = ["ClinicalNarrative", "generate_narrative", "render_pdf"]
