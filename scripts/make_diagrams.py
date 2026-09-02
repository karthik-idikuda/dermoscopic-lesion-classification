#!/usr/bin/env python3
"""Generate the architecture, workflow and database diagrams used by the
Review-2 deck and the project report.

    python scripts/make_diagrams.py

Writes into ``docs/figures/``:

    architecture.png     four-tier component/interaction diagram
    workflow.png         the nine-stage analysis pipeline
    er_diagram.png       SQLite case-store schema
    module_progress.png  per-module completion chart

Everything is drawn with matplotlib so the figures regenerate deterministically
and no binary assets need to live in version control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
FIGURES = DOCS / "figures"

# A restrained clinical palette, consistent across all four figures.
INK = "#101823"
MUTED = "#5b6a7f"
TEAL = "#0d9488"
TEAL_BG = "#e3f5f3"
AMBER = "#b45309"
AMBER_BG = "#fdf3e3"
ROSE = "#be123c"
ROSE_BG = "#fdeaee"
SLATE = "#334155"
SLATE_BG = "#eef2f7"
LINE = "#c3cddb"

FONT = {"family": "DejaVu Sans"}


def _box(ax, x, y, w, h, *, face, edge, radius=0.02):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.3,
        )
    )


def _text(ax, x, y, s, *, size=8, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, s, size=size, color=color, weight=weight, ha=ha, va=va, **FONT)


def _arrow(ax, xy_from, xy_to, *, color=MUTED, style="-|>", rad=0.0, lw=1.2):
    ax.annotate(
        "",
        xy=xy_to,
        xytext=xy_from,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "linewidth": lw,
            "connectionstyle": f"arc3,rad={rad}",
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def _canvas(w=13.0, h=7.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {path.relative_to(DOCS.parent)}")
    return path


# --------------------------------------------------------------------------- #
# 1. Architecture / component interaction
# --------------------------------------------------------------------------- #


def architecture() -> None:
    fig, ax = _canvas(13.0, 7.6)

    _text(ax, 50, 97, "Explainable Dermoscopic Lesion Analysis — System Architecture",
          size=13, weight="bold")
    _text(ax, 50, 93, "Four tiers. The browser holds no logic beyond rendering; every "
                      "measurement is produced server-side and is reproducible.",
          size=7.5, color=MUTED)

    tiers = [
        ("PRESENTATION", 74, 15, TEAL_BG, TEAL),
        ("SERVICE / API", 55, 15, AMBER_BG, AMBER),
        ("DOMAIN LOGIC  ·  src/derm", 27, 24, SLATE_BG, SLATE),
        ("PERSISTENCE & ARTEFACTS", 6, 14, ROSE_BG, ROSE),
    ]
    for label, y, h, face, edge in tiers:
        _box(ax, 3, y, 94, h, face=face, edge=edge)
        _text(ax, 5, y + h - 2.6, label, size=7, color=edge, weight="bold", ha="left")

    # -- presentation tier
    views = [
        "Analysis\nstudio", "Batch\ntriage", "Change\ntracking",
        "Case\nhistory", "Model\nmetrics", "Method\n/ docs",
    ]
    for i, v in enumerate(views):
        x = 6 + i * 15.0
        _box(ax, x, 76.5, 13.2, 8.6, face="white", edge=TEAL)
        _text(ax, x + 6.6, 80.8, v, size=7.2)
    _text(ax, 50, 74.9, "index.html  ·  styles.css  ·  app.js      (vanilla ES2020, no build step, "
                        "dark/light themes, reduced-motion aware)",
          size=6.8, color=MUTED)

    # -- api tier
    endpoints = [
        "POST /api/analyze", "POST /api/analyze/batch", "POST /api/compare",
        "POST /api/report/pdf", "GET /api/cases*", "GET /api/metrics",
    ]
    for i, e in enumerate(endpoints):
        x = 6 + i * 15.0
        _box(ax, x, 58.6, 13.2, 6.2, face="white", edge=AMBER)
        _text(ax, x + 6.6, 61.7, e, size=6.1)
    _text(ax, 50, 56.6, "FastAPI  ·  app/main.py + app/schemas.py   —   12 endpoints, Pydantic "
                        "validation, optional X-API-Key, CORS, GZip, upload type/size gate",
          size=6.8, color=MUTED)

    # -- domain tier: two rows
    row1 = [
        ("preprocessing", "hair inpainting\ncolour constancy\nvignette crop"),
        ("quality", "focus · exposure\nglare · skin OOD"),
        ("segmentation", "lesion-enhanced\nOtsu + morphology"),
        ("morphology", "Stolz ABCD\nTDS + descriptors"),
        ("model", "EfficientNet-B3\ncheckpoint bundle"),
    ]
    row2 = [
        ("gradcam", "Grad-CAM / ++\nattention align"),
        ("uncertainty", "TTA · MC dropout\nentropy · BALD"),
        ("severity", "composite 0-100\nsafety overrides"),
        ("report", "narrative +\nPDF export"),
        ("monitoring", "longitudinal\nchange tracking"),
    ]
    for row, y in ((row1, 39.6), (row2, 29.4)):
        for i, (name, detail) in enumerate(row):
            x = 6 + i * 18.0
            _box(ax, x, y, 16.4, 9.0, face="white", edge=SLATE)
            _text(ax, x + 8.2, y + 6.7, name, size=7.4, weight="bold", color=SLATE)
            _text(ax, x + 8.2, y + 3.1, detail, size=5.9, color=MUTED)

    _box(ax, 6, 22.4, 88, 5.4, face="white", edge=TEAL)
    _text(ax, 50, 25.1, "inference.py  —  orchestrator: sequences every stage, degrades "
                        "gracefully, returns one AnalysisResult",
          size=7.2, weight="bold", color=TEAL)

    # -- persistence tier
    stores = [
        ("models/best_model.pth", "trained weights\n(absent → untrained mode)"),
        ("data/cases.sqlite3", "case history\n+ thumbnails"),
        ("data/ham10000/", "metadata CSV\n+ images"),
        ("docs/*.json  *.png", "audits, metrics,\nfigures"),
    ]
    for i, (name, detail) in enumerate(stores):
        x = 6 + i * 22.6
        _box(ax, x, 7.0, 20.8, 8.4, face="white", edge=ROSE)
        _text(ax, x + 10.4, 12.6, name, size=6.8, weight="bold", color=ROSE)
        _text(ax, x + 10.4, 9.6, detail, size=6.0, color=MUTED)

    # -- inter-tier flow
    for x in (26, 50, 74):
        _arrow(ax, (x, 76.3), (x, 65.0), color=TEAL, lw=1.4)
        _arrow(ax, (x + 3, 65.0), (x + 3, 76.3), color=MUTED, lw=1.0, style="-|>")
    _arrow(ax, (50, 58.4), (50, 51.2), color=AMBER, lw=1.4)
    _arrow(ax, (50, 22.2), (50, 15.6), color=SLATE, lw=1.4)

    _text(ax, 21.5, 70.5, "HTTP  ·  multipart upload", size=6.4, color=MUTED, ha="right")
    _text(ax, 78.5, 70.5, "JSON  ·  base64 PNG renders", size=6.4, color=MUTED, ha="left")
    _text(ax, 51.5, 54.0, "AnalysisOptions", size=6.4, color=MUTED, ha="left")
    _text(ax, 51.5, 18.9, "read / write", size=6.4, color=MUTED, ha="left")

    _save(fig, "architecture.png")


# --------------------------------------------------------------------------- #
# 2. Workflow / pipeline
# --------------------------------------------------------------------------- #


def workflow() -> None:
    fig, ax = _canvas(13.0, 7.2)

    _text(ax, 50, 97, "Analysis Workflow — nine stages from upload to graded report",
          size=13, weight="bold")
    _text(ax, 50, 92.8, "Stages 1-2 can terminate the run early. Stages 5 and 6 are "
                        "deliberately independent evidence streams.",
          size=7.5, color=MUTED)

    stages = [
        ("1", "Upload &\nvalidation", "type, size, decode\nEXIF orientation", AMBER),
        ("2", "Quality &\nOOD gate", "focus, exposure, glare\nskin chromaticity", AMBER),
        ("3", "Vignette\ncrop", "remove black lens\nbarrel", SLATE),
        ("4", "Restoration", "black-hat hair detect\n+ Telea inpaint\nShades-of-Gray", SLATE),
        ("5", "Segmentation", "lesion-enhanced Otsu\ncentrality-weighted\nellipse fallback", TEAL),
        ("6", "ABCD\nmorphometry", "asymmetry, border,\ncolour, structures\n→ TDS", TEAL),
        ("7", "Classification\n+ uncertainty", "EfficientNet-B3\nTTA, MC dropout\nentropy, BALD", TEAL),
        ("8", "Grad-CAM\nexplanation", "Grad-CAM / ++\nattention alignment\nvs lesion mask", TEAL),
        ("9", "Severity\ngrading", "composite 0-100\none-way overrides\nnarrative + PDF", ROSE),
    ]

    x0, y0, w, h, gap = 3.2, 55.0, 9.6, 26.0, 1.05
    for i, (num, title, detail, colour) in enumerate(stages):
        x = x0 + i * (w + gap)
        bg = {TEAL: TEAL_BG, AMBER: AMBER_BG, ROSE: ROSE_BG, SLATE: SLATE_BG}[colour]
        _box(ax, x, y0, w, h, face=bg, edge=colour)
        _box(ax, x + w / 2 - 2.0, y0 + h - 6.2, 4.0, 4.0, face=colour, edge=colour, radius=0.03)
        _text(ax, x + w / 2, y0 + h - 4.2, num, size=8, color="white", weight="bold")
        _text(ax, x + w / 2, y0 + h - 10.4, title, size=7.3, weight="bold", color=colour)
        _text(ax, x + w / 2, y0 + 5.6, detail, size=5.8, color=MUTED)
        if i < len(stages) - 1:
            _arrow(ax, (x + w, y0 + h / 2), (x + w + gap, y0 + h / 2), color=MUTED, lw=1.1)

    # early-exit branch
    _text(ax, 3.2, 48.0, "Early exit", size=7.4, weight="bold", color=ROSE, ha="left")
    _box(ax, 3.2, 37.5, 32.0, 8.6, face=ROSE_BG, edge=ROSE)
    _text(ax, 19.2, 43.4, "Not skin-like  ·  unusable quality", size=6.9, weight="bold", color=ROSE)
    _text(ax, 19.2, 40.0, "→ tier INDETERMINATE, neural output suppressed", size=6.2, color=MUTED)
    _arrow(ax, (17.0, 55.0), (17.0, 46.3), color=ROSE, lw=1.2)

    # weights branch
    _box(ax, 39.0, 37.5, 27.0, 8.6, face=AMBER_BG, edge=AMBER)
    _text(ax, 52.5, 43.4, "No trained checkpoint", size=6.9, weight="bold", color=AMBER)
    _text(ax, 52.5, 40.0, "→ geometry still valid, class output flagged", size=6.2, color=MUTED)
    _arrow(ax, (70.0, 55.0), (60.0, 46.3), color=AMBER, lw=1.2, rad=-0.15)

    # outputs
    _box(ax, 69.5, 37.5, 27.3, 8.6, face=TEAL_BG, edge=TEAL)
    _text(ax, 83.1, 43.4, "Persisted to case store", size=6.9, weight="bold", color=TEAL)
    _text(ax, 83.1, 40.0, "→ history, comparison, aggregate stats", size=6.2, color=MUTED)
    _arrow(ax, (92.0, 55.0), (92.0, 46.3), color=TEAL, lw=1.2)

    # severity composition
    _text(ax, 50, 31.0, "Severity score composition", size=8.6, weight="bold")
    parts = [
        ("Neural risk", 52, TEAL),
        ("ABCD morphometry", 24, SLATE),
        ("Uncertainty", 16, AMBER),
        ("Image quality", 8, ROSE),
    ]
    bx, bw = 12.0, 76.0
    cursor = bx
    for label, pct, colour in parts:
        seg = bw * pct / 100.0
        _box(ax, cursor, 21.0, seg, 6.4, face=colour, edge=colour, radius=0.01)
        _text(ax, cursor + seg / 2, 24.2, f"{pct}%", size=7.6, color="white", weight="bold")
        _text(ax, cursor + seg / 2, 18.2, label, size=6.3, color=MUTED)
        cursor += seg

    _text(ax, 50, 12.4, "Overrides (one-directional — they raise a tier, never lower one)",
          size=8.0, weight="bold", color=ROSE)
    rules = (
        "melanoma top class → ≥ HIGH  (≥ CRITICAL above 70% confidence)      "
        "melanoma probability ≥ 25% → HIGH      TDS > 5.45 → HIGH      "
        "confidence < 50% → ≥ MODERATE + review flag"
    )
    _text(ax, 50, 8.2, rules, size=6.3, color=MUTED)
    _text(ax, 50, 4.0, "Rationale: the cost of a missed melanoma is not symmetric with the "
                       "cost of an unnecessary referral.",
          size=6.4, color=INK)

    _save(fig, "workflow.png")


# --------------------------------------------------------------------------- #
# 3. Database design
# --------------------------------------------------------------------------- #


def er_diagram() -> None:
    fig, ax = _canvas(12.0, 6.6)

    _text(ax, 50, 96, "Database Design — SQLite case store (data/cases.sqlite3)",
          size=13, weight="bold")
    _text(ax, 50, 91.4, "Single-table design. An analysis is self-contained, so there is nothing "
                        "to normalise into a second table; the full result is kept as JSON for "
                        "replay while the\nqueried fields are promoted to real columns and indexed.",
          size=7.4, color=MUTED)

    columns = [
        ("id", "TEXT", "PK", "UUID4 case identifier"),
        ("created_at", "TEXT", "NOT NULL, IDX", "ISO-8601 UTC timestamp"),
        ("filename", "TEXT", "", "original upload name"),
        ("top_code", "TEXT", "NOT NULL, IDX", "argmax class code (akiec … vasc)"),
        ("top_name", "TEXT", "NOT NULL", "human-readable diagnosis"),
        ("confidence", "REAL", "NOT NULL", "softmax probability of top class"),
        ("tier", "TEXT", "NOT NULL, IDX", "LOW | MODERATE | HIGH | CRITICAL | INDETERMINATE"),
        ("severity_score", "REAL", "NOT NULL", "composite grade, 0-100"),
        ("malignancy_probability", "REAL", "NOT NULL", "summed malignant + premalignant mass"),
        ("tds", "REAL", "", "ABCD total dermoscopy score"),
        ("quality_score", "REAL", "", "image quality, 0-100"),
        ("requires_review", "INTEGER", "NOT NULL DEFAULT 0", "human-review flag (0/1)"),
        ("weights_status", "TEXT", "", "trained | untrained — provenance of the prediction"),
        ("thumbnail", "TEXT", "", "small base64 PNG (full renders deliberately not stored)"),
        ("payload", "TEXT", "NOT NULL", "complete AnalysisResult as JSON, for replay/PDF"),
        ("notes", "TEXT", "", "clinician annotation, editable via PATCH"),
    ]

    tx, tw = 8.0, 84.0
    header_y = 80.0
    row_h = 4.05

    _box(ax, tx, header_y, tw, 5.0, face=SLATE, edge=SLATE)
    for label, dx, ha in (("column", 2.0, "left"), ("type", 30.0, "left"),
                          ("constraints", 44.0, "left"), ("meaning", 66.0, "left")):
        _text(ax, tx + dx, header_y + 2.5, label, size=7.2, color="white",
              weight="bold", ha=ha)

    for i, (name, typ, cons, meaning) in enumerate(columns):
        y = header_y - (i + 1) * row_h
        face = "white" if i % 2 == 0 else SLATE_BG
        _box(ax, tx, y, tw, row_h, face=face, edge=LINE, radius=0.0)
        key = "PK" in cons
        _text(ax, tx + 2.0, y + row_h / 2, name, size=6.8,
              weight="bold" if key else "normal",
              color=TEAL if key else INK, ha="left")
        _text(ax, tx + 30.0, y + row_h / 2, typ, size=6.4, color=MUTED, ha="left")
        _text(ax, tx + 44.0, y + row_h / 2, cons, size=6.4,
              color=ROSE if "IDX" in cons or key else MUTED, ha="left")
        _text(ax, tx + 66.0, y + row_h / 2, meaning, size=6.4, color=MUTED, ha="left")

    _text(ax, 8.0, 9.0, "Indexes", size=7.6, weight="bold", ha="left")
    _text(ax, 8.0, 5.4,
          "idx_cases_created_at (created_at DESC)     idx_cases_tier (tier)     "
          "idx_cases_top_code (top_code)          "
          "— the three axes the history view filters and sorts on.",
          size=6.4, color=MUTED, ha="left")

    _save(fig, "er_diagram.png")


# --------------------------------------------------------------------------- #
# 4. Module progress
# --------------------------------------------------------------------------- #


def module_progress() -> None:
    modules = [
        ("Configuration & taxonomy", 100),
        ("Preprocessing / restoration", 100),
        ("Quality & OOD gating", 100),
        ("Segmentation", 100),
        ("ABCD morphometry", 100),
        ("Grad-CAM explainability", 100),
        ("Uncertainty quantification", 100),
        ("Severity grading", 100),
        ("Clinical report & PDF", 100),
        ("Change tracking", 100),
        ("Case store (SQLite)", 100),
        ("FastAPI backend", 100),
        ("Frontend workstation UI", 100),
        ("Automated test suite", 100),
        ("Data-leakage audit", 100),
        ("Dataset acquisition (10,015 imgs)", 100),
        ("Evaluation & calibration CLI", 100),
        ("Trained weights — linear probe", 100),
        ("End-to-end fine-tune (needs GPU)", 40),
        ("Deployment / packaging", 35),
    ]

    fig, ax = plt.subplots(figsize=(11.0, 6.8), dpi=200)
    fig.patch.set_facecolor("white")

    names = [m[0] for m in modules][::-1]
    values = [m[1] for m in modules][::-1]
    colours = [TEAL if v == 100 else (AMBER if v >= 60 else ROSE) for v in values]

    bars = ax.barh(names, values, color=colours, height=0.66, edgecolor="none")
    for bar, v in zip(bars, values):
        ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2, f"{v}%",
                va="center", size=8, color=INK, weight="bold", **FONT)

    ax.set_xlim(0, 108)
    ax.set_xlabel("completion", size=9, color=MUTED, **FONT)
    ax.set_title("Module-wise Completion at Review-2", size=13, weight="bold",
                 color=INK, pad=14, **FONT)
    ax.tick_params(axis="y", labelsize=8.4, colors=INK, length=0)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    handles = [
        patches.Patch(color=TEAL, label="complete"),
        patches.Patch(color=AMBER, label="in progress"),
        patches.Patch(color=ROSE, label="planned for Review-3"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, "module_progress.png")


def main() -> int:
    print("Generating diagrams into docs/figures/ …")
    architecture()
    workflow()
    er_diagram()
    module_progress()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
