"""FastAPI service for explainable dermoscopic lesion analysis.

Run from the project root:

    uvicorn app.main:app --reload

Security posture, stated explicitly because this handles medical images:

* The API is **unauthenticated by default**, which is appropriate only for a
  local single-user deployment on ``127.0.0.1``. Set ``DERM_API_KEY`` to require
  an ``X-API-Key`` header before exposing this on any network.
* CORS defaults to localhost origins only. Widen it with ``DERM_CORS_ORIGINS``.
* Uploads are size- and type-checked before decoding, and the case database
  stores thumbnails rather than full-resolution images.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Allow `uvicorn app.main:app` from the project root without installing the
# package first: src/ is prepended so `import derm` resolves.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import (  # noqa: E402
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from derm import __version__  # noqa: E402
from derm import monitoring, report as report_module, store  # noqa: E402
from derm.config import (  # noqa: E402
    CLASS_CODES,
    DOCS_DIR,
    FIGURES_DIR,
    LESION_CLASSES,
    MEDICAL_DISCLAIMER,
    SETTINGS,
)
from derm.inference import AnalysisOptions, analyze_image  # noqa: E402
from derm.model import get_bundle  # noqa: E402
from derm.severity import TIER_GUIDANCE  # noqa: E402

from .schemas import AnalyzeOptions, CompareRequest, NotesUpdate  # noqa: E402

logger = logging.getLogger("derm.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
API_KEY = os.environ.get("DERM_API_KEY")


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model at startup so the first request is not slow."""
    try:
        bundle = get_bundle()
        logger.info(
            "Model ready: %s on %s (weights: %s)",
            bundle.config.architecture,
            bundle.device,
            bundle.weights_status,
        )
        for warning in bundle.warnings:
            logger.warning(warning)
    except Exception:  # noqa: BLE001 - keep serving the UI even if the model fails
        logger.exception("Model initialisation failed; /api/analyze will error.")

    if not API_KEY:
        logger.warning(
            "DERM_API_KEY is not set: the API is unauthenticated. This is fine for "
            "local use on 127.0.0.1, but do not expose this port on a network."
        )
    yield


app = FastAPI(
    title="Explainable Dermoscopic Lesion Analysis",
    description=(
        "EfficientNet-B3 classification with Grad-CAM explanations, ABCD "
        "morphometry, uncertainty quantification and automated severity grading. "
        "Research prototype - not a medical device."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "DERM_CORS_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Auth + upload validation
# --------------------------------------------------------------------------- #


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce ``X-API-Key`` when ``DERM_API_KEY`` is configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key."
        )


async def read_upload(upload: UploadFile) -> bytes:
    """Validate and read an uploaded image."""
    suffix = Path(upload.filename or "").suffix.lower()
    content_type = (upload.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES and suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{content_type or suffix or 'unknown'}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
            ),
        )

    limit = SETTINGS.inference.max_upload_bytes
    data = await upload.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is {len(data) / 1e6:.1f} MB; the limit is {limit / 1e6:.0f} MB.",
        )
    return data


def parse_options(raw: str | None) -> tuple[AnalysisOptions, bool]:
    """Parse the ``options`` form field into pipeline options plus persist flag."""
    if not raw:
        model = AnalyzeOptions()
    else:
        try:
            model = AnalyzeOptions.model_validate(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"`options` is not valid JSON: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - pydantic validation error
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    data = model.model_dump()
    persist = bool(data.pop("persist", True))
    return AnalysisOptions.from_mapping(data), persist


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    """Turn decoding/validation ValueErrors into clean 400s."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# --------------------------------------------------------------------------- #
# Metadata endpoints
# --------------------------------------------------------------------------- #


@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, Any]:
    try:
        bundle = get_bundle()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "degraded",
            "version": __version__,
            "model_loaded": False,
            "weights_status": "error",
            "device": "unknown",
            "warnings": [str(exc)],
        }
    return {
        "status": "ok" if bundle.is_trained else "degraded",
        "version": __version__,
        "model_loaded": True,
        "weights_status": bundle.weights_status,
        "device": str(bundle.device),
        "warnings": bundle.warnings,
    }


@app.get("/api/meta", tags=["meta"])
async def meta() -> dict[str, Any]:
    """Class taxonomy, model status, tier guidance and request limits."""
    try:
        model_info = get_bundle().describe()
    except Exception as exc:  # noqa: BLE001
        model_info = {"error": str(exc), "is_trained": False, "weights_status": "error"}

    return {
        "version": __version__,
        "disclaimer": MEDICAL_DISCLAIMER,
        "model": model_info,
        "classes": [
            {
                "code": code,
                "name": LESION_CLASSES[code].name,
                "short_name": LESION_CLASSES[code].short_name,
                "malignancy": LESION_CLASSES[code].malignancy,
                "base_risk": LESION_CLASSES[code].base_risk,
                "description": LESION_CLASSES[code].description,
                "management": LESION_CLASSES[code].management,
                "ham10000_count": LESION_CLASSES[code].ham10000_count,
                "color": LESION_CLASSES[code].color,
            }
            for code in CLASS_CODES
        ],
        "tiers": TIER_GUIDANCE,
        "limits": {
            "max_upload_bytes": SETTINGS.inference.max_upload_bytes,
            "max_batch_size": SETTINGS.inference.max_batch_size,
            "allowed_types": sorted(ALLOWED_SUFFIXES),
        },
        "auth_required": bool(API_KEY),
    }


@app.get("/api/metrics", tags=["meta"])
async def metrics() -> dict[str, Any]:
    """Training/evaluation artefacts written by the CLI scripts."""

    def load(name: str) -> Any:
        path = DOCS_DIR / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    figures = (
        sorted(p.name for p in FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    )
    return {
        "comparison": load("model_comparison.json"),
        "evaluation": load("evaluation.json"),
        "svm_baseline": load("svm_baseline_results.json"),
        "training_history": load("training_history.json"),
        # Reproducible leakage audit written by scripts/audit_leakage.py. Needs
        # only the metadata CSV, so it is available even without a checkpoint.
        "split_audit": load("split_audit.json"),
        "figures": figures,
        "case_stats": store.stats(),
    }


@app.get("/api/figures/{name}", tags=["meta"])
async def figure(name: str) -> FileResponse:
    """Serve a figure from ``docs/figures/`` with path traversal blocked."""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid figure name.")
    path = (FIGURES_DIR / name).resolve()
    if not path.is_file() or FIGURES_DIR.resolve() not in path.parents:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such figure: {name}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unsupported figure type.")
    return FileResponse(path)


@app.post("/api/model/reload", tags=["meta"], dependencies=[Depends(require_api_key)])
async def reload_model() -> dict[str, Any]:
    """Re-read the checkpoint from disk, e.g. after dropping in new weights."""
    bundle = get_bundle(reload=True)
    return {"reloaded": True, "model": bundle.describe()}


# --------------------------------------------------------------------------- #
# Analysis endpoints
# --------------------------------------------------------------------------- #


@app.post("/api/analyze", tags=["analysis"], dependencies=[Depends(require_api_key)])
async def analyze(
    response: Response,
    file: UploadFile = File(..., description="Dermoscopic image"),
    options: str | None = Form(default=None, description="JSON-encoded AnalyzeOptions"),
) -> dict[str, Any]:
    """Full pipeline on a single image."""
    data = await read_upload(file)
    pipeline_options, persist = parse_options(options)

    try:
        result = analyze_image(
            data, options=pipeline_options, filename=file.filename
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Analysis failed: {exc}"
        ) from exc

    payload = result.to_dict()
    # Analysis responses are specific to uploaded bytes and must never be
    # reused by a browser/proxy. Echoing the digest also makes that identity
    # observable outside the JSON body.
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["X-Input-SHA256"] = payload["source"]["sha256"]
    if persist:
        try:
            store.save_case(payload)
            payload["persisted"] = True
        except Exception as exc:  # noqa: BLE001 - history is not critical
            logger.warning("Could not persist case: %s", exc)
            payload["persisted"] = False
    else:
        payload["persisted"] = False
    return payload


@app.post(
    "/api/analyze/batch", tags=["analysis"], dependencies=[Depends(require_api_key)]
)
async def analyze_batch(
    files: list[UploadFile] = File(...),
    options: str | None = Form(default=None),
) -> dict[str, Any]:
    """Analyse several images, returning per-file success or error.

    One bad file does not fail the batch. Renders are skipped by default to keep
    the response small; the summary carries what a triage list needs.
    """
    limit = SETTINGS.inference.max_batch_size
    if len(files) > limit:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"{len(files)} files exceeds the batch limit of {limit}.",
        )

    pipeline_options, persist = parse_options(options)
    items: list[dict[str, Any]] = []

    for upload in files:
        try:
            data = await read_upload(upload)
            result = analyze_image(
                data, options=pipeline_options, filename=upload.filename
            )
            payload = result.to_dict(include_images=pipeline_options.include_images)
            if persist:
                try:
                    store.save_case(result.to_dict())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not persist batch case: %s", exc)
            items.append(
                {"filename": upload.filename, "ok": True, "error": None, "result": payload}
            )
        except HTTPException as exc:
            items.append(
                {"filename": upload.filename, "ok": False, "error": exc.detail, "result": None}
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Batch item failed: %s", upload.filename)
            items.append(
                {"filename": upload.filename, "ok": False, "error": str(exc), "result": None}
            )

    successful = [item for item in items if item["ok"]]
    tiers: dict[str, int] = {}
    for item in successful:
        tier = item["result"]["severity"]["tier"]
        tiers[tier] = tiers.get(tier, 0) + 1

    flagged = [
        {
            "filename": item["filename"],
            "case_id": item["result"]["case_id"],
            "tier": item["result"]["severity"]["tier"],
            "prediction": item["result"]["prediction"]["short_name"],
            "confidence": item["result"]["prediction"]["percentage"],
        }
        for item in successful
        if item["result"]["severity"]["tier"] in {"HIGH", "CRITICAL"}
    ]

    return {
        "count": len(items),
        "succeeded": len(successful),
        "failed": len(items) - len(successful),
        "tier_distribution": tiers,
        "priority_queue": sorted(
            flagged, key=lambda row: row["confidence"], reverse=True
        ),
        "items": items,
    }


@app.post("/api/compare", tags=["analysis"], dependencies=[Depends(require_api_key)])
async def compare(
    baseline: UploadFile = File(..., description="Earlier capture"),
    followup: UploadFile = File(..., description="Later capture"),
    baseline_date: str | None = Form(default=None),
    followup_date: str | None = Form(default=None),
    frame_width_mm: float | None = Form(default=None),
    include_images: bool = Form(default=True),
) -> dict[str, Any]:
    """Longitudinal comparison of two captures of the same lesion."""
    try:
        request_model = CompareRequest(
            baseline_date=baseline_date,
            followup_date=followup_date,
            frame_width_mm=frame_width_mm,
            include_images=include_images,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    baseline_bytes = await read_upload(baseline)
    followup_bytes = await read_upload(followup)

    try:
        change = monitoring.compare(
            baseline_bytes,
            followup_bytes,
            baseline_date=request_model.baseline_date,
            followup_date=request_model.followup_date,
            frame_width_mm=request_model.frame_width_mm,
            include_images=request_model.include_images,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Comparison failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Comparison failed: {exc}"
        ) from exc

    return {
        "baseline_filename": baseline.filename,
        "followup_filename": followup.filename,
        **change.to_dict(),
        "disclaimer": MEDICAL_DISCLAIMER,
    }


@app.post("/api/report/pdf", tags=["analysis"], dependencies=[Depends(require_api_key)])
async def report_pdf(payload: dict[str, Any]) -> Response:
    """Render a PDF from an analysis payload, or from a stored ``case_id``."""
    result = payload
    if set(payload.keys()) <= {"case_id"} and payload.get("case_id"):
        stored = store.get_case(str(payload["case_id"]))
        if stored is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown case id.")
        result = stored

    if "prediction" not in result or "severity" not in result:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Payload must be a full analysis result or {'case_id': '...'}.",
        )

    try:
        pdf = report_module.render_pdf(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF rendering failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"PDF rendering failed: {exc}"
        ) from exc

    filename = f"lesion-report-{result.get('case_id', 'case')}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------- #
# Case history
# --------------------------------------------------------------------------- #


@app.get("/api/cases", tags=["cases"])
async def list_cases(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    tier: str | None = Query(default=None),
    code: str | None = Query(default=None),
    review_only: bool = Query(default=False),
) -> dict[str, Any]:
    if tier and tier not in TIER_GUIDANCE:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown tier '{tier}'.")
    if code and code not in CLASS_CODES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown class code '{code}'.")
    return store.list_cases(
        limit=limit, offset=offset, tier=tier, code=code, review_only=review_only
    )


@app.get("/api/cases/stats", tags=["cases"])
async def case_stats() -> dict[str, Any]:
    return store.stats()


@app.get("/api/cases/{case_id}", tags=["cases"])
async def get_case(case_id: str) -> dict[str, Any]:
    stored = store.get_case(case_id)
    if stored is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown case id.")
    return stored


@app.patch("/api/cases/{case_id}", tags=["cases"], dependencies=[Depends(require_api_key)])
async def update_case(case_id: str, body: NotesUpdate) -> dict[str, Any]:
    if not store.set_notes(case_id, body.notes):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown case id.")
    return {"case_id": case_id, "notes": body.notes}


@app.delete(
    "/api/cases/{case_id}", tags=["cases"], dependencies=[Depends(require_api_key)]
)
async def delete_case(case_id: str) -> dict[str, Any]:
    if not store.delete_case(case_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown case id.")
    return {"deleted": case_id}


# --------------------------------------------------------------------------- #
# Static frontend (mounted last so it does not shadow /api)
# --------------------------------------------------------------------------- #

if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:  # pragma: no cover

    @app.get("/", include_in_schema=False)
    async def missing_frontend() -> dict[str, str]:
        return {
            "detail": (
                f"Static frontend not found at {STATIC_DIR}. The API is available "
                "under /api and documented at /docs."
            )
        }


__all__ = ["app"]
