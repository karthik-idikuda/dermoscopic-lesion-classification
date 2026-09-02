"""End-to-end tests for the inference pipeline, store, reporting and HTTP API."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from derm import monitoring, report, store
from derm.config import CLASS_CODES
from derm.inference import AnalysisOptions, analyze_image, quick_probabilities


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class TestInferencePipeline:
    @pytest.fixture(scope="class")
    def result(self, bundle, suspicious_bytes):
        return analyze_image(suspicious_bytes, bundle=bundle, filename="lesion.png")

    def test_returns_all_sections(self, result):
        payload = result.to_dict()
        for key in (
            "case_id",
            "created_at",
            "prediction",
            "probabilities",
            "severity",
            "quality",
            "uncertainty",
            "morphology",
            "segmentation",
            "explanation",
            "narrative",
            "images",
            "model",
            "timings_ms",
            "disclaimer",
        ):
            assert key in payload, f"missing '{key}' in the analysis payload"

    def test_probabilities_are_sorted_and_normalised(self, result):
        probabilities = [entry.probability for entry in result.predictions]
        assert probabilities == sorted(probabilities, reverse=True)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)
        assert len(result.predictions) == len(CLASS_CODES)

    def test_payload_is_json_serialisable(self, result):
        json.dumps(result.to_dict())

    def test_renders_expected_images(self, result):
        assert "original" in result.images
        for key, uri in result.images.items():
            assert uri.startswith("data:image/png;base64,"), f"{key} is not a data URI"

    def test_gradcam_and_attention_present(self, result):
        assert result.cam is not None
        assert result.attention is not None
        assert 0.0 <= result.attention["inside_ratio"] <= 1.0

    def test_timings_recorded(self, result):
        assert result.timings_ms["total"] > 0
        assert "classification" in result.timings_ms

    def test_untrained_model_marks_indeterminate(self, result):
        assert result.model_info["is_trained"] is False
        assert result.severity.tier == "INDETERMINATE"
        assert any(
            "trained weights" in reason for reason in result.severity.review_reasons
        ), result.severity.review_reasons

    def test_options_disable_stages(self, bundle, benign_bytes):
        options = AnalysisOptions(
            segmentation=False,
            morphometry=False,
            gradcam=False,
            tta=False,
            mc_dropout=False,
            narrative=False,
            include_images=False,
        )
        result = analyze_image(benign_bytes, options=options, bundle=bundle)
        assert result.morphology is None
        assert result.segmentation is None
        assert result.cam is None
        assert result.narrative is None
        assert result.images == {}
        assert result.predictions  # classification always runs

    def test_options_from_mapping_ignores_unknown_keys(self):
        options = AnalysisOptions.from_mapping({"gradcam": False, "nonsense": 1})
        assert options.gradcam is False

    def test_rejects_invalid_bytes(self, bundle):
        with pytest.raises(ValueError):
            analyze_image(b"not an image", bundle=bundle)

    def test_non_skin_input_is_flagged(self, bundle, non_skin_bytes):
        result = analyze_image(non_skin_bytes, bundle=bundle)
        assert not result.quality.is_skin_like
        assert result.severity.tier == "INDETERMINATE"

    def test_quick_probabilities(self, bundle, benign_bytes):
        probabilities = quick_probabilities(benign_bytes, bundle=bundle)
        assert set(probabilities) == set(CLASS_CODES)
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


class TestReporting:
    @pytest.fixture(scope="class")
    def payload(self, bundle, suspicious_bytes):
        return analyze_image(suspicious_bytes, bundle=bundle).to_dict()

    def test_narrative_sections(self, payload):
        narrative = payload["narrative"]
        assert narrative["impression"]
        assert narrative["summary"]
        assert narrative["findings"]
        assert narrative["limitations"]
        assert narrative["recommendation"]
        assert narrative["disclaimer"]

    def test_narrative_warns_about_untrained_weights(self, payload):
        combined = " ".join(payload["narrative"]["limitations"])
        assert "trained" in combined.lower()

    def test_pdf_renders(self, payload):
        pdf = report.render_pdf(payload)
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 5000

    def test_pdf_survives_missing_optional_sections(self, payload):
        stripped = {
            **payload,
            "morphology": None,
            "narrative": None,
            "images": {},
        }
        assert report.render_pdf(stripped).startswith(b"%PDF")

    def test_narrative_to_text(self, payload):
        from derm.report import ClinicalNarrative

        narrative = ClinicalNarrative(**payload["narrative"])
        text = narrative.to_text()
        assert "IMPRESSION" in text
        assert "DISCLAIMER" in text


# --------------------------------------------------------------------------- #
# Change tracking
# --------------------------------------------------------------------------- #


class TestMonitoring:
    def test_identical_images_are_stable(self, benign_bytes):
        change = monitoring.compare(benign_bytes, benign_bytes, include_images=False)
        assert change.verdict == "stable"
        assert change.change_score < 20
        assert change.structural_similarity > 0.95

    def test_growth_is_detected(self, make_lesion, encode_image):
        small = encode_image(make_lesion(radius_fraction=0.15, seed=9))
        large = encode_image(make_lesion(radius_fraction=0.34, seed=9))
        change = monitoring.compare(small, large, include_images=False)

        diameter = next(m for m in change.metrics if m.name.startswith("Diameter (fraction"))
        assert diameter.relative_change > 0.2
        assert diameter.significant
        assert change.change_score > 20

    def test_dates_produce_growth_rate(self, benign_bytes, make_lesion, encode_image):
        later = encode_image(make_lesion(radius_fraction=0.34, seed=1))
        change = monitoring.compare(
            benign_bytes,
            later,
            baseline_date="2025-01-01",
            followup_date="2025-04-01",
            include_images=False,
        )
        assert change.days_between == 90
        assert change.growth_per_month is not None

    def test_reversed_dates_are_rejected(self, benign_bytes):
        change = monitoring.compare(
            benign_bytes,
            benign_bytes,
            baseline_date="2025-04-01",
            followup_date="2025-01-01",
            include_images=False,
        )
        assert change.days_between is None

    def test_field_of_view_enables_mm(self, benign_bytes):
        change = monitoring.compare(
            benign_bytes, benign_bytes, frame_width_mm=20.0, include_images=False
        )
        assert change.metrics[0].name.startswith("Diameter (mm")

    def test_images_are_rendered_when_requested(self, benign_bytes):
        change = monitoring.compare(benign_bytes, benign_bytes, include_images=True)
        assert {"baseline", "followup", "difference"} <= set(change.images)

    def test_serialises(self, benign_bytes):
        json.dumps(
            monitoring.compare(benign_bytes, benign_bytes, include_images=False).to_dict()
        )


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class TestStore:
    @pytest.fixture
    def sample(self, bundle, benign_bytes):
        return analyze_image(benign_bytes, bundle=bundle, filename="a.png").to_dict()

    def test_save_and_get(self, sample, temp_db):
        case_id = store.save_case(sample, path=temp_db)
        assert case_id == sample["case_id"]

        loaded = store.get_case(case_id, path=temp_db)
        assert loaded is not None
        assert loaded["prediction"]["code"] == sample["prediction"]["code"]

    def test_full_images_are_not_persisted(self, sample, temp_db):
        store.save_case(sample, path=temp_db)
        loaded = store.get_case(sample["case_id"], path=temp_db)
        # Only the thumbnail survives, not the full render set.
        assert set(loaded["images"]) <= {"original"}

    def test_list_and_filter(self, sample, temp_db):
        store.save_case(sample, path=temp_db)
        page = store.list_cases(path=temp_db)
        assert page["total"] == 1
        assert page["items"][0]["id"] == sample["case_id"]

        tier = sample["severity"]["tier"]
        assert store.list_cases(tier=tier, path=temp_db)["total"] == 1
        assert store.list_cases(tier="LOW" if tier != "LOW" else "HIGH", path=temp_db)["total"] == 0

    def test_notes_roundtrip(self, sample, temp_db):
        store.save_case(sample, path=temp_db)
        assert store.set_notes(sample["case_id"], "biopsy booked", path=temp_db)
        assert store.get_case(sample["case_id"], path=temp_db)["notes"] == "biopsy booked"
        assert not store.set_notes("does-not-exist", "x", path=temp_db)

    def test_delete(self, sample, temp_db):
        store.save_case(sample, path=temp_db)
        assert store.delete_case(sample["case_id"], path=temp_db)
        assert not store.delete_case(sample["case_id"], path=temp_db)
        assert store.get_case(sample["case_id"], path=temp_db) is None

    def test_stats(self, sample, temp_db):
        store.save_case(sample, path=temp_db)
        stats = store.stats(path=temp_db)
        assert stats["total"] == 1
        assert stats["by_tier"]
        assert stats["by_class"]

    def test_save_is_idempotent(self, sample, temp_db):
        store.save_case(sample, path=temp_db)
        store.save_case(sample, path=temp_db)
        assert store.list_cases(path=temp_db)["total"] == 1

    def test_empty_stats(self, temp_db):
        stats = store.stats(path=temp_db)
        assert stats["total"] == 0
        assert stats["by_tier"] == []


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_session=None):
    """TestClient with the case database redirected to a temp file."""
    import derm.config as config_module

    temp_db = tmp_path_factory.mktemp("api") / "cases.sqlite3"
    original = config_module.CASE_DB_PATH
    config_module.CASE_DB_PATH = temp_db

    import derm.store as store_module

    store_module.CASE_DB_PATH = temp_db

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client

    config_module.CASE_DB_PATH = original
    store_module.CASE_DB_PATH = original


class TestAPI:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["model_loaded"] is True
        # No checkpoint in the test environment, so the service reports degraded.
        assert body["status"] in {"ok", "degraded"}

    def test_meta(self, client):
        body = client.get("/api/meta").json()
        assert len(body["classes"]) == len(CLASS_CODES)
        assert body["disclaimer"]
        assert "max_upload_bytes" in body["limits"]
        assert set(body["tiers"]) >= {"LOW", "MODERATE", "HIGH", "CRITICAL"}

    def test_metrics(self, client):
        body = client.get("/api/metrics").json()
        assert "comparison" in body
        assert "figures" in body
        assert "case_stats" in body

    def test_analyze(self, client, suspicious_bytes):
        response = client.post(
            "/api/analyze",
            files={"file": ("lesion.png", suspicious_bytes, "image/png")},
            data={"options": json.dumps({"persist": True})},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["prediction"]["code"] in CLASS_CODES
        assert body["severity"]["tier"]
        assert body["images"]["original"].startswith("data:image/png")
        assert body["persisted"] is True

    def test_analyze_without_persisting(self, client, benign_bytes):
        response = client.post(
            "/api/analyze",
            files={"file": ("b.png", benign_bytes, "image/png")},
            data={"options": json.dumps({"persist": False})},
        )
        assert response.json()["persisted"] is False

    def test_analyze_rejects_missing_file(self, client):
        assert client.post("/api/analyze").status_code == 422

    def test_analyze_rejects_wrong_type(self, client):
        response = client.post(
            "/api/analyze",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 415
        assert "Unsupported file type" in response.json()["detail"]

    def test_analyze_rejects_empty_file(self, client):
        response = client.post(
            "/api/analyze", files={"file": ("empty.png", b"", "image/png")}
        )
        assert response.status_code == 400

    def test_analyze_rejects_corrupt_image(self, client):
        response = client.post(
            "/api/analyze",
            files={"file": ("fake.png", b"\x89PNG not really", "image/png")},
        )
        assert response.status_code == 400

    def test_analyze_rejects_bad_options_json(self, client, benign_bytes):
        response = client.post(
            "/api/analyze",
            files={"file": ("b.png", benign_bytes, "image/png")},
            data={"options": "{not json"},
        )
        assert response.status_code == 400
        assert "not valid JSON" in response.json()["detail"]

    def test_batch(self, client, benign_bytes, suspicious_bytes):
        response = client.post(
            "/api/analyze/batch",
            files=[
                ("files", ("a.png", benign_bytes, "image/png")),
                ("files", ("b.png", suspicious_bytes, "image/png")),
                ("files", ("bad.txt", b"nope", "text/plain")),
            ],
            data={"options": json.dumps({"include_images": False, "persist": False})},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 3
        assert body["succeeded"] == 2
        assert body["failed"] == 1
        assert "priority_queue" in body
        # A single bad file must not fail the whole batch.
        assert any(item["ok"] is False for item in body["items"])

    def test_compare(self, client, benign_bytes):
        response = client.post(
            "/api/compare",
            files=[
                ("baseline", ("a.png", benign_bytes, "image/png")),
                ("followup", ("b.png", benign_bytes, "image/png")),
            ],
            data={
                "baseline_date": "2025-01-01",
                "followup_date": "2025-03-01",
                "include_images": "false",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["verdict"] == "stable"
        assert body["days_between"] == 59
        assert body["metrics"]

    def test_pdf_from_payload(self, client, benign_bytes):
        analysis = client.post(
            "/api/analyze",
            files={"file": ("b.png", benign_bytes, "image/png")},
            data={"options": json.dumps({"persist": True})},
        ).json()

        response = client.post("/api/report/pdf", json=analysis)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")

    def test_pdf_from_case_id(self, client, benign_bytes):
        analysis = client.post(
            "/api/analyze",
            files={"file": ("b.png", benign_bytes, "image/png")},
            data={"options": json.dumps({"persist": True})},
        ).json()

        response = client.post("/api/report/pdf", json={"case_id": analysis["case_id"]})
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")

    def test_pdf_rejects_unknown_case(self, client):
        assert client.post("/api/report/pdf", json={"case_id": "nope"}).status_code == 404

    def test_pdf_rejects_incomplete_payload(self, client):
        assert client.post("/api/report/pdf", json={"foo": "bar"}).status_code == 422

    def test_case_lifecycle(self, client, benign_bytes):
        analysis = client.post(
            "/api/analyze",
            files={"file": ("case.png", benign_bytes, "image/png")},
            data={"options": json.dumps({"persist": True})},
        ).json()
        case_id = analysis["case_id"]

        listing = client.get("/api/cases").json()
        assert any(item["id"] == case_id for item in listing["items"])

        detail = client.get(f"/api/cases/{case_id}")
        assert detail.status_code == 200

        patched = client.patch(f"/api/cases/{case_id}", json={"notes": "reviewed"})
        assert patched.status_code == 200
        assert client.get(f"/api/cases/{case_id}").json()["notes"] == "reviewed"

        assert client.delete(f"/api/cases/{case_id}").status_code == 200
        assert client.get(f"/api/cases/{case_id}").status_code == 404

    def test_unknown_case_404(self, client):
        assert client.get("/api/cases/deadbeef").status_code == 404

    def test_case_filters_validate(self, client):
        assert client.get("/api/cases?tier=NONSENSE").status_code == 400
        assert client.get("/api/cases?code=nonsense").status_code == 400

    def test_case_stats(self, client):
        assert "total" in client.get("/api/cases/stats").json()

    def test_figure_traversal_is_blocked(self, client):
        assert client.get("/api/figures/..%2F..%2Fetc%2Fpasswd").status_code in {400, 404}

    def test_figure_serves_existing_png(self, client):
        figures = client.get("/api/metrics").json()["figures"]
        if not figures:
            pytest.skip("no figures present in docs/")
        response = client.get(f"/api/figures/{figures[0]}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/")

    def test_model_reload(self, client):
        response = client.post("/api/model/reload")
        assert response.status_code == 200
        assert response.json()["reloaded"] is True

    def test_frontend_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Dermoscopic Lesion Analysis" in response.text

    def test_openapi_schema(self, client):
        schema = client.get("/openapi.json").json()
        for path in ("/api/analyze", "/api/compare", "/api/cases", "/api/health"):
            assert path in schema["paths"]
