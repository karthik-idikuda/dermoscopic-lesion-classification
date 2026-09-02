"""Tests for preprocessing, quality gating, segmentation and morphometry."""

from __future__ import annotations

import numpy as np
import pytest

from derm import morphology, preprocessing, quality, segmentation


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


class TestPreprocessing:
    def test_load_image_rejects_empty_payload(self):
        with pytest.raises(ValueError, match="Empty image"):
            preprocessing.load_image(b"")

    def test_load_image_rejects_garbage(self):
        with pytest.raises(ValueError, match="Unsupported or corrupt"):
            preprocessing.load_image(b"this is definitely not an image")

    def test_load_image_flattens_alpha(self):
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGBA", (32, 32), (255, 0, 0, 128)).save(buffer, format="PNG")
        image = preprocessing.load_image(buffer.getvalue())
        assert image.mode == "RGB"
        assert image.size == (32, 32)

    def test_detect_hair_finds_strands(self, hairy_array):
        mask, ratio = preprocessing.detect_hair(hairy_array)
        assert mask.shape == hairy_array.shape[:2]
        assert ratio > 0, "hair strands should be detected in the hairy fixture"
        assert np.any(mask)

    def test_detect_hair_ignores_clean_image(self, benign_array):
        _, ratio = preprocessing.detect_hair(benign_array)
        assert ratio < 0.05, "a clean lesion should not read as mostly hair"

    def test_remove_hair_changes_only_masked_pixels(self, hairy_array):
        mask, _ = preprocessing.detect_hair(hairy_array)
        restored = preprocessing.remove_hair(hairy_array, mask)
        assert restored.shape == hairy_array.shape
        # Inpainting brightens dark hair towards the surrounding skin.
        assert restored[mask > 0].mean() > hairy_array[mask > 0].mean()

    def test_shades_of_gray_is_stable_on_neutral_input(self):
        neutral = np.full((64, 64, 3), 128, dtype=np.uint8)
        corrected = preprocessing.shades_of_gray(neutral)
        assert np.allclose(corrected, neutral, atol=3)

    def test_shades_of_gray_reduces_color_cast(self):
        cast = np.zeros((64, 64, 3), dtype=np.uint8)
        cast[..., 0] = 200  # heavy red cast
        cast[..., 1] = 110
        cast[..., 2] = 90
        corrected = preprocessing.shades_of_gray(cast).astype(np.float32)
        before = np.std([200.0, 110.0, 90.0])
        after = np.std(corrected.reshape(-1, 3).mean(axis=0))
        assert after < before

    def test_vignette_detection_and_crop(self, benign_array):
        framed = np.zeros((400, 400, 3), dtype=np.uint8)
        resized = benign_array[:240, :240]
        framed[80:320, 80:320] = resized
        assert preprocessing.detect_vignette(framed) > 0.35
        cropped = preprocessing.crop_vignette(framed)
        assert cropped.shape[0] < framed.shape[0]

    def test_encode_png_produces_data_uri(self, benign_array):
        uri = preprocessing.encode_png(benign_array, max_size=64)
        assert uri.startswith("data:image/png;base64,")
        assert len(uri) > 100

    def test_preprocess_reports_its_steps(self, hairy_array):
        result = preprocessing.preprocess(hairy_array)
        assert result.image.shape == hairy_array.shape
        assert result.hair_removed
        assert result.color_constancy_applied
        assert result.summary


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #


class TestQuality:
    def test_good_image_scores_well(self, benign_array):
        report = quality.assess(benign_array)
        assert report.score >= 55
        assert report.is_skin_like
        assert not report.blocking

    def test_tiny_image_is_blocked(self):
        tiny = np.full((32, 32, 3), 200, dtype=np.uint8)
        tiny[..., 1] = 150
        tiny[..., 2] = 130
        report = quality.assess(tiny)
        assert report.blocking
        assert any(issue.code == "resolution_too_low" for issue in report.issues)

    def test_blurred_image_is_flagged(self, benign_array):
        import cv2

        blurred = cv2.GaussianBlur(benign_array, (51, 51), 0)
        report = quality.assess(blurred)
        codes = {issue.code for issue in report.issues}
        assert codes & {"out_of_focus", "soft_focus"}

    def test_grayscale_is_rejected(self, grayscale_bytes):
        array = preprocessing.to_array(preprocessing.load_image(grayscale_bytes))
        report = quality.assess(array)
        assert report.blocking
        assert not report.is_skin_like
        assert any(issue.code == "not_color_image" for issue in report.issues)

    def test_non_skin_is_flagged_out_of_distribution(self, non_skin_bytes):
        array = preprocessing.to_array(preprocessing.load_image(non_skin_bytes))
        report = quality.assess(array)
        assert not report.is_skin_like
        assert report.blocking

    def test_report_serialises(self, benign_array):
        payload = quality.assess(benign_array).to_dict()
        assert {"score", "verdict", "is_skin_like", "metrics", "issues"} <= payload.keys()


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


class TestSegmentation:
    def test_finds_central_lesion(self, benign_array):
        result = segmentation.segment(benign_array)
        assert result.mask.shape == benign_array.shape[:2]
        assert 0.02 < result.area_ratio < 0.9
        assert result.contour is not None

        height, width = benign_array.shape[:2]
        # The synthetic lesion is centred, so the centroid should be too.
        assert abs(result.centroid[0] - width / 2) < width * 0.2
        assert abs(result.centroid[1] - height / 2) < height * 0.2

    def test_uniform_image_falls_back_with_low_confidence(self):
        flat = np.full((256, 256, 3), 180, dtype=np.uint8)
        result = segmentation.segment(flat)
        assert result.confidence < 0.6
        assert result.method in {"adaptive", "fallback_ellipse"}

    def test_larger_lesion_yields_larger_mask(self, make_lesion):
        small = segmentation.segment(make_lesion(radius_fraction=0.15, seed=5))
        large = segmentation.segment(make_lesion(radius_fraction=0.35, seed=5))
        assert large.area_ratio > small.area_ratio

    def test_overlays_do_not_mutate_input(self, benign_array):
        result = segmentation.segment(benign_array)
        original = benign_array.copy()
        segmentation.overlay_contour(benign_array, result)
        segmentation.mask_preview(benign_array, result)
        assert np.array_equal(benign_array, original)

    def test_serialises(self, benign_array):
        payload = segmentation.segment(benign_array).to_dict()
        assert {"area_ratio", "confidence", "method", "reliable"} <= payload.keys()


# --------------------------------------------------------------------------- #
# Morphometry
# --------------------------------------------------------------------------- #


class TestMorphology:
    def test_abcd_ranges_are_respected(self, suspicious_array):
        seg = segmentation.segment(suspicious_array)
        features = morphology.analyze(suspicious_array, seg)
        abcd = features.abcd
        assert 0 <= abcd.asymmetry <= 2
        assert 0 <= abcd.border <= 8
        assert 0 <= abcd.colors <= 6
        assert 0 <= abcd.structures <= 5
        assert 0 <= abcd.tds <= abcd.max_tds

    def test_tds_matches_the_stolz_formula(self, suspicious_array):
        seg = segmentation.segment(suspicious_array)
        abcd = morphology.analyze(suspicious_array, seg).abcd
        expected = (
            1.3 * abcd.asymmetry + 0.1 * abcd.border + 0.5 * abcd.colors + 0.5 * abcd.structures
        )
        assert abcd.tds == pytest.approx(expected, abs=1e-9)

    def test_interpretation_cutpoints(self):
        assert morphology.interpret_tds(3.0) == "benign"
        assert morphology.interpret_tds(4.75) == "benign"
        assert morphology.interpret_tds(5.0) == "suspicious"
        assert morphology.interpret_tds(5.45) == "suspicious"
        assert morphology.interpret_tds(6.0) == "highly_suspicious"

    def test_irregular_lesion_scores_higher_than_round_one(
        self, benign_array, suspicious_array
    ):
        benign = morphology.analyze(benign_array, segmentation.segment(benign_array))
        suspicious = morphology.analyze(
            suspicious_array, segmentation.segment(suspicious_array)
        )
        assert suspicious.abcd.tds > benign.abcd.tds, (
            f"irregular multi-coloured lesion scored {suspicious.abcd.tds:.2f} but the "
            f"round uniform one scored {benign.abcd.tds:.2f}"
        )

    def test_round_lesion_is_more_circular(self, benign_array, suspicious_array):
        benign = morphology.analyze(benign_array, segmentation.segment(benign_array))
        suspicious = morphology.analyze(
            suspicious_array, segmentation.segment(suspicious_array)
        )
        assert benign.circularity > suspicious.circularity

    def test_multicolor_lesion_detects_more_colors(self, benign_array, suspicious_array):
        benign_seg = segmentation.segment(benign_array)
        suspicious_seg = segmentation.segment(suspicious_array)
        benign_count, _ = morphology.color_score(benign_array, benign_seg.mask)
        suspicious_count, _ = morphology.color_score(suspicious_array, suspicious_seg.mask)
        assert suspicious_count >= benign_count

    def test_empty_mask_is_handled(self, benign_array):
        empty = np.zeros(benign_array.shape[:2], dtype=np.uint8)
        assert morphology.asymmetry(empty) == (0, 0.0, 0.0)
        assert morphology.border_score(empty, None) == (0, 0.0)
        assert morphology.color_score(benign_array, empty) == (0, [])
        assert morphology.structure_score(benign_array, empty) == (0, [])
        assert morphology.veil_fraction(benign_array, empty) == 0.0

    def test_serialises(self, benign_array):
        seg = segmentation.segment(benign_array)
        payload = morphology.analyze(benign_array, seg).to_dict()
        assert {"abcd", "shape", "color", "reliable"} <= payload.keys()
        assert "tds" in payload["abcd"]
