"""Tests for the model layer, Grad-CAM, uncertainty and severity grading."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dataclasses import replace

from derm import gradcam, preprocessing, quality, segmentation, severity, uncertainty
from derm.config import CLASS_CODES, NUM_CLASSES, SETTINGS
from derm.model import (
    build_eval_transform,
    build_model,
    create_bundle,
    denormalize,
    load_checkpoint,
    resolve_device,
    resolve_gradcam_layer,
)


class TestModelLayer:
    def test_resolve_device_falls_back_to_cpu(self):
        assert resolve_device("cpu").type == "cpu"
        assert resolve_device("auto").type in {"cpu", "cuda"}

    def test_build_model_has_seven_outputs(self):
        model = build_model()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 224, 224))
        assert logits.shape == (1, NUM_CLASSES)

    def test_missing_checkpoint_degrades_gracefully(self, tmp_path):
        bundle = create_bundle(tmp_path / "nope.pth", device="cpu")
        assert bundle.weights_status == "untrained"
        assert not bundle.is_trained
        assert bundle.warnings, "a missing checkpoint must produce an explicit warning"
        assert "No checkpoint" in bundle.warnings[0]

    def test_roundtrip_checkpoint(self, tmp_path):
        """A checkpoint written in the training format must load as 'trained'."""
        model = build_model()
        path = tmp_path / "best_model.pth"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_codes": list(CLASS_CODES),
                "temperature": 1.42,
                "test_accuracy": 77.5,
            },
            path,
        )

        state, metadata = load_checkpoint(path)
        assert metadata["temperature"] == 1.42
        assert len(state) > 0

        bundle = create_bundle(path, device="cpu")
        assert bundle.is_trained
        assert bundle.temperature == pytest.approx(1.42)
        assert bundle.class_codes == CLASS_CODES

    def test_bare_state_dict_checkpoint_loads(self, tmp_path):
        """The original notebook saved a bare state_dict; that must still work."""
        model = build_model()
        path = tmp_path / "bare.pth"
        torch.save(model.state_dict(), path)
        bundle = create_bundle(path, device="cpu")
        assert bundle.is_trained

    def test_prefixed_keys_are_stripped(self, tmp_path):
        model = build_model()
        prefixed = {f"module.{k}": v for k, v in model.state_dict().items()}
        path = tmp_path / "prefixed.pth"
        torch.save(prefixed, path)
        bundle = create_bundle(path, device="cpu")
        assert bundle.is_trained

    def test_corrupt_checkpoint_does_not_crash(self, tmp_path):
        """A file that cannot even be unpickled must degrade, not raise.

        The warning distinguishes *read* failure (the file is not a checkpoint)
        from *load* failure (it unpickled but does not fit the graph), because
        the two need different fixes.
        """
        path = tmp_path / "corrupt.pth"
        path.write_bytes(b"not a torch file")
        bundle = create_bundle(path, device="cpu")
        assert bundle.weights_status == "untrained"
        assert any("Failed to read" in warning for warning in bundle.warnings)
        # A corrupt file must not be reported as a missing file.
        assert not any("No checkpoint at" in warning for warning in bundle.warnings)

    def test_checkpoint_architecture_overrides_settings(self, tmp_path):
        """A checkpoint declaring a different backbone must be built as declared.

        Otherwise a lighter backbone silently fails to populate the configured
        graph and the model reports "untrained" with no obvious cause.
        """
        light = build_model(replace(SETTINGS.model, architecture="efficientnet_b0"))
        path = tmp_path / "b0.pth"
        torch.save(
            {
                "state_dict": light.state_dict(),
                "architecture": "efficientnet_b0",
                "class_codes": list(CLASS_CODES),
            },
            path,
        )
        bundle = create_bundle(path, device="cpu")
        assert bundle.is_trained
        assert bundle.config.architecture == "efficientnet_b0"
        assert any("declares architecture" in w for w in bundle.warnings)

    def test_probabilities_sum_to_one(self, bundle, benign_bytes):
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        probabilities = bundle.probabilities(batch)[0].numpy()
        assert probabilities.shape == (NUM_CLASSES,)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(probabilities >= 0)

    def test_transform_output_shape(self):
        transform = build_eval_transform()
        from PIL import Image

        tensor = transform(Image.new("RGB", (97, 61), (200, 150, 130)))
        assert tensor.shape == (3, 224, 224)

    def test_denormalize_roundtrip(self):
        from PIL import Image

        original = np.full((224, 224, 3), 150, dtype=np.uint8)
        tensor = build_eval_transform()(Image.fromarray(original))
        recovered = denormalize(tensor)
        assert recovered.shape == (224, 224, 3)
        assert abs(int(recovered.mean()) - 150) <= 2

    def test_gradcam_layer_resolves(self, bundle):
        layer = resolve_gradcam_layer(bundle.model, bundle.config)
        assert isinstance(layer, torch.nn.Module)

    def test_describe_is_json_safe(self, bundle):
        import json

        json.dumps(bundle.describe())


class TestGradCAM:
    @pytest.fixture(scope="class")
    def cam_result(self, bundle, benign_bytes):
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        explainer = gradcam.GradCAM(
            bundle.model, resolve_gradcam_layer(bundle.model, bundle.config)
        )
        return explainer(batch, 0, method="gradcam", output_size=(128, 128))

    def test_shape_and_range(self, cam_result):
        result, logits = cam_result
        assert result.cam.shape == (128, 128)
        assert result.cam.min() >= 0.0
        assert result.cam.max() <= 1.0 + 1e-6
        assert logits.shape == (1, NUM_CLASSES)

    def test_records_target_class(self, cam_result):
        result, _ = cam_result
        assert result.class_index == 0
        assert 0.0 <= result.concentration <= 1.0

    def test_gradcam_plus_plus_runs(self, bundle, benign_bytes):
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        explainer = gradcam.GradCAM(
            bundle.model, resolve_gradcam_layer(bundle.model, bundle.config)
        )
        result, _ = explainer(batch, 4, method="gradcam++")
        assert result.method == "gradcam++"
        assert result.cam.shape == (224, 224)

    def test_hooks_are_removed_after_each_call(self, bundle, benign_bytes):
        """Repeated use must not accumulate hooks - the notebook version leaked."""
        layer = resolve_gradcam_layer(bundle.model, bundle.config)
        before = len(layer._forward_hooks)

        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        explainer = gradcam.GradCAM(bundle.model, layer)
        for _ in range(3):
            explainer(batch, 0)

        assert len(layer._forward_hooks) == before

    def test_works_inside_inference_mode(self, bundle, benign_bytes):
        """FastAPI handlers may already be in inference_mode; backward must still run."""
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        explainer = gradcam.GradCAM(
            bundle.model, resolve_gradcam_layer(bundle.model, bundle.config)
        )
        with torch.inference_mode():
            result, _ = explainer(batch.clone(), 0)
        assert result.cam.shape == (224, 224)

    def test_model_returns_to_eval_mode(self, bundle, benign_bytes):
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        explainer = gradcam.GradCAM(
            bundle.model, resolve_gradcam_layer(bundle.model, bundle.config)
        )
        explainer(batch, 0)
        assert not bundle.model.training

    def test_rejects_multi_image_batch(self, bundle):
        explainer = gradcam.GradCAM(
            bundle.model, resolve_gradcam_layer(bundle.model, bundle.config)
        )
        with pytest.raises(ValueError, match="exactly one image"):
            explainer(torch.randn(2, 3, 224, 224))

    def test_attention_alignment_detects_overlap(self):
        cam = np.zeros((64, 64), dtype=np.float32)
        cam[24:40, 24:40] = 1.0
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 255

        aligned = gradcam.attention_alignment(cam, mask)
        assert aligned["inside_ratio"] > 0.95
        assert aligned["verdict"] > 0.5

        elsewhere = np.zeros((64, 64), dtype=np.uint8)
        elsewhere[0:12, 0:12] = 255
        misaligned = gradcam.attention_alignment(cam, elsewhere)
        assert misaligned["inside_ratio"] < 0.1
        assert misaligned["verdict"] < aligned["verdict"]

    def test_overlay_and_colorize_shapes(self, benign_array):
        cam = np.random.default_rng(0).random(benign_array.shape[:2]).astype(np.float32)
        heatmap = gradcam.colorize(cam)
        assert heatmap.shape == benign_array.shape

        blended = gradcam.overlay(benign_array, cam)
        assert blended.shape == benign_array.shape
        assert blended.dtype == np.uint8

        outlined = gradcam.contour_overlay(benign_array, cam)
        assert outlined.shape == benign_array.shape

    def test_overlay_resizes_mismatched_cam(self, benign_array):
        cam = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        assert gradcam.overlay(benign_array, cam).shape == benign_array.shape


class TestUncertainty:
    def test_entropy_bounds(self):
        one_hot = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        uniform = np.full(7, 1 / 7)
        assert uncertainty.normalized_entropy(one_hot) == pytest.approx(0.0, abs=1e-6)
        assert uncertainty.normalized_entropy(uniform) == pytest.approx(1.0, abs=1e-6)

    def test_margin(self):
        assert uncertainty.top_margin(np.array([0.6, 0.3, 0.1])) == pytest.approx(0.3)
        assert uncertainty.top_margin(np.array([0.4, 0.4, 0.2])) == pytest.approx(0.0)

    def test_verdicts(self):
        assert uncertainty.classify_verdict(0.95, 0.05, 1.0) == "confident"
        assert uncertainty.classify_verdict(0.65, 0.30, 0.9) == "borderline"
        assert uncertainty.classify_verdict(0.30, 0.80, 0.4) == "uncertain"

    def test_tta_averages_over_augmentations(self, bundle, benign_bytes):
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        mean, agreement, spread, argmaxes = uncertainty.tta_predict(bundle, batch, 4)
        assert mean.shape == (NUM_CLASSES,)
        assert mean.sum() == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= agreement <= 1.0
        assert spread >= 0.0
        assert len(argmaxes) == 4

    def test_mc_dropout_restores_eval_mode(self, bundle, benign_bytes):
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        mean, spread, mutual_information, passes = uncertainty.mc_dropout_predict(
            bundle, batch, 3
        )
        assert mean.shape == (NUM_CLASSES,)
        assert spread >= 0.0
        assert mutual_information >= 0.0
        assert passes in {0, 3}
        assert not bundle.model.training, "MC dropout must leave the model in eval mode"

    def test_mc_dropout_is_actually_stochastic(self, bundle, benign_bytes):
        """Regression: timm uses functional dropout, so module scanning finds none.

        If activation silently fails, every pass is identical and the reported
        spread is a meaningless zero. Guard against that.
        """
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        _, spread, _, passes = uncertainty.mc_dropout_predict(bundle, batch, 8)
        assert passes == 8, "dropout should have been activated for efficientnet_b3"
        assert spread > 0.0, "MC dropout passes were identical, so dropout was inert"

    def test_batchnorm_stays_in_eval_during_mc_dropout(self, bundle):
        """Enabling train mode must not let BatchNorm use single-sample statistics."""
        activated = uncertainty._activate_stochasticity(bundle.model)
        try:
            assert activated
            norm_layers = [
                module
                for module in bundle.model.modules()
                if isinstance(module, uncertainty._NORM_TYPES)
            ]
            assert norm_layers, "efficientnet_b3 should contain BatchNorm layers"
            assert all(not layer.training for layer in norm_layers)
        finally:
            bundle.ensure_eval()

    def test_deterministic_passes_are_identical(self, bundle, benign_bytes):
        """Sanity check that normal inference is repeatable."""
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        first = bundle.probabilities(batch)[0].numpy()
        second = bundle.probabilities(batch)[0].numpy()
        assert np.allclose(first, second, atol=1e-6)

    def test_estimate_produces_full_report(self, bundle, benign_bytes):
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        report = uncertainty.estimate(bundle, batch, n_tta=3, mc_passes=2)
        assert report.probabilities.shape == (NUM_CLASSES,)
        assert report.verdict in {"confident", "borderline", "uncertain"}
        assert 0.0 <= report.entropy <= 1.0
        assert report.to_dict()["verdict"] == report.verdict

    def test_disabling_estimators_still_returns_probabilities(self, bundle, benign_bytes):
        image = preprocessing.load_image(benign_bytes)
        batch = bundle.prepare(image)
        report = uncertainty.estimate(bundle, batch, use_tta=False, use_mc_dropout=False)
        assert report.n_tta == 0
        assert report.n_mc == 0
        assert report.probabilities.sum() == pytest.approx(1.0, abs=1e-5)


class TestSeverity:
    @staticmethod
    def probabilities_for(code: str, confidence: float = 0.9) -> np.ndarray:
        array = np.full(NUM_CLASSES, (1.0 - confidence) / (NUM_CLASSES - 1))
        array[CLASS_CODES.index(code)] = confidence
        return array

    def test_confident_nevus_is_low_risk(self):
        assessment = severity.grade(self.probabilities_for("nv", 0.96), CLASS_CODES)
        assert assessment.tier == "LOW"
        assert assessment.score < 30

    def test_confident_melanoma_is_critical(self):
        assessment = severity.grade(self.probabilities_for("mel", 0.93), CLASS_CODES)
        assert assessment.tier == "CRITICAL"
        assert assessment.requires_human_review
        assert any("Melanoma is the top prediction" in o for o in assessment.overrides_applied)

    def test_low_confidence_melanoma_is_at_least_high(self):
        assessment = severity.grade(self.probabilities_for("mel", 0.42), CLASS_CODES)
        assert assessment.tier in {"HIGH", "CRITICAL"}

    def test_hidden_melanoma_probability_escalates(self):
        """Melanoma need not win argmax to trigger the safety net."""
        array = np.zeros(NUM_CLASSES)
        array[CLASS_CODES.index("nv")] = 0.55
        array[CLASS_CODES.index("mel")] = 0.30
        array[CLASS_CODES.index("bkl")] = 0.15
        assessment = severity.grade(array, CLASS_CODES)
        assert assessment.tier in {"HIGH", "CRITICAL"}
        assert any("safety-net threshold" in o for o in assessment.overrides_applied)

    def test_malignant_class_is_never_low(self):
        for code in ("bcc", "akiec"):
            assessment = severity.grade(self.probabilities_for(code, 0.95), CLASS_CODES)
            assert severity.TIER_ORDER[assessment.tier] >= severity.TIER_ORDER["MODERATE"]

    def test_low_confidence_forces_review(self):
        uniform = np.full(NUM_CLASSES, 1 / NUM_CLASSES)
        assessment = severity.grade(uniform, CLASS_CODES)
        assert assessment.requires_human_review
        assert severity.TIER_ORDER[assessment.tier] >= severity.TIER_ORDER["MODERATE"]

    def test_untrained_model_is_indeterminate(self):
        assessment = severity.grade(
            self.probabilities_for("mel", 0.99), CLASS_CODES, model_is_trained=False
        )
        assert assessment.tier == "INDETERMINATE"
        assert assessment.requires_human_review

    def test_non_skin_input_is_indeterminate(self, non_skin_bytes):
        array = preprocessing.to_array(preprocessing.load_image(non_skin_bytes))
        report = quality.assess(array)
        assessment = severity.grade(
            self.probabilities_for("nv", 0.99), CLASS_CODES, quality=report
        )
        assert assessment.tier == "INDETERMINATE"

    def test_high_tds_escalates(self, suspicious_array):
        seg = segmentation.segment(suspicious_array)
        features = morphology_analyze(suspicious_array, seg)
        # Force the TDS above the highly-suspicious cut-point.
        features.abcd.tds = 6.0
        features.abcd.interpretation = "highly_suspicious"
        assessment = severity.grade(
            self.probabilities_for("nv", 0.95), CLASS_CODES, morphology=features
        )
        assert severity.TIER_ORDER[assessment.tier] >= severity.TIER_ORDER["HIGH"]

    def test_overrides_never_lower_the_tier(self):
        """Every override is monotonic upward by construction."""
        assessment = severity.grade(self.probabilities_for("mel", 0.99), CLASS_CODES)
        assert severity.TIER_ORDER[assessment.tier] >= severity.TIER_ORDER["CRITICAL"]

    def test_escalate_helper(self):
        assert severity._escalate("LOW", "HIGH") == "HIGH"
        assert severity._escalate("CRITICAL", "MODERATE") == "CRITICAL"

    def test_malignancy_probability_sums_malignant_classes(self):
        array = np.zeros(NUM_CLASSES)
        array[CLASS_CODES.index("mel")] = 0.4
        array[CLASS_CODES.index("bcc")] = 0.2
        array[CLASS_CODES.index("akiec")] = 0.1
        array[CLASS_CODES.index("nv")] = 0.3
        assessment = severity.grade(array, CLASS_CODES)
        assert assessment.malignancy_probability == pytest.approx(0.7, abs=1e-6)

    def test_serialises(self):
        payload = severity.grade(self.probabilities_for("nv"), CLASS_CODES).to_dict()
        assert {"tier", "score", "headline", "drivers", "components"} <= payload.keys()


def morphology_analyze(array, seg):
    from derm.morphology import analyze

    return analyze(array, seg)
