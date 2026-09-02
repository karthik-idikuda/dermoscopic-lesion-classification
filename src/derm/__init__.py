"""
derm - Explainable dermoscopic skin lesion classification.

A production-oriented toolkit around EfficientNet-B3 that adds:
  * Grad-CAM / Grad-CAM++ visual explanations
  * classical ABCD(E) morphometry from an automatic lesion segmentation
  * uncertainty quantification (test-time augmentation + MC dropout)
  * image quality / out-of-distribution gating
  * composite automated severity grading
  * narrative clinical report generation and lesion change tracking

Nothing in this package should be used for real clinical decision making.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
