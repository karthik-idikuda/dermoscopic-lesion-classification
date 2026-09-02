"""Central configuration, class taxonomy and clinical metadata.

Every tunable in the project is resolved here so that the notebooks, the CLI
scripts and the FastAPI service all agree on class ordering, image size and
normalisation statistics. Class order matters: it must match the order produced
by ``sklearn.preprocessing.LabelEncoder`` on the HAM10000 ``dx`` column, which
is plain alphabetical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent
SRC_ROOT: Final[Path] = PACKAGE_ROOT.parent
PROJECT_ROOT: Final[Path] = SRC_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


MODELS_DIR: Final[Path] = _env_path("DERM_MODELS_DIR", PROJECT_ROOT / "models")
DOCS_DIR: Final[Path] = _env_path("DERM_DOCS_DIR", PROJECT_ROOT / "docs")
# Generated images live in their own subdirectory so ``docs/`` stays readable:
# machine-readable evidence (*.json) at the top level, figures here, and
# human-facing deliverables under ``docs/review/``.
FIGURES_DIR: Final[Path] = _env_path("DERM_FIGURES_DIR", DOCS_DIR / "figures")
REVIEW_DIR: Final[Path] = _env_path("DERM_REVIEW_DIR", DOCS_DIR / "review")
DATA_DIR: Final[Path] = _env_path("DERM_DATA_DIR", PROJECT_ROOT / "data")
CHECKPOINT_PATH: Final[Path] = _env_path(
    "DERM_CHECKPOINT", MODELS_DIR / "best_model.pth"
)
CASE_DB_PATH: Final[Path] = _env_path("DERM_CASE_DB", DATA_DIR / "cases.sqlite3")

# --------------------------------------------------------------------------- #
# Class taxonomy - alphabetical, matching LabelEncoder output on HAM10000.dx
# --------------------------------------------------------------------------- #

CLASS_CODES: Final[tuple[str, ...]] = (
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
)

NUM_CLASSES: Final[int] = len(CLASS_CODES)
CLASS_INDEX: Final[dict[str, int]] = {c: i for i, c in enumerate(CLASS_CODES)}


@dataclass(frozen=True)
class LesionClass:
    """Clinical descriptor for one dermoscopic diagnosis category."""

    code: str
    name: str
    short_name: str
    malignancy: str  # "malignant" | "premalignant" | "benign"
    base_risk: str  # "high" | "moderate" | "low"
    description: str
    management: str
    ham10000_count: int
    color: str  # hex, used by the UI

    @property
    def is_malignant(self) -> bool:
        return self.malignancy in {"malignant", "premalignant"}


LESION_CLASSES: Final[dict[str, LesionClass]] = {
    "akiec": LesionClass(
        code="akiec",
        name="Actinic Keratosis / Intraepithelial Carcinoma",
        short_name="Actinic Keratosis",
        malignancy="premalignant",
        base_risk="moderate",
        description=(
            "Sun-induced keratinocyte dysplasia (actinic keratosis and Bowen's "
            "disease). Considered a carcinoma in situ that can progress to "
            "invasive squamous cell carcinoma if untreated."
        ),
        management=(
            "Dermatology review. Commonly treated with cryotherapy, topical "
            "field therapy or curettage; biopsy if indurated or ulcerated."
        ),
        ham10000_count=327,
        color="#f39c12",
    ),
    "bcc": LesionClass(
        code="bcc",
        name="Basal Cell Carcinoma",
        short_name="Basal Cell Carcinoma",
        malignancy="malignant",
        base_risk="moderate",
        description=(
            "The most common human cancer. Locally invasive and destructive but "
            "very rarely metastasises. Dermoscopy typically shows arborising "
            "vessels and shiny white structures."
        ),
        management=(
            "Dermatology referral for biopsy and definitive excision or "
            "Mohs surgery depending on site and subtype."
        ),
        ham10000_count=514,
        color="#e67e22",
    ),
    "bkl": LesionClass(
        code="bkl",
        name="Benign Keratosis-like Lesion",
        short_name="Benign Keratosis",
        malignancy="benign",
        base_risk="low",
        description=(
            "Group covering solar lentigo, seborrhoeic keratosis and lichen "
            "planus-like keratosis. Benign, though regressing lesions can mimic "
            "melanoma on dermoscopy."
        ),
        management=(
            "Reassurance and routine monitoring. Removal is cosmetic or for "
            "symptomatic irritation only."
        ),
        ham10000_count=1099,
        color="#16a085",
    ),
    "df": LesionClass(
        code="df",
        name="Dermatofibroma",
        short_name="Dermatofibroma",
        malignancy="benign",
        base_risk="low",
        description=(
            "Benign fibrohistiocytic proliferation, often post-traumatic. "
            "Classic dermoscopy shows a central white scar-like patch with a "
            "delicate peripheral pigment network."
        ),
        management="Reassurance. Excision only if symptomatic or diagnostically unclear.",
        ham10000_count=115,
        color="#27ae60",
    ),
    "mel": LesionClass(
        code="mel",
        name="Melanoma",
        short_name="Melanoma",
        malignancy="malignant",
        base_risk="high",
        description=(
            "Malignant melanocytic tumour with genuine metastatic potential and "
            "the dominant cause of skin cancer mortality. Prognosis depends "
            "steeply on Breslow depth at excision, so early detection matters."
        ),
        management=(
            "Urgent dermatology referral for excision biopsy. Do not delay: "
            "staging and definitive wide local excision follow histology."
        ),
        ham10000_count=1113,
        color="#c0392b",
    ),
    "nv": LesionClass(
        code="nv",
        name="Melanocytic Nevus",
        short_name="Melanocytic Nevus",
        malignancy="benign",
        base_risk="low",
        description=(
            "Ordinary mole: a benign proliferation of melanocytes. By far the "
            "largest class in HAM10000 (67% of images), which is why plain "
            "accuracy is a misleading metric on this dataset."
        ),
        management=(
            "Routine self-monitoring. Re-present if the lesion changes in size, "
            "shape, colour, or begins to itch or bleed."
        ),
        ham10000_count=6705,
        color="#2ecc71",
    ),
    "vasc": LesionClass(
        code="vasc",
        name="Vascular Lesion",
        short_name="Vascular Lesion",
        malignancy="benign",
        base_risk="low",
        description=(
            "Angiomas, angiokeratomas, pyogenic granulomas and haemorrhage. "
            "Benign, with characteristic red-purple lacunae on dermoscopy."
        ),
        management=(
            "Reassurance. Laser or excision for cosmetic reasons; exclude "
            "amelanotic melanoma if the lesion is rapidly growing."
        ),
        ham10000_count=142,
        color="#8e44ad",
    ),
}

MALIGNANT_CODES: Final[frozenset[str]] = frozenset(
    c for c, m in LESION_CLASSES.items() if m.is_malignant
)


def class_name(code_or_index: str | int) -> str:
    """Human readable class name from either a code or a model index."""
    code = CLASS_CODES[code_or_index] if isinstance(code_or_index, int) else code_or_index
    return LESION_CLASSES[code].name


# --------------------------------------------------------------------------- #
# Model / preprocessing settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and preprocessing contract shared by training and serving."""

    architecture: str = os.environ.get("DERM_ARCH", "efficientnet_b3")
    image_size: int = int(os.environ.get("DERM_IMAGE_SIZE", "224"))
    num_classes: int = NUM_CLASSES
    # ImageNet statistics - must match what the checkpoint was trained with.
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # timm module used as the Grad-CAM target; ``conv_head`` is the final 1x1
    # convolution of EfficientNet and gives the sharpest clinically useful maps.
    gradcam_layer: str = os.environ.get("DERM_GRADCAM_LAYER", "conv_head")
    dropout: float = 0.3


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters for :mod:`derm.train`."""

    epochs: int = 15
    batch_size: int = 32
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    num_workers: int = 2
    seed: int = 42
    val_size: float = 0.15
    test_size: float = 0.15
    early_stopping_patience: int = 5
    # Select the checkpoint on macro-F1 rather than accuracy: with 67% `nv` the
    # accuracy-optimal epoch is often the one that ignores the rare classes.
    monitor: str = "macro_f1"
    use_focal_loss: bool = False
    mixup_alpha: float = 0.0


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime switches for the analysis pipeline."""

    tta_transforms: int = int(os.environ.get("DERM_TTA", "5"))
    mc_dropout_passes: int = int(os.environ.get("DERM_MC_PASSES", "10"))
    # Softmax confidence under which a case is escalated for human review.
    low_confidence_threshold: float = 0.50
    high_confidence_threshold: float = 0.70
    # Predictive-entropy (normalised 0-1) above which the model is "unsure".
    high_entropy_threshold: float = 0.55
    max_upload_bytes: int = 12 * 1024 * 1024
    max_batch_size: int = 24
    temperature: float = 1.0  # overwritten by a calibrated checkpoint


@dataclass(frozen=True)
class Settings:
    """Top level settings object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    checkpoint_path: Path = CHECKPOINT_PATH
    device: str = os.environ.get("DERM_DEVICE", "auto")


SETTINGS: Final[Settings] = Settings()

MEDICAL_DISCLAIMER: Final[str] = (
    "This tool is a research prototype for educational use. It is not a medical "
    "device, has not been clinically validated, and must never be used to "
    "diagnose, treat or rule out disease. Any skin lesion that is new, changing, "
    "bleeding, itching or otherwise concerning should be assessed in person by a "
    "qualified clinician regardless of what this software reports."
)

__all__ = [
    "CLASS_CODES",
    "CLASS_INDEX",
    "NUM_CLASSES",
    "LESION_CLASSES",
    "MALIGNANT_CODES",
    "LesionClass",
    "ModelConfig",
    "TrainConfig",
    "InferenceConfig",
    "Settings",
    "SETTINGS",
    "MEDICAL_DISCLAIMER",
    "PROJECT_ROOT",
    "MODELS_DIR",
    "DOCS_DIR",
    "FIGURES_DIR",
    "REVIEW_DIR",
    "DATA_DIR",
    "CHECKPOINT_PATH",
    "CASE_DB_PATH",
    "class_name",
]
