"""Shared pytest fixtures.

The tests are designed to run without the HAM10000 dataset and without a trained
checkpoint: every fixture image is synthesised. That keeps the suite fast and
makes it usable in CI, while still exercising the real code paths.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
for path in (str(SRC), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _skin_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """A plausible skin-toned, slightly textured background."""
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 214.0
    base[..., 1] = 168.0
    base[..., 2] = 142.0
    base += rng.normal(0, 6, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def synth_lesion(
    size: int = 320,
    *,
    radius_fraction: float = 0.28,
    irregular: bool = False,
    multicolor: bool = False,
    with_hair: bool = False,
    seed: int = 7,
) -> np.ndarray:
    """Render a synthetic dermoscopic image with a controllable lesion.

    Deliberately simple, but it has the properties the pipeline reads: a darker
    central blob on skin-toned background, optionally with an irregular border,
    several pigment colours, and overlaid hair strands.
    """
    import cv2

    rng = np.random.default_rng(seed)
    image = _skin_background(size, rng)

    centre = size // 2
    radius = int(size * radius_fraction)

    mask = np.zeros((size, size), dtype=np.uint8)
    if irregular:
        angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        wobble = 1.0 + 0.32 * np.sin(5 * angles) + rng.normal(0, 0.09, angles.shape)
        points = np.stack(
            [
                centre + (radius * wobble * np.cos(angles)),
                centre + (radius * wobble * np.sin(angles)),
            ],
            axis=1,
        ).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    else:
        cv2.circle(mask, (centre, centre), radius, 255, -1)

    lesion = image.copy()
    lesion[mask > 0] = (92, 64, 52)  # dark brown

    if multicolor:
        cv2.circle(lesion, (centre - radius // 3, centre), radius // 3, (28, 22, 20), -1)
        cv2.circle(lesion, (centre + radius // 3, centre - radius // 4), radius // 4, (168, 122, 96), -1)
        cv2.circle(lesion, (centre, centre + radius // 2), radius // 5, (196, 196, 205), -1)
        cv2.circle(lesion, (centre + radius // 4, centre + radius // 3), radius // 6, (176, 58, 52), -1)
        lesion[mask == 0] = image[mask == 0]

    lesion = cv2.GaussianBlur(lesion, (5, 5), 0)

    # Real dermoscopic photographs carry fine skin texture, which the focus
    # measure (variance of the Laplacian) depends on. Without it the fixture
    # reads as heavily out of focus and the quality gate correctly rejects it.
    texture = rng.normal(0, 11, lesion.shape)
    lesion = np.clip(lesion.astype(np.float32) + texture, 0, 255).astype(np.uint8)

    if with_hair:
        for _ in range(14):
            start = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            end = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            cv2.line(lesion, start, end, (38, 28, 24), thickness=2)

    return lesion


def encode(array: np.ndarray, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def make_lesion():
    """Expose the synthetic-image generator as a fixture.

    Importing ``tests.conftest`` directly is unreliable: an installed package
    named ``tests`` elsewhere on ``sys.path`` can shadow it.
    """
    return synth_lesion


@pytest.fixture(scope="session")
def encode_image():
    """Expose the PNG encoder as a fixture, for the same reason."""
    return encode


@pytest.fixture(scope="session")
def benign_array() -> np.ndarray:
    """Round, uniformly pigmented lesion - should score low on ABCD."""
    return synth_lesion(irregular=False, multicolor=False, seed=1)


@pytest.fixture(scope="session")
def suspicious_array() -> np.ndarray:
    """Irregular, multi-coloured lesion - should score higher on ABCD."""
    return synth_lesion(irregular=True, multicolor=True, seed=2)


@pytest.fixture(scope="session")
def hairy_array() -> np.ndarray:
    return synth_lesion(irregular=False, multicolor=False, with_hair=True, seed=3)


@pytest.fixture(scope="session")
def benign_bytes(benign_array) -> bytes:
    return encode(benign_array)


@pytest.fixture(scope="session")
def suspicious_bytes(suspicious_array) -> bytes:
    return encode(suspicious_array)


@pytest.fixture(scope="session")
def non_skin_bytes() -> bytes:
    """A blue/grey synthetic pattern that is clearly not skin."""
    rng = np.random.default_rng(11)
    array = np.zeros((256, 256, 3), dtype=np.uint8)
    array[..., 2] = 210  # dominant blue
    array[..., 0] = 20
    array[..., 1] = 40
    array = np.clip(array + rng.normal(0, 10, array.shape), 0, 255).astype(np.uint8)
    return encode(array)


@pytest.fixture(scope="session")
def grayscale_bytes() -> bytes:
    array = np.full((256, 256, 3), 128, dtype=np.uint8)
    return encode(array)


@pytest.fixture(scope="session")
def bundle():
    """A real (untrained) ModelBundle on CPU, built once for the whole session."""
    from derm.model import create_bundle

    return create_bundle(Path("/nonexistent/checkpoint.pth"), device="cpu")


@pytest.fixture
def temp_db(tmp_path) -> Path:
    return tmp_path / "cases.sqlite3"
