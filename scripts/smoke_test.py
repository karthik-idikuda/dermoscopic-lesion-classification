"""End-to-end smoke test against a running server.

    python -m uvicorn app.main:app --port 8000 &
    python scripts/smoke_test.py --base-url http://127.0.0.1:8000

Exercises every endpoint with synthetic images and prints a readable summary.
Exits non-zero if any check fails, so it is usable as a deployment gate.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


# --------------------------------------------------------------------------- #
# Synthetic dermoscopic images (self-contained, no dataset needed)
# --------------------------------------------------------------------------- #


def synth(size=320, radius=0.28, irregular=False, multicolor=False, hair=False, seed=7):
    import cv2
    from PIL import Image

    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0], image[..., 1], image[..., 2] = 214.0, 168.0, 142.0
    image = np.clip(image + rng.normal(0, 6, image.shape), 0, 255).astype(np.uint8)

    centre, r = size // 2, int(size * radius)
    mask = np.zeros((size, size), np.uint8)
    if irregular:
        angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        wobble = 1.0 + 0.32 * np.sin(5 * angles) + rng.normal(0, 0.09, angles.shape)
        points = np.stack(
            [centre + r * wobble * np.cos(angles), centre + r * wobble * np.sin(angles)], axis=1
        ).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    else:
        cv2.circle(mask, (centre, centre), r, 255, -1)

    lesion = image.copy()
    lesion[mask > 0] = (92, 64, 52)
    if multicolor:
        cv2.circle(lesion, (centre - r // 3, centre), r // 3, (28, 22, 20), -1)
        cv2.circle(lesion, (centre + r // 3, centre - r // 4), r // 4, (168, 122, 96), -1)
        cv2.circle(lesion, (centre, centre + r // 2), r // 5, (196, 196, 205), -1)
        cv2.circle(lesion, (centre + r // 4, centre + r // 3), r // 6, (176, 58, 52), -1)
        lesion[mask == 0] = image[mask == 0]

    lesion = cv2.GaussianBlur(lesion, (5, 5), 0)
    lesion = np.clip(lesion.astype(np.float32) + rng.normal(0, 11, lesion.shape), 0, 255)
    lesion = lesion.astype(np.uint8)

    if hair:
        for _ in range(14):
            a = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            b = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            cv2.line(lesion, a, b, (38, 28, 24), 2)

    buffer = io.BytesIO()
    Image.fromarray(lesion).save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Minimal multipart client (stdlib only)
# --------------------------------------------------------------------------- #


def post_multipart(url: str, files: list[tuple[str, str, bytes]], fields: dict[str, str]):
    boundary = f"----derm{uuid.uuid4().hex}"
    body = io.BytesIO()

    def write(text: str) -> None:
        body.write(text.encode())

    for name, value in fields.items():
        write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
    for field, filename, payload in files:
        write(
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{field}"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        )
        body.write(payload)
        write("\r\n")
    write(f"--{boundary}--\r\n")

    request = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


def post_json(url: str, payload: dict, *, raw: bool = False):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
        return data if raw else json.loads(data)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    small = synth(radius=0.18, seed=9)
    big = synth(radius=0.34, irregular=True, multicolor=True, hair=True, seed=9)

    print(f"\nSmoke test against {base}\n" + "=" * 72)

    print("\n1. Health and metadata")
    health = get_json(f"{base}/api/health")
    check("GET /api/health", health["model_loaded"] is True, f"weights: {health['weights_status']}")
    meta = get_json(f"{base}/api/meta")
    check("GET /api/meta returns 7 classes", len(meta["classes"]) == 7)
    check("disclaimer present", bool(meta["disclaimer"]))
    trained = meta["model"].get("is_trained", False)
    if not trained:
        print("       note: no trained checkpoint, so class probabilities are random")

    print("\n2. Single-image analysis")
    result = post_multipart(f"{base}/api/analyze", [("file", "lesion.png", big)], {})
    check("prediction present", result["prediction"]["code"] in [c["code"] for c in meta["classes"]])
    check("severity graded", bool(result["severity"]["tier"]),
          f"{result['severity']['tier']} @ {result['severity']['score']}/100")
    check("quality assessed", result["quality"]["score"] > 0,
          f"{result['quality']['score']}/100 {result['quality']['verdict']}")
    check("segmentation reliable", result["segmentation"]["reliable"] is True,
          f"{result['segmentation']['method']}, conf {result['segmentation']['confidence']}")
    abcd = result["morphology"]["abcd"]
    check("ABCD computed", abcd["tds"] > 0,
          f"TDS {abcd['tds']} A{abcd['asymmetry']} B{abcd['border']} C{abcd['colors']} D{abcd['structures']}")
    check("Grad-CAM produced", "overlay" in result["images"])
    check("attention alignment computed", result["explanation"]["attention"] is not None)
    uncertainty = result["uncertainty"]
    check("TTA ran", uncertainty["n_tta"] > 0, f"{uncertainty['n_tta']} views, "
          f"agreement {uncertainty['tta_agreement']}")
    # An untrained head emits logits of order 1e-5, so softmax sits at exactly
    # 1/7 and the MC spread is ~1e-6. Only require a visible spread once real
    # weights are loaded; otherwise just require that the passes actually ran.
    if trained:
        check("MC dropout is stochastic", uncertainty["n_mc"] > 0 and uncertainty["mc_std"] > 0,
              f"{uncertainty['n_mc']} passes, std {uncertainty['mc_std']}")
    else:
        check("MC dropout passes ran", uncertainty["n_mc"] > 0,
              f"{uncertainty['n_mc']} passes, std {uncertainty['mc_std']} "
              "(near-zero is expected with an untrained head)")
    check("hair was detected and inpainted",
          any("hair" in step for step in result["preprocessing"]),
          "; ".join(result["preprocessing"]))
    check("narrative generated", bool(result["narrative"]["impression"]))
    check("all images are data URIs",
          all(v.startswith("data:image/png;base64,") for v in result["images"].values()),
          f"{len(result['images'])} renders")

    print("\n3. Input validation")
    for label, payload, filename in (
        ("corrupt image rejected", b"not an image", "x.png"),
        ("empty file rejected", b"", "x.png"),
    ):
        try:
            post_multipart(f"{base}/api/analyze", [("file", filename, payload)], {})
            check(label, False, "server accepted invalid input")
        except urllib.error.HTTPError as error:
            check(label, error.code == 400, f"HTTP {error.code}")

    print("\n4. Batch triage")
    batch = post_multipart(
        f"{base}/api/analyze/batch",
        [("files", "a.png", small), ("files", "b.png", big)],
        {"options": json.dumps({"include_images": False, "persist": False})},
    )
    check("batch analysed both", batch["succeeded"] == 2, f"tiers {batch['tier_distribution']}")
    check("priority queue present", "priority_queue" in batch)

    print("\n5. Longitudinal change tracking")
    change = post_multipart(
        f"{base}/api/compare",
        [("baseline", "a.png", small), ("followup", "b.png", big)],
        {
            "baseline_date": "2025-01-10",
            "followup_date": "2025-06-10",
            "frame_width_mm": "20",
            "include_images": "false",
        },
    )
    check("growth detected", change["verdict"] != "stable",
          f"{change['verdict']} @ {change['change_score']}/100")
    check("interval computed", change["days_between"] == 151, f"{change['days_between']} days")
    check("mm measurement present", change["metrics"][0]["name"].startswith("Diameter (mm"))
    diameter = next(m for m in change["metrics"] if m["name"].startswith("Diameter (fraction"))
    check("diameter increased", diameter["percent_change"] > 0, f"{diameter['percent_change']:+.1f}%")

    identical = post_multipart(
        f"{base}/api/compare",
        [("baseline", "a.png", small), ("followup", "a.png", small)],
        {"include_images": "false"},
    )
    check("identical images read as stable", identical["verdict"] == "stable",
          f"score {identical['change_score']}")

    print("\n6. PDF report")
    pdf = post_json(f"{base}/api/report/pdf", result, raw=True)
    check("PDF renders", pdf.startswith(b"%PDF"), f"{len(pdf) / 1024:.0f} KB")
    pdf_by_id = post_json(f"{base}/api/report/pdf", {"case_id": result["case_id"]}, raw=True)
    check("PDF by stored case id", pdf_by_id.startswith(b"%PDF"))

    print("\n7. Case history")
    cases = get_json(f"{base}/api/cases")
    check("case was persisted", cases["total"] >= 1, f"{cases['total']} stored")
    detail = get_json(f"{base}/api/cases/{result['case_id']}")
    check("case detail retrievable", detail["case_id"] == result["case_id"])
    stats = get_json(f"{base}/api/cases/stats")
    check("stats aggregate", stats["total"] >= 1, f"tiers {stats['by_tier']}")

    print("\n8. Frontend and docs")
    with urllib.request.urlopen(f"{base}/", timeout=30) as response:
        html = response.read().decode()
    check("index.html served", "Dermoscopic Lesion Analysis" in html)
    for asset in ("styles.css", "app.js"):
        with urllib.request.urlopen(f"{base}/{asset}", timeout=30) as response:
            check(f"{asset} served", response.status == 200, f"{len(response.read()) / 1024:.0f} KB")
    schema = get_json(f"{base}/openapi.json")
    check("OpenAPI schema", "/api/analyze" in schema["paths"],
          f"{len(schema['paths'])} paths documented")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
