#!/usr/bin/env python3
"""Run every image in ``samples/`` through the live API and tabulate the result.

    python scripts/demo_samples.py --base-url http://127.0.0.1:8010

This is the quickest end-to-end proof that the whole stack works on real data:
it exercises upload validation, quality gating, segmentation, ABCD morphometry,
classification, uncertainty, Grad-CAM and severity grading, then prints the
prediction beside the known ground truth from the folder name.

Accuracy on this handful of images is not a meaningful metric — it is far too
small a sample. Use ``python -m derm.evaluate`` for that. This is a smoke check
that the pipeline produces sane output on genuine dermoscopic input.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.request
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def post_image(url: str, path: Path, options: dict) -> dict:
    boundary = f"----derm{uuid.uuid4().hex}"
    ctype = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    body = bytearray()

    body += f"--{boundary}\r\n".encode()
    body += (f'Content-Disposition: form-data; name="file"; '
             f'filename="{path.name}"\r\n').encode()
    body += f"Content-Type: {ctype}\r\n\r\n".encode()
    body += path.read_bytes()
    body += b"\r\n"

    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="options"\r\n\r\n'
    body += json.dumps(options).encode()
    body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        url, data=bytes(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8010")
    p.add_argument("--samples", type=Path, default=PROJECT_ROOT / "samples")
    args = p.parse_args()

    files = sorted(args.samples.rglob("*.jpg"))
    if not files:
        print(f"No images under {args.samples}. Run scripts/make_samples.py first.")
        return 1

    base = args.base_url.rstrip("/")
    health = json.loads(urllib.request.urlopen(f"{base}/api/health", timeout=30).read())
    print(f"\nServer: {base}   weights: {health['weights_status']}   "
          f"device: {health['device']}")
    if health["weights_status"] != "trained":
        print("  ! no trained weights loaded - class predictions will be meaningless")

    print(f"\nAnalysing {len(files)} sample image(s)\n")
    header = (f"{'image':<20} {'true':<7} {'predicted':<7} {'conf':>6} "
              f"{'tier':<14} {'TDS':>5} {'qual':>5}  {'note'}")
    print(header)
    print("-" * len(header))

    correct = 0
    escalated_mel = 0
    total_mel = 0
    options = {"include_images": False, "persist": False}

    for path in files:
        truth = path.parent.name.split("-")[0]
        try:
            r = post_image(f"{base}/api/analyze", path, options)
        except Exception as exc:  # noqa: BLE001
            print(f"{path.name:<20} {truth:<7} ERROR: {exc}")
            continue

        pred = r["prediction"]["code"]
        conf = r["prediction"]["confidence"]
        sev = r["severity"]
        tds = r.get("morphology", {}).get("abcd", {}).get("tds")
        qual = r.get("quality", {}).get("score")
        hit = pred == truth
        correct += hit
        if truth == "mel":
            total_mel += 1
            if sev["tier"] in {"HIGH", "CRITICAL"}:
                escalated_mel += 1

        note = "correct" if hit else ""
        if truth == "mel" and sev["tier"] in {"HIGH", "CRITICAL"}:
            note = (note + " · melanoma escalated").strip(" ·")
        print(f"{path.name:<20} {truth:<7} {pred:<7} {conf * 100:5.1f}% "
              f"{sev['tier']:<14} {tds if tds is not None else '—':>5} "
              f"{qual if qual is not None else '—':>5}  {note}")

    print("-" * len(header))
    print(f"\n  argmax correct: {correct}/{len(files)}  "
          f"({correct / len(files) * 100:.0f}% on a tiny sample - not a metric)")
    if total_mel:
        print(f"  melanomas escalated to HIGH/CRITICAL by the safety net: "
              f"{escalated_mel}/{total_mel}")
    print("\n  For real numbers: python -m derm.evaluate "
          "--checkpoint models/best_model.pth --data-root data/ham10000\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
