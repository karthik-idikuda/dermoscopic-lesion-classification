"""Download and prepare HAM10000 from Harvard Dataverse.

    python scripts/prepare_data.py

Why not Kaggle: the Kaggle mirror needs an API token. Harvard Dataverse hosts the
authoritative dataset publicly (doi:10.7910/DVN/DBW86T), so this runs with no
credentials.

The download is deliberately disk-frugal, because a full naive extraction needs
around 5.7 GB of headroom and this project is expected to run on machines that do
not have it:

  * each archive is streamed to a temporary file, one at a time
  * images are decoded, downscaled so the short side is 256 px, and re-encoded
  * the archive is deleted as soon as it has been consumed

Peak additional disk use is roughly one archive (~1.4 GB) plus the growing
resized set (~350 MB), instead of ~5.7 GB. Downscaling to 256 px is lossless for
this pipeline, which trains at 224 px; the originals are 600x450.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"

FILES = {
    "metadata": 4338392,       # HAM10000_metadata.tab      0.8 MB
    "images_part_1": 3172585,  # HAM10000_images_part_1.zip  1366 MB
    "images_part_2": 3172584,  # HAM10000_images_part_2.zip  1404 MB
    "segmentations": 3838943,  # lesion masks                 11 MB
}

TARGET_SHORT_SIDE = 256
JPEG_QUALITY = 92


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def free_space(path: Path) -> int:
    return shutil.disk_usage(path).free


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context that also works behind a TLS-inspecting proxy.

    Corporate and campus networks frequently re-sign HTTPS with a private root.
    That root is in the OS keychain but not in certifi's bundle, so Python fails
    with "self signed certificate in certificate chain" on networks where curl
    succeeds. ``truststore`` delegates verification to the OS store, which fixes
    it; without truststore installed we fall back to the default bundle and let
    the caller retry via curl.
    """
    try:
        import truststore  # type: ignore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


def _download_with_curl(url: str, destination: Path, label: str) -> Path:
    """Fallback downloader.

    curl uses the system trust store natively, so it works on the same networks
    where urllib's certifi bundle rejects an intercepting proxy's certificate.
    """
    print(f"    {label}: retrying with curl (system trust store)", flush=True)
    result = subprocess.run(
        [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "3", "--retry-delay", "2",
            "--connect-timeout", "60", "--max-time", "7200",
            "-o", str(destination), url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"curl failed for {label} (exit {result.returncode}): "
            f"{result.stderr.strip() or 'no stderr'}"
        )
    print(f"    {label}: complete, {human(destination.stat().st_size)}", flush=True)
    return destination


def download(file_id: int, destination: Path, label: str) -> Path:
    """Stream a Dataverse file to disk with progress reporting."""
    url = f"{DATAVERSE}/{file_id}"
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": "derm-prepare/1.0"})
    started = time.time()
    try:
        return _stream(request, destination, label, started)
    except (urllib.error.URLError, ssl.SSLError) as exc:
        reason = getattr(exc, "reason", exc)
        print(f"    {label}: urllib failed ({reason})", flush=True)
        destination.unlink(missing_ok=True)
        return _download_with_curl(url, destination, label)


def _stream(request, destination: Path, label: str, started: float) -> Path:
    with urllib.request.urlopen(request, timeout=120, context=_ssl_context()) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 2.0:
                    last = now
                    rate = done / max(now - started, 1e-6)
                    pct = f"{done / total * 100:5.1f}%" if total else "  ?  "
                    print(
                        f"    {label}: {pct} {human(done)} at {human(rate)}/s",
                        flush=True,
                    )
    print(f"    {label}: complete, {human(destination.stat().st_size)}", flush=True)
    return destination


def extract_resized(archive: Path, out_dir: Path, label: str) -> int:
    """Extract JPEGs from an archive, downscaling each one as it is written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    with zipfile.ZipFile(archive) as zf:
        members = [
            m for m in zf.namelist()
            if m.lower().endswith((".jpg", ".jpeg", ".png"))
            and not Path(m).name.startswith("._")
        ]
        print(f"    {label}: {len(members)} images in archive", flush=True)

        for index, member in enumerate(members, start=1):
            name = Path(member).name
            target = out_dir / f"{Path(name).stem}.jpg"
            if target.exists():
                written += 1
                continue
            try:
                with zf.open(member) as source:
                    payload = source.read()
                image = Image.open(io.BytesIO(payload))
                image = image.convert("RGB")

                short = min(image.size)
                if short > TARGET_SHORT_SIDE:
                    scale = TARGET_SHORT_SIDE / short
                    new_size = (
                        max(1, round(image.width * scale)),
                        max(1, round(image.height * scale)),
                    )
                    image = image.resize(new_size, Image.LANCZOS)

                image.save(target, format="JPEG", quality=JPEG_QUALITY, optimize=True)
                written += 1
            except Exception as exc:  # noqa: BLE001 - skip a bad member, keep going
                print(f"    {label}: skipped {name} ({exc})", flush=True)

            if index % 1000 == 0:
                print(f"    {label}: resized {index}/{len(members)}", flush=True)

    return written


def write_metadata_csv(tab_path: Path, csv_path: Path) -> int:
    """Convert the Dataverse .tab metadata into the expected CSV filename.

    The rest of the codebase looks for ``HAM10000_metadata.csv``; Dataverse ships
    the same table as tab-separated. Column names and values are identical.
    """
    with open(tab_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"No rows parsed from {tab_path}")

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "ham10000")
    parser.add_argument("--skip-segmentations", action="store_true")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=3.0,
        help="Abort if free disk space would drop below this during download.",
    )
    args = parser.parse_args()

    out = args.out
    images_dir = out / "images"
    out.mkdir(parents=True, exist_ok=True)

    free = free_space(out)
    print(f"HAM10000 preparation\n  target : {out}\n  free   : {human(free)}\n")

    # One archive at a time plus the resized output; ~1.8 GB of working room.
    needed = int(1.8 * 1024**3)
    if free < needed:
        print(
            f"Not enough free disk space: {human(free)} available, ~{human(needed)} "
            "needed. Free up space and re-run.",
            file=sys.stderr,
        )
        return 1

    # ---- metadata -------------------------------------------------------- #
    csv_path = out / "HAM10000_metadata.csv"
    if csv_path.exists():
        print("1/3 metadata: already present")
    else:
        print("1/3 metadata")
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            tab = download(FILES["metadata"], Path(tmp) / "meta.tab", "metadata")
            count = write_metadata_csv(tab, csv_path)
        print(f"    wrote {csv_path.name} with {count} rows")

    # ---- images ---------------------------------------------------------- #
    print("\n2/3 images")
    existing = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    if existing >= 10015:
        print(f"    already have {existing} images, skipping download")
    else:
        for part in ("images_part_1", "images_part_2"):
            if free_space(out) < args.min_free_gb * 1024**3:
                print(
                    f"    aborting: free space below {args.min_free_gb} GB",
                    file=sys.stderr,
                )
                return 1
            with tempfile.TemporaryDirectory(dir=out) as tmp:
                archive = Path(tmp) / f"{part}.zip"
                print(f"  {part}: downloading")
                download(FILES[part], archive, part)
                print(f"  {part}: extracting and downscaling to {TARGET_SHORT_SIDE}px")
                written = extract_resized(archive, images_dir, part)
                print(f"  {part}: {written} images written")
                archive.unlink(missing_ok=True)  # free the 1.4 GB immediately

    # ---- segmentation masks (small, optional) ---------------------------- #
    if args.skip_segmentations:
        print("\n3/3 segmentations: skipped")
    else:
        masks_dir = out / "segmentations"
        if masks_dir.exists() and any(masks_dir.iterdir()):
            print("\n3/3 segmentations: already present")
        else:
            print("\n3/3 segmentations (ground-truth lesion masks)")
            with tempfile.TemporaryDirectory(dir=out) as tmp:
                archive = download(
                    FILES["segmentations"], Path(tmp) / "seg.zip", "segmentations"
                )
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(masks_dir)
            print(f"    extracted to {masks_dir}")

    # ---- verify ---------------------------------------------------------- #
    total = len(list(images_dir.glob("*.jpg")))
    print(f"\nDone.\n  images : {total}")
    print(f"  size   : {human(sum(f.stat().st_size for f in images_dir.glob('*.jpg')))}")
    print(f"  free   : {human(free_space(out))}")

    try:
        from derm.data import load_metadata

        frame = load_metadata(out)
        print(f"  loader : resolved {len(frame)} rows, "
              f"{frame['lesion_id'].nunique()} distinct lesions")
        print("\nClass distribution:")
        for code, n in frame["dx"].value_counts().items():
            print(f"    {code:<6} {n:>5}")
    except Exception as exc:  # noqa: BLE001
        print(f"  loader : could not verify ({exc})", file=sys.stderr)
        return 1

    print(f"\nNext: python -m derm.train --data-root {out} --device mps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
