"""Measure training throughput per device and batch size.

Run before committing to a long training job: on an 8 GB Apple-silicon machine
the useful batch size is set by memory pressure, not by the GPU, and MPS pays a
large one-off graph-compilation cost that is easy to mistake for a hang.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def bench(device_name: str, batch: int, steps: int = 6, arch: str = "efficientnet_b3") -> None:
    import timm

    device = torch.device(device_name)
    model = timm.create_model(arch, pretrained=False, num_classes=7).to(device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, 7, (batch,), device=device)

    sync = (
        torch.mps.synchronize if device_name == "mps"
        else torch.cuda.synchronize if device_name == "cuda"
        else lambda: None
    )

    times: list[float] = []
    for i in range(steps):
        start = time.time()
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()
        sync()
        elapsed = time.time() - start
        times.append(elapsed)
        tag = " (includes graph compilation)" if i == 0 else ""
        print(f"  {device_name} b{batch} step {i}: {elapsed:6.2f}s{tag}", flush=True)

    steady = times[1:] or times
    per_step = sum(steady) / len(steady)
    per_image = per_step / batch
    # 7,000 training images is roughly the lesion-grouped 70% split of HAM10000.
    epoch_minutes = (7000 * per_image) / 60
    print(
        f"  -> {device_name} b{batch}: {per_step:.2f}s/step, "
        f"{per_image * 1000:.0f}ms/image, ~{epoch_minutes:.1f} min per training epoch",
        flush=True,
    )
    del model, optimiser, x, y
    if device_name == "mps":
        torch.mps.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="efficientnet_b3")
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    devices = args.devices or (
        ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]
    )
    print(f"torch {torch.__version__} · arch {args.arch}\n")
    for device in devices:
        for batch in args.batches:
            try:
                bench(device, batch, args.steps, args.arch)
            except Exception as exc:  # noqa: BLE001
                print(f"  {device} b{batch}: FAILED — {exc}", flush=True)
            print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
