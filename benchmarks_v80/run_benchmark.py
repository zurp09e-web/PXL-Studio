#!/usr/bin/env python3
"""PXL v8 reproducible synthetic benchmark with lossless verification."""
from __future__ import annotations
import argparse, csv, statistics, tempfile, time
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pxl
IMAGES = ROOT / "benchmarks_v50" / "images"


def _delta_image(path: Path):
    rng = np.random.default_rng(24080)
    patch = rng.integers(0, 12, size=(32, 32, 3), dtype=np.uint8)
    base = np.tile(patch, (8, 8, 1))
    image = np.concatenate([base, base.copy()], axis=1)
    # Slightly alter some pixels in the repeated second tile.
    right = image[:, 512:, :]
    right.reshape(-1)[::193] = (right.reshape(-1)[::193].astype(np.uint16) + 3) % 256
    Image.fromarray(image, "RGB").save(path)


def _split_image(path: Path):
    rng = np.random.default_rng(24081)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[:, 256:, :] = rng.integers(0, 256, size=(512, 256, 3), dtype=np.uint8)
    Image.fromarray(image, "RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", choices=("fast", "balanced", "maximum"), default="balanced")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--checkpoint-interval", type=int, default=8)
    ap.add_argument("--fixed-tiles", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.runs <= 20:
        ap.error("--runs must be between 1 and 20")
    rows = []
    with tempfile.TemporaryDirectory(prefix="pxl_v80_bench_") as temp:
        temp = Path(temp)
        extra = temp / "split_field.png"
        _split_image(extra)
        delta = temp / "delta_repeat.png"
        _delta_image(delta)
        cases = [(name, IMAGES / f"{name}.png", 256, True)
                 for name in ("gradient", "texture", "photo_like", "noise", "blocks")]
        cases += [("delta_repeat", delta, 256, True), ("split_field", extra, 512, True)]
        for name, source, tile_size, use_adaptive in cases:
            with Image.open(source) as image:
                original = image.convert("RGB").copy()
                raw_size = len(original.tobytes())
            packed = temp / f"{name}.pxl"
            restored = temp / f"{name}_restored.png"
            encode_times, decode_times = [], []
            meta = None
            for _ in range(args.runs):
                t0 = time.perf_counter()
                meta = pxl.compress(source, packed, tile_size=tile_size, effort=args.effort,
                                    backend="cpu", workers=1,
                                    adaptive_tiles=(use_adaptive and not args.fixed_tiles),
                                    checkpoint_interval=args.checkpoint_interval)
                encode_times.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                decoded = pxl.decompress(packed, restored, backend="cpu", workers=4)
                decode_times.append(time.perf_counter() - t0)
            with Image.open(restored) as image:
                exact = original.tobytes() == image.convert("RGB").tobytes()
            if not exact:
                raise RuntimeError(f"Exact reconstruction failed: {name}")
            row = {
                "image": name,
                "tile_size": tile_size,
                "adaptive_tiles": bool(use_adaptive and not args.fixed_tiles),
                "checkpoint_interval": args.checkpoint_interval,
                "tile_count": meta["tiles"],
                "input_png_bytes": source.stat().st_size,
                "raw_pixel_bytes": raw_size,
                "output_pxl_bytes": packed.stat().st_size,
                "ratio_raw": packed.stat().st_size / raw_size,
                "encode_median_s": statistics.median(encode_times),
                "decode_median_s": statistics.median(decode_times),
                "decode_groups": decoded.get("parallel_groups", 1),
                "decode_workers": decoded.get("workers", 1),
                "exact": exact,
                "pxl_version": meta["version"],
                "effort": args.effort,
            }
            rows.append(row)
            print(row, flush=True)
    out = ROOT / "benchmarks_v80" / f"v8_{args.effort}.csv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"total={sum(row['output_pxl_bytes'] for row in rows):,} bytes; saved {out}")


if __name__ == "__main__":
    main()
