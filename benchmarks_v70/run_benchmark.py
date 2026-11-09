#!/usr/bin/env python3
"""Reproducible PXL v7 synthetic lossless benchmark and exactness check."""
from __future__ import annotations
import argparse, csv, statistics, tempfile, time, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pxl
IMAGES = ROOT / "benchmarks_v50" / "images"
NAMES = ("gradient", "texture", "photo_like", "noise", "blocks")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", choices=("fast", "balanced", "maximum"), default="balanced")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    if args.runs < 1 or args.runs > 20:
        ap.error("--runs must be between 1 and 20")
    rows = []
    for name in NAMES:
        src = IMAGES / f"{name}.png"
        with Image.open(src) as im:
            original = im.convert("RGB").copy()
            raw_bytes = len(original.tobytes())
        with tempfile.TemporaryDirectory() as td:
            packed, restored = Path(td)/"sample.pxl", Path(td)/"restored.png"
            pxl.compress(src, packed, tile_size=256, effort=args.effort, backend="cpu", workers=1)
            pxl.decompress(packed, restored, backend="cpu", workers=1)
            enc, dec = [], []
            for _ in range(args.runs):
                t = time.perf_counter()
                info = pxl.compress(src, packed, tile_size=256, effort=args.effort, backend="cpu", workers=1)
                enc.append(time.perf_counter() - t)
                t = time.perf_counter()
                pxl.decompress(packed, restored, backend="cpu", workers=1)
                dec.append(time.perf_counter() - t)
            with Image.open(restored) as decoded:
                exact = original.tobytes() == decoded.convert("RGB").tobytes()
            if not exact:
                raise RuntimeError(f"Lossless verification failed for {name}")
            row = {"image": name, "input_png_bytes": src.stat().st_size,
                   "raw_pixel_bytes": raw_bytes, "output_pxl_bytes": packed.stat().st_size,
                   "ratio_raw": packed.stat().st_size / raw_bytes,
                   "encode_median_s": statistics.median(enc),
                   "decode_median_s": statistics.median(dec), "exact": exact,
                   "pxl_version": info["version"], "effort": args.effort}
            rows.append(row)
            print(row, flush=True)
    out = ROOT / "benchmarks_v70" / f"v7_{args.effort}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"total={sum(r['output_pxl_bytes'] for r in rows):,} bytes; saved {out}")

if __name__ == "__main__":
    main()
