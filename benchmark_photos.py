#!/usr/bin/env python3
"""Benchmark every bundled high-resolution photo with installed codecs."""
from pathlib import Path
import argparse
from codec_compare import compare_codecs, export_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lossless", "lossy"), default="lossless")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--effort", choices=("fast", "balanced", "maximum"), default="balanced")
    args = ap.parse_args()
    root = Path(__file__).parent / "test_photos"
    images = sorted(p for p in root.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".webp"})
    if not images:
        raise SystemExit("No sample photos found. See test_photos/SOURCES.md.")
    for image in images:
        print(f"\\n=== {image.name} | {args.mode} ===")
        rows = compare_codecs(image, mode=args.mode, quality=args.quality, effort=args.effort)
        for row in rows:
            size = f"{row['size_bytes']:,}" if row["size_bytes"] else "—"
            print(f"{row['codec']:<20} {row['status']:<12} {size:>12} bytes  exact={row['exact']}")
        out = root / f"{image.stem}_{args.mode}_results.csv"
        export_csv(rows, out)
        print(f"CSV: {out}")
if __name__ == "__main__":
    main()
