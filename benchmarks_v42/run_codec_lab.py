#!/usr/bin/env python3
"""Run PXL Studio's lossless Codec Lab over the generated synthetic set."""
from pathlib import Path
from codec_compare import compare_codecs, export_csv

ROOT = Path(__file__).parent
IMAGES = ROOT / "images"
OUT = ROOT / "codec_lab"
OUT.mkdir(exist_ok=True)
for image in sorted(IMAGES.glob("*.png")):
    rows = compare_codecs(image, mode="lossless", effort="balanced",
                          backend="cpu", workers=1)
    target = OUT / f"{image.stem}.csv"
    export_csv(rows, target)
    print(f"{image.name}: wrote {target}")
