#!/usr/bin/env python3
"""Warm, repeatable PXL v6 synthetic lossless benchmark."""
from __future__ import annotations
import argparse, csv, statistics, tempfile, time, sys
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pxl
IMAGES = ROOT / "benchmarks_v50" / "images"
NAMES = ("gradient", "texture", "photo_like", "noise", "blocks")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", choices=("fast", "balanced", "maximum"), default="balanced")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    rows=[]
    for name in NAMES:
        src=IMAGES/f"{name}.png"
        with tempfile.TemporaryDirectory() as td:
            packed=Path(td)/"sample.pxl"; restored=Path(td)/"restored.png"
            pxl.compress(src,packed,tile_size=256,effort=args.effort,backend="cpu",workers=1)
            pxl.decompress(packed,restored,backend="cpu",workers=1)
            enc=[]; dec=[]
            for _ in range(args.runs):
                t=time.perf_counter(); info=pxl.compress(src,packed,tile_size=256,effort=args.effort,backend="cpu",workers=1); enc.append(time.perf_counter()-t)
                t=time.perf_counter(); pxl.decompress(packed,restored,backend="cpu",workers=1); dec.append(time.perf_counter()-t)
            with Image.open(src) as a, Image.open(restored) as b:
                exact = a.convert("RGB").tobytes() == b.convert("RGB").tobytes()
            if not exact: raise RuntimeError(f"Lossless verification failed for {name}")
            row={"image":name,"input_png_bytes":src.stat().st_size,"output_pxl_bytes":packed.stat().st_size,
                 "encode_median_s":statistics.median(enc),"decode_median_s":statistics.median(dec),
                 "exact":exact,"pxl_version":info["version"],"effort":args.effort}
            rows.append(row); print(row)
    out=ROOT/"benchmarks_v60"/f"v6_{args.effort}.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(f"total={sum(r['output_pxl_bytes'] for r in rows):,} bytes; saved {out}")
if __name__=="__main__": main()
