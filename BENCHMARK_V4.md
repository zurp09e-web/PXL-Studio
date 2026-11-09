# PXL v4 synthetic benchmark

Environment: build container, Python implementation, 256×256 RGB inputs,
64-pixel tiles, `balanced` effort, CPU backend, one worker. Timings are single
observations, not a rigorous speed benchmark. The images are generated test
patterns, not camera photographs. Pixel-exact reconstruction was verified for
all PXL v3 and v4 outputs.

| Input | PXL v3 bytes | PXL v4 bytes | Size change | v3 encode | v4 encode |
|---|---:|---:|---:|---:|---:|
| Smooth gradient | 13,417 | **2,429** | **−81.9%** | 0.445 s | 1.600 s |
| Repeated 8×8 block mosaic | 8,964 | **1,554** | **−82.7%** | 0.203 s | 1.469 s |
| Repeated 32×32 patch | 60,772 | **5,017** | **−91.7%** | 0.161 s | 1.454 s |
| Random noise | 197,492 | **196,916** | **−0.3%** | 0.098 s | 0.013 s |
| Flat color | 756 | **756** | **0.0%** | 0.128 s | 0.184 s |

## Interpretation

PXL v4 substantially improves the generated gradient and repeated-pattern
cases because cross-tile pixel references and 8×8 block dictionary entries can
reuse exact data that the tile-independent v3 path could not share. It is
unchanged on flat color and only slightly smaller on random noise. These
results do not establish that v4 is universally smaller than PNG, WebP lossless
or JPEG XL; the corpus is synthetic and small.

The current v4 implementation is materially slower in pure Python on highly
compressible tiles. It tries multiple custom candidates and maintains global
pixel/block dictionaries. The incompressible-noise path is much quicker because
it bypasses match-finding and most predictor searches. The flat-tile path
avoids dictionary indexing and now compares raw, palette and the PXL-native
residual/rANS candidate by completed payload size.

Multi-process predictor analysis is available, but cross-tile dictionaries and
sequential rANS streams are committed in raster order. CUDA is optional and
accelerates only selected predictor/color-transform analysis when CuPy and a
working NVIDIA device are present. No CUDA hardware was available in this
build environment, so GPU execution is not claimed as tested.

## Reproducibility

Run from the project root:

```bash
python bench_v4_synthetic.py
python -m pytest tests -q
```

The script regenerates `bench_v4_synthetic.csv`. It benchmarks PXL v3 and the
v4 writer against the same five generated 256×256 RGB arrays, decodes each file,
and checks exact pixel reconstruction.
