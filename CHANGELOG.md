# Changelog

## PXL Studio 3.0.0 (historical)

- Replaced new-file DEFLATE output with PXL-native byte rANS entropy coding.
- Added order-0, four-context, sixteen-context and pixel-layout-aware entropy models.
- Added a custom 64 KiB bounded hash-chain matcher with distance/length tokens.
- Added run tokens for zero sequences and repeated bytes.
- Added seven row predictors and five reversible color transforms.
- Added exact palette coding, raw-tile escape and per-tile actual-size selection.
- Added optional Numba JIT for predictor search, entropy kernels and match finding.
- Kept zlib only in the size-bounded legacy v1/v2 decoder.
- Kept CPU multicore and optional CuPy CUDA paths; entropy coding remains CPU-side.
- Retained the Tkinter GUI, codec comparison lab and optional JPEG XL comparison.

### Internal benchmark (synthetic images)

Compared with the previous PXL v2 encoder at 1024×768 RGB, 256-pixel tiles:

| Input | v2 | v3 | Reduction |
|---|---:|---:|---:|
| Structured gradient | 17,642 B | 2,264 B | 87.2% |
| Synthetic photo-like texture | 1,494,558 B | 1,340,307 B | 10.3% |
| Flat graphic | 680 B | 596 B | 12.4% |
| Repeating 64×64 texture | 618,596 B | 163,040 B | 73.6% |
| Random noise | 2,362,604 B | 2,359,532 B | 0.1% |

These are controlled generated inputs, not a broad camera-photo benchmark.


## PXL Studio 4.0.0 — Cross-tile compression architecture

- Added PXL v4 file version while preserving v1-v3 decoding.
- Added 17-context 2D residual rANS mode.
- Added pixel-aware LZ with a 512 KiB cross-tile history window.
- Added 8x8 exact 2D block dictionary references across tiles.
- Added deterministic tile-order dictionary maintenance and true bounded
  concurrent CPU predictor analysis.
- Tile representations are compared by actual encoded payload byte count.
- CUDA, when available, accelerates predictor/color-transform analysis;
  dictionary tokenization and rANS remain on CPU.

### PXL v4 validation update

- Re-ran the synthetic 256×256 RGB set after the flat-tile candidate fix.
- v4 sizes: gradient 2,429 B; 8×8 mosaic 1,554 B; repeated patch 5,017 B;
  random noise 196,916 B; flat 756 B. Every decoded output was pixel-exact.
- See `BENCHMARK_V4.md` and `bench_v4_synthetic.csv` for single-run timings
  and reproducibility commands. These figures are synthetic, not camera-photo
  generalization claims.
