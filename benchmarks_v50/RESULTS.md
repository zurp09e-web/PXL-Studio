# PXL Studio 5.0 — C++ backend performance report

## Summary

This report compares the previous PXL v4.2 Python/Numba implementation with
the PXL v5.0 implementation using the custom C++17 native backend. On these
five synthetic 512×512 RGB regression images, **all five v5 outputs are exactly
the same size as the corresponding v4 outputs**. The C++ build substantially
improves most cases; the photo-like encoder result is effectively unchanged
(with a sub-1% timing difference in this run), while photo-like decoding improves.

This is a comparison against the previous PXL implementation, **not** a claim
that PXL now beats PNG, WebP or JPEG XL. The corpus is intentionally small and
synthetic; real photographs still need separate Codec Lab tests.

## Test method

- 512×512, 8-bit RGB PNG inputs, identical source files for both versions.
- 256-pixel PXL tiles, lossless mode, balanced effort, CPU backend, one internal
  worker.
- One warm-up encode and one warm-up decode, followed by five measured runs.
- Reported times are medians of those five runs.
- Timing is end-to-end wall clock and includes normal file I/O and writing the
  decoded PNG. Both versions use the same public API harness.
- Exactness is checked by comparing normalized decoded RGB pixel bytes.
- v4 baseline: pristine PXL Studio 4.2 source tree (Python/Numba execution).
- v5: PXL Studio 5.0 worktree, native C++ library built with `g++ 14.2.0`,
  `-O3 -std=c++17`, Linux x86-64.

## Results

| Synthetic image | PXL v4 bytes | PXL v5 bytes | Size change | Encode speedup | Decode speedup | Exact |
|---|---:|---:|---:|---:|---:|---|
| Gradient | 1,092 | 1,092 | 0.0% | 1.35× | 1.90× | Yes |
| Repeating texture | 5,054 | 5,054 | 0.0% | 1.36× | 1.92× | Yes |
| Photo-like | 112,632 | 112,632 | 0.0% | 0.99× | 1.12× | Yes |
| Random noise | 786,524 | 786,524 | 0.0% | 1.93× | 1.37× | Yes |
| Block mosaic | 7,491 | 7,491 | 0.0% | 1.38× | 2.31× | Yes |

Total PXL output across all five images: **912,793 bytes** in both versions.
There was no compression-ratio regression in this test set; the encoded PXL
files have identical sizes, not merely similar ratios.

## What changed in the implementation

1. **Fused C++ predictor pipeline:** reversible color transform, adaptive
   per-channel predictor scoring, residual generation and context ID generation
   now run together instead of through multiple Python/NumPy passes.
2. **Fused inverse filter:** predictor reconstruction and inverse color transform
   are performed in a native pass.
3. **Native custom rANS:** frequency normalization, encoding and table-based
   decoding no longer iterate symbol-by-symbol in Python for native-enabled
   operations.
4. **Independent entropy chunks:** larger PixelLZ and block-token streams may be
   divided into independently coded rANS chunks. The encoder compares the
   complete chunked representation with a single stream and only keeps chunks
   when their final serialized size is smaller. This makes entropy work
   parallelizable without changing token order.
5. **Persistent native cross-tile dictionaries:** PixelLZ hash lookup and block
   history live in C++ state across raster-ordered tiles rather than being
   rebuilt and walked in Python.
6. **Reduced Python/native crossings:** multi-stage filtering and stateful
   matching are exposed as larger native calls.
7. **Effort profiles:** fast/balanced/maximum alter the candidate search breadth
   and chunk size. Lossless output remains exact; output size may vary by profile.

## Interpretation and limitations

- The gains are meaningful but not uniform in magnitude. The photo-like encode
  time was 0.99× the previous implementation (effectively the same within run
  variance), while its decode time improved by 1.12×. Photo compression remains
  an important category to optimize against camera photos.
- The files are byte-size identical to v4 only on these five test inputs and
  the balanced profile. Other images and effort levels may select a different
  representation or chunking decision.
- These figures measure the whole API path, not an isolated kernel microbench.
- The native C++ library was compiled and executed on Linux here. The project
  contains MSVC/MinGW and macOS build paths, but those platforms were not
  available for validation in this run.
- CUDA hardware was not available, so the optional CuPy path is not included in
  this measurement.
- This corpus is not a statistically representative photographic benchmark.
  Test a varied camera corpus and compare lossless JXL/WebP/PNG in Codec Lab
  before drawing conclusions about codec competitiveness.

## Reproduce

The exact PNG test images and CSV baselines are included next to this report.
With the C++ library built, run:

```bash
python benchmarks_v50/run_benchmark.py
```

The script warms both encode/decode paths, runs five repetitions and writes
`v5_reproduced.csv`. The v4 measurements are retained in `v4_baseline.csv`.
