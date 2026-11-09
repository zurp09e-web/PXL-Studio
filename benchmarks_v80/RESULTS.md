# PXL Studio 8.0 synthetic benchmark results

## Scope and method

The benchmark runs on the Linux development environment with the native C++17
backend enabled (runtime SIMD dispatch: AVX-512). Each case is compressed and
decoded three times; the reported encode/decode times are medians. Every output
is decoded and compared byte-for-byte with its source pixels. All seven cases
reconstructed exactly in both profiles.

The first five 512x512 RGB patterns are the matched v7/v8 synthetic regression
set. `delta_repeat` is a custom two-half, near-repeated image intended to make
reversible delta references useful. `split_field` is a 512x512 image with a
smooth left region and random-noise right region; it exercises adaptive
segmentation. These are engineering tests, not a representative camera-photo
corpus.

## Matched v7/v8 file sizes

| Sample | v7 Balanced | v8 Balanced | v7 Maximum | v8 Maximum |
|---|---:|---:|---:|---:|
| Gradient | 389 B | 421 B | 389 B | 421 B |
| Repeating texture | 4,231 B | 4,403 B | 4,142 B | 4,314 B |
| Photo-like | 108,858 B | 108,890 B | 99,667 B | 99,699 B |
| Random noise | 786,524 B | 786,556 B | 786,524 B | 786,556 B |
| Block mosaic | 7,491 B | 7,523 B | 7,491 B | 7,523 B |
| **Total** | **907,493 B** | **907,793 B** | **898,213 B** | **898,513 B** |

The five-image v8 total is **300 bytes larger** in each profile. Most of this
is the deliberate v8 tile-record overhead: explicit coordinates and dimensions
add eight bytes per tile (four tiles per sample in this test). The texture
sample carries an additional candidate/model cost. This is not a compression
ratio win on the matched corpus.

## New mechanism-specific samples

| Sample | Control | PXL 8.0 Balanced | PXL 8.0 Maximum | Result |
|---|---:|---:|---:|---|
| Near-repeat / delta | v7 fixed-tile: 1,946 B | 2,102 B | 2,089 B | Codec 27 emitted reversible delta references and decoded exactly, but the whole file is 143–156 B larger than the v7 control. |
| Smooth/noisy split | v7 fixed 512-pixel tile: 398,020 B | 393,358 B | 393,358 B | Adaptive layout emitted four tiles and saved 4,662 B (1.17%) on this sample. |

The delta feature is verified functionally, but its current token/header overhead
outweighs the saved differences in this small test. More efficient delta run
coding, candidate pruning and local search thresholds are worthwhile next
optimizations. Adaptive segmentation has a measurable win on the split-field
case but is not guaranteed to help uniformly distributed images.

## v8 Balanced timing snapshot

Three-run medians in seconds on the same development environment:

| Sample | Encode | Decode |
|---|---:|---:|
| Gradient | 0.182 | 0.016 |
| Repeating texture | 0.352 | 0.015 |
| Photo-like | 0.468 | 0.079 |
| Random noise | 0.022 | 0.039 |
| Block mosaic | 0.380 | 0.013 |
| Near-repeat / delta | 0.171 | 0.008 |
| Smooth/noisy split | 0.089 | 0.027 |

These figures are machine- and build-specific. The v8 native build selected
AVX-512 on this host. The SIMD work is currently limited to equality scoring
in the delta candidate search; it does not imply all prediction and entropy
kernels are vectorized. Parallel history-group decoding is verified by the
automated tests with multiple independent groups; the small benchmark images
above mostly fit inside one default group and are not a performance claim for
parallel decode scaling.

## Parallel decode smoke test

A separate 512x320 mixed smooth/noise CLI test was encoded with checkpoint
interval 2. The four-tile image produced two independent reference groups;
the decoder used two worker threads and reconstructed every pixel exactly.
This confirms the grouped decode path. It is a correctness smoke test, not a
claim of a measured throughput speedup.

## Reproduce

```bash
python benchmarks_v80/run_benchmark.py --effort balanced --runs 3
python benchmarks_v80/run_benchmark.py --effort maximum --runs 3
```

CSV data is in `v8_balanced.csv` and `v8_maximum.csv`. All output files in this
run carried version 8 and exact pixel reconstruction passed.
