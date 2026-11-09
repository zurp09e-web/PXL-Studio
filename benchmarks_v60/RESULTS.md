# PXL Studio 6.0 — adaptive spatial entropy benchmark

## Summary

PXL 6 adds two reversible predictor entropy models. The encoder compares each
fully serialized candidate against the previous predictor and tile codecs, and
only adopts a new context model when the encoded payload is smaller.

The five 512×512 synthetic RGB images were re-run from both the PXL 5.0 source
tree and PXL 6.0 on the same machine, with the same tile size (256), CPU backend,
one worker, warm-up and five measured repetitions. Both versions reconstructed
all pixels exactly.

- **Balanced:** total PXL bytes fall from 912,793 to 908,316 (**0.49% smaller**).
- **Maximum:** total PXL bytes fall from 912,704 to 899,036 (**1.50% smaller**).
- The largest individual win is the gradient, **64.38% smaller**.
- The photo-like synthetic sample is **3.35% smaller** in Balanced and
  **11.51% smaller** in Maximum.
- Random noise and the block mosaic retain their previous best tile encodings.

These gains are concentrated in predictable residual fields. This is a small
synthetic regression set, not a representative camera-photo benchmark and not
evidence that PXL beats WebP or JPEG XL on photographic collections.

## File-size results

| Image | PXL v5 Balanced | PXL v6 Balanced | PXL v5 Maximum | PXL v6 Maximum |
|---|---:|---:|---:|---:|
| Gradient | 1,092 B | **389 B** | 1,092 B | **389 B** |
| Repeating texture | 5,054 B | 5,054 B | 4,965 B | 4,965 B |
| Photo-like | 112,632 B | **108,858 B** | 112,632 B | **99,667 B** |
| Random noise | 786,524 B | 786,524 B | 786,524 B | 786,524 B |
| Block mosaic | 7,491 B | 7,491 B | 7,491 B | 7,491 B |
| **Total** | **912,793 B** | **908,316 B** | **912,704 B** | **899,036 B** |

## Balanced timing comparison

Timings are wall-clock medians, five runs after warm-up. They include reading
the input, compression/decompression and output file I/O. Lower is better.

| Image | v5 encode (s) | v6 encode (s) | v5 decode (s) | v6 decode (s) |
|---|---:|---:|---:|---:|
| Gradient | 0.1029 | 0.1166 | 0.0120 | 0.0120 |
| Repeating texture | 0.2874 | 0.2735 | 0.0114 | 0.0107 |
| Photo-like | 0.4005 | 0.3924 | 0.0714 | 0.0747 |
| Random noise | 0.0150 | 0.0137 | 0.0355 | 0.0341 |
| Block mosaic | 0.3226 | 0.3303 | 0.0093 | 0.0090 |

Balanced stays close to v5 speed: some cases are marginally faster and some
slower. The signed-context candidate search costs extra work, especially on the
gradient case, but it materially reduces that file size. The encoder screens
out high-entropy tiles and only runs the candidate on the most promising base
transform in Balanced mode.

## Maximum effort trade-off

Maximum evaluates more predictor transforms and adds a 65-model context using
left, up and upper-left residual classes. It is intentionally slower in exchange
for smaller files. Compared with PXL v5 Maximum, PXL v6 Maximum encoding takes
about 2–27% longer on these samples (the noise case is also affected by run
variation); photo-like decoding is about 20% slower because the 65-model table
is larger. The photo-like file is 11.51% smaller. Use Balanced for throughput
and Maximum when file size matters more than encode/decode time.

## New context codecs

- **Codec 24 — 17 contexts:** residual 0, +1 and -1 are separate classes; all
  other residual bytes share a fourth class. The context is
  `left_class * 4 + up_class`; row selector bytes use context 16.
- **Codec 25 — 65 contexts:** uses the same four classes for left, up and
  upper-left residuals. Context is `left * 16 + up * 4 + upper_left`; row
  selector bytes use context 64. This model is tested only in Maximum effort.
- The encoder measures the fully serialized rANS frame and compares it with the
  existing representations. If the new stream is not smaller, it is discarded.

The C++ decoder uses native ABI 4. The Python fallback supports both models.
The project was built/tested on Linux in this environment; Windows users must
rebuild `pxl_native.dll` with the included MSVC build script after upgrading.

## Reproduce

```bash
python -m pytest tests -q
python benchmarks_v60/run_benchmark.py --effort balanced --runs 5
python benchmarks_v60/run_benchmark.py --effort maximum --runs 5
```

CSV files in this directory contain matched PXL v5 and v6 balanced/maximum
measurements.
