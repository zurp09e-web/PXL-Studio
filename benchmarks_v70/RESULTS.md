# PXL Studio 7.0 — synthetic regression results

## Scope

The benchmark uses five deterministic 512×512 RGB synthetic images: gradient,
repeating texture, photo-like pattern, random noise and block mosaic. Each output
was decoded and compared against the input pixels. All 5/5 files reconstructed
exactly in both effort profiles.

These are synthetic regression samples, not a representative camera-photo
corpus. The v6 baseline files were recorded with five measured runs after
warm-up; v7 was recorded with three measured runs after warm-up. File sizes are
comparable; timing figures should be treated as indicative rather than a
controlled performance regression test.

## File sizes: v6 versus v7

### Balanced

| Sample | PXL v6 | PXL v7 | Change |
|---|---:|---:|---:|
| Gradient | 389 B | 389 B | — |
| Repeating texture | 5,054 B | **4,231 B** | **−823 B (−16.28%)** |
| Photo-like | 108,858 B | 108,858 B | — |
| Random noise | 786,524 B | 786,524 B | — |
| Block mosaic | 7,491 B | 7,491 B | — |
| **Total** | **908,316 B** | **907,493 B** | **−823 B (−0.091%)** |

### Maximum

| Sample | PXL v6 | PXL v7 | Change |
|---|---:|---:|---:|
| Gradient | 389 B | 389 B | — |
| Repeating texture | 4,965 B | **4,142 B** | **−823 B (−16.58%)** |
| Photo-like | 99,667 B | 99,667 B | — |
| Random noise | 786,524 B | 786,524 B | — |
| Block mosaic | 7,491 B | 7,491 B | — |
| **Total** | **899,036 B** | **898,213 B** | **−823 B (−0.092%)** |

The v7 size improvement in this dataset comes entirely from the repeating
texture test. The other samples select the same-sized payloads as v6. This is
useful evidence that the new candidates are safely discarded when they do not
win, but it is not evidence of a broad overall compression breakthrough.

## PXL v7 timing snapshots

Three-run medians after warm-up, seconds. Lower is faster.

| Sample | Balanced encode | Balanced decode | Maximum encode | Maximum decode |
|---|---:|---:|---:|---:|
| Gradient | 0.260 | 0.040 | 0.707 | 0.040 |
| Repeating texture | 0.331 | 0.035 | 0.482 | 0.041 |
| Photo-like | 0.462 | 0.100 | 1.135 | 0.121 |
| Random noise | 0.043 | 0.063 | 0.044 | 0.062 |
| Block mosaic | 0.388 | 0.035 | 1.103 | 0.034 |

Maximum effort is significantly slower for some patterned images. The new
variable-block candidate is deliberately gated by a cheap search for exact
16×16/32×32 cross-tile matches, but Python dictionary/candidate work still has
noticeable costs. Use Balanced for a more practical size/speed trade-off. The
native C++ backend does not yet accelerate the new codec-27 match search.

## High-precision validation

The test suite verifies exact sample reconstruction for uint16 grayscale PNG,
uint16 RGB TIFF, float16 RGBA TIFF and float32 HDR TIFF, including bitwise
checks for floating-point values. These tests exercise the byte-plane framing,
per-plane CRCs and reversible uint16 RGB transforms. They validate sample
fidelity, not source metadata preservation or HDR color-management behavior.
OpenEXR file I/O is optional and was not run in the current environment because
the OpenEXR Python package was not installed. 16-bit RGB/RGBA PNG import uses
PyPNG; it is included as a dependency but was not separately exercised here.

## Reproduce

```bash
python -m pytest tests -q
python benchmarks_v70/run_benchmark.py --effort balanced --runs 3
python benchmarks_v70/run_benchmark.py --effort maximum --runs 3
```

CSV captures are `v7_balanced.csv` and `v7_maximum.csv`. The matched v6 file
sizes are in `../benchmarks_v60/v6_balanced.csv` and
`../benchmarks_v60/v6_maximum.csv`.
