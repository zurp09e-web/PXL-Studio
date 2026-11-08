# PXL Studio 5.0 validation report

## Environment

- Operating system: Linux x86-64
- C++ compiler: g++ 14.2.0
- C++ standard: C++17
- Native build flags: `-O3 -std=c++17 -fPIC -shared -pthread`
- Python native binding: `ctypes`, ABI version 2

## Results

```text
46 passed, 1 skipped
```

The skipped test requires usable NVIDIA CUDA hardware, which was not available
in this environment. The C++ CPU library was rebuilt from the included source
and loaded successfully.

## Covered behavior

- C++ fused transform/predictor filtering is byte-identical to the Python
  reference for grayscale, RGB and RGBA across all five reversible transforms.
- C++ inverse filtering reconstructs source samples exactly.
- Native spatial rANS output matches the Python reference stream byte-for-byte.
- Native context-rANS encoder/decoder paths round-trip exactly.
- Cross-tile PixelLZ state and block-reference state round-trip across multiple
  tiles.
- Independent `PXMS` rANS chunks are selected when their full serialization is
  smaller, then decoded and concatenated exactly.
- `fast`, `balanced` and `maximum` effort profiles all write v5 files and
  preserve exact pixels in lossless mode.
- A stored PXL v4 file is still readable and reconstructs the expected image.
- Existing legacy, lossy, alpha, GUI/batch worker and malformed-file tests pass.

## Benchmark qualification

See `benchmarks_v50/RESULTS.md`. It compares old PXL v4.2 and PXL v5.0 using the
same five deterministic 512×512 synthetic PNGs and a warm-up + five-run median
harness. It measures end-to-end timings on this Linux host, not a universal
throughput guarantee. Windows and macOS native builds, CUDA, large real-photo
corpora and Codec Lab comparisons against JPEG XL/WebP have not been validated
in this run.
