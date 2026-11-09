# PXL Studio 4.2 test report

- `PYTHONPATH=. pytest -q`: 39 passed, 1 skipped.
- The skipped test requires a usable NVIDIA CUDA device, which was not available in the build environment.
- Numba-compiled predictor output was compared byte-for-byte with the previous reference implementation for grayscale, RGB and RGBA inputs.
- Pixel-LZ encoder/decoder was tested across multiple tiles with exact reconstruction, including long-distance repeated-tile references.
- The pure-Python fallback was tested with Numba import disabled.
- Five 512×512 synthetic files encoded by PXL 4.1 and 4.2 produced byte-for-byte identical PXL files.
- `benchmarks_v42/RESULTS.md` records timing and Codec Lab comparisons. Timings are environment-specific and are not claims that PXL exceeds the speed of mature codecs.
