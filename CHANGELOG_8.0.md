# PXL Studio 8.0

## Added

- Version 8 standard tile framing with explicit tile coordinates and dimensions.
- Adaptive tile segmentation driven by differences in local spatial activity.
- A strict partition validator that rejects rectangle gaps and overlaps.
- Independent cross-tile reference-history groups for parallel decoding.
- Native C++ context-mixed rANS encoding/decoding matching the Python reference.
- Native variable-size block matching for 8x8, 16x16 and 32x32 regions.
- Exact block references and reversible modulo-256 delta references.
- SIMD-assisted match scoring with runtime dispatch to AVX-512BW, AVX2, ARM NEON or scalar fallback where supported.
- GUI and batch controls for adaptive tiles and checkpoint interval.
- v8 byte-plane streams inside the compatible v7 high-precision uint16/float16/float32 envelope.

## Compatibility

- Existing PXL v1-v7 files remain readable.
- High-precision outer headers remain v7; their new embedded byte-plane streams use v8.
- Native ABI is now 5. Rebuild the platform library with `python build_native.py`.

## Validation

- Full suite: 70 passed, 3 skipped in the Linux development environment.
- Skips: CUDA device unavailable, optional OpenEXR bindings absent, optional PyPNG 16-bit PNG support absent.
- Native Linux build succeeded and runtime dispatch selected AVX-512.
- Delta-token round-trip, native/Python mixed-rANS parity, adaptive partition coverage, independent-group decode and forced Python fallback are covered by tests.
- Synthetic benchmark includes a matched five-image regression set plus a near-repeat delta sample and a smooth/noisy adaptive-split sample. See `benchmarks_v80/RESULTS.md`.
