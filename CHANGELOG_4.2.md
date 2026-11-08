# PXL Studio 4.2 — CPU throughput update

## Compatibility

- PXL container version remains **4**. No new wire-format version was introduced.
- PXL v4.1 files remain readable.
- Lossless encoded payload size and tile selection were checked against v4.1 on the bundled synthetic benchmark set; the measured files were byte-for-byte equal in size.

## Performance work

- Added Numba kernels for per-channel adaptive predictor selection and inverse filtering.
- Added compiled 17-context residual ID generation and rANS spatial decoding.
- Replaced the Python pixel-LZ hash-chain hot path with a bounded open-addressed hash table and ring links compiled with Numba. The output token grammar is unchanged.
- Added compiled pixel-LZ and legacy run-token decoders.
- Decoder history no longer maintains encoder-only dictionaries.
- Hash tables are allocated lazily and sized using tile/window limits.
- Optimized palette discovery using fixed-width pixel records and a safe sample-based early rejection.
- Numba is now installed by the standard requirements for faster default operation.

## Validation

- 39 automated tests passed; one CUDA hardware test was skipped because this build environment has no usable CUDA device.
- Compiled and pure-Python fallback paths both round-trip exactly.
- File sizes remained unchanged versus the PXL 4.1 encoder on the five recorded 512×512 synthetic cases.
