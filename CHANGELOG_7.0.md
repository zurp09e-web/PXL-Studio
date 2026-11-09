# PXL Studio 7.0

- Bumped the standard PXL writer version to 7; readers retain v1-v6 support.
- Added predictor codec 26: blended local 2D, per-channel and global rANS
  statistics, selected by complete encoded size in maximum-effort mode.
- Added codec 27: collision-checked 8x8/16x16/32x32 exact block references.
- Added a high-precision v7 container for uint16, float16 and float32 grayscale,
  RGB and RGBA samples. Byte-plane substreams are encoded using the PXL engine
  and checked with per-plane CRC32 values.
- Added reversible 16-bit RGB transforms (G/R/B base and YCoCg-R); balanced
  and maximum modes compare final sizes before selecting a transform.
- Added TIFF input/output through tifffile and optional single-part OpenEXR
  input/output through the OpenEXR 3.3+ Python API.
- Removed the EMMA/BMF comparison adapters and their tool-folder setup.
- Kept the existing 8-bit lossy profile, batch queue, and native C++ ABI.

The new context-mixing and variable-block candidates currently use the portable
Python implementation. The C++ backend continues to accelerate the established
filtering, predictor and entropy paths. CUDA does not accelerate the new
variable-block reference search.
