# PXL Studio 5.0 — Native C++ codec core

## Architecture

- Kept Python/Tkinter for the GUI, image I/O, Codec Lab and batch orchestration.
- Added a custom C++17 shared library behind a stable `ctypes` C ABI.
- New encoder writes PXL v5; reader retains PXL v1-v4 compatibility.
- Added platform build scripts for MSVC/MinGW, Linux g++ and macOS clang++.

## Native execution paths

- Fused reversible color transforms, adaptive per-channel prediction, residual
  emission and 17-context generation.
- Fused inverse prediction and color reconstruction.
- Native custom rANS encode/decode with precomputed frequency and decode tables.
- Independent entropy chunks for eligible larger token streams; use only when
  the serialized chunked form is smaller.
- Persistent C++ cross-tile PixelLZ history and match index.
- Persistent C++ block-reference dictionary with collision verification.
- Reduced Python/native transitions by combining multiple stages into fewer
  calls.

## Performance profiles

- `fast`, `balanced` and `maximum` select different reversible transform sets
  and independent entropy chunk sizes.
- Maximum effort explores additional candidates; it can take longer and does
  not guarantee a smaller file for every image.
- Parallel entropy coding is used for sufficiently large independent token
  chunks. Dictionary-dependent decisions remain deterministic and sequential.

## Compatibility and validation

- PXL v5 writer; v1-v4 readers retained.
- v4 tile representations remain usable as measured candidates in the v5
  writer, preserving exact bytes and semantics.
- Native filter parity, rANS parity, PixelLZ cross-tile, block dictionary,
  corruption handling, batch/multiprocess and legacy fixture tests pass.
- Linux C++ build and native execution validated in this development
  environment. CUDA-specific hardware test was skipped because no usable CUDA
  device was available.
