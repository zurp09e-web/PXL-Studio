# PXL Studio 8.0.2 — CUDA predictor routing fix

## Fixed

- Fixed the CUDA lossless encode path raising `NameError: _filter_rows_cuda_v3
  is not defined`. The implementation is defined in `pxl.py`; the v8 engine now
  calls it through its imported `core` module.
- Added a predictor-level regression test and a full public compression / exact
  decode integration test with the CUDA boundaries stubbed. This validates
  that the CUDA branch reaches the correct implementation without requiring a
  physical GPU on the test host.
- `diagnose_cuda.py` now detects conflicting `cupy`, `cupy-cuda12x` and
  `cupy-cuda13x` distributions and prints safe cleanup instructions.
- README now documents a clean uninstall/reinstall sequence for duplicate CuPy
  installations.

## User-reported hardware status

The supplied RTX 4070 SUPER diagnostic already passed CuPy initialization,
CUDA allocation, kernel execution and synchronization (compute capability
8.9; driver 13.5; runtime 13.2). The failure occurred only after entering the
PXL-specific predictor encode path. This fix addresses that source-level
NameError. The Windows GPU path still needs to be rerun on the user's machine;
this development environment does not expose the user's NVIDIA device.

## Validation in this environment

- `tests/test_cuda_fallback.py`: 3 passed, including full public compression and
  exact reconstruction through the patched CUDA branch using mocked CUDA
  transform/filter boundaries.
- Full suite: 64 passed, 13 skipped (optional CUDA / external-dependency tests
  skipped on this host).
- No claim is made that a physical RTX 4070 SUPER kernel run was performed here.
