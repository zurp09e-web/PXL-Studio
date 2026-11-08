# PXL Studio 8.0.1 — CUDA and Codec Lab hotfix

## Fixed

- Codec Lab now keeps PXL in the comparison when CUDA is unavailable or a CUDA
  encoder attempt fails. It retries PXL on CPU and records the GPU failure in
  the row note and CSV backend columns.
- PXL Codec Lab decoding explicitly uses CPU because the v8 reconstruction path
  is CPU-side; requesting CUDA for this stage never improves timing and could
  previously make a valid PXL encode look unavailable.
- PXL Codec Lab identifies its own encoder as PXL v8 rather than the stale v7
  label.
- CUDA detection now records the actual CuPy import / driver / runtime error and
  runs a small allocation, GPU reduction and synchronization smoke test.
- The GUI has a Recheck control and reports CuPy, driver/runtime, compute
  capability, native-library path, ABI and SIMD backend.
- Native-library discovery now includes CMake multi-configuration Release,
  Debug and RelWithDebInfo output folders. The loader can be explicitly
  refreshed after building a DLL while the application is open.
- Added `diagnose_cuda.py`, which checks `nvidia-smi`, CuPy, the native backend,
  a real CUDA PXL encode, CPU decode and exact reconstruction.
- CUDA requirements now install CuPy with its `[ctk]` runtime-component extra.
  The native C++ build is explicitly described as CPU/SIMD-only.

## Validation in this environment

- `python -m pytest -q`: **72 passed, 3 skipped**.
- Native C++ shared library built successfully with ABI 5; runtime dispatch
  selected AVX-512 on the Linux validation host.
- Codec Lab PXL lossless comparison returned `OK`, exact pixels, and a CPU
  fallback note when a CUDA backend was explicitly requested but CuPy was not
  installed.
- CUDA hardware test could not run in this environment (`ModuleNotFoundError:
  No module named 'cupy'`, no `nvidia-smi`). This release improves diagnostics and
  fallback behavior; it does not claim the GPU path was validated here.
- Windows/MSVC build and RTX hardware runtime validation still need to be run
  on the target Windows machine.
