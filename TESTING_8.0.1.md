# PXL Studio 8.0.1 hotfix validation

## Automated tests

```powershell
py -m pytest -q
```

Result in the Linux validation environment: **72 passed, 3 skipped**.

The added regression tests verify:

- Codec Lab PXL lossless comparison remains `OK` and pixel-exact when CUDA is
  explicitly requested but unavailable; the row records the CPU fallback and
  the precise CUDA reason.
- `backend="auto"` retries predictor processing on CPU if a CUDA kernel fails
  after the initial device smoke test.
- Existing lossless, lossy, batch, native ABI, SIMD, high-precision and legacy
  round-trip tests continue to pass.

## Native build

`python build_native.py` successfully built the ABI-5 Linux shared library.
Runtime dispatch reported AVX-512 on the validation host. The native build is
CPU/SIMD-only; it does not install or compile CuPy/CUDA.

## CUDA environment limitation

This validation container has no CuPy installation and no `nvidia-smi`, so an
actual CUDA kernel run could not be performed here. `python diagnose_cuda.py`
reported `ModuleNotFoundError: No module named 'cupy'` and exited with code 2,
which is the expected diagnostic result when the optional CUDA package is not
installed.

Windows/MSVC and the user's NVIDIA GPU still require a target-machine run of
`py build_native.py` and `py diagnose_cuda.py`.
