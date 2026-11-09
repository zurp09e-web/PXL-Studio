# PXL Studio 8.0 testing notes

## Automated tests

Run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q -rs
```

Latest Linux run: **70 passed, 3 skipped**.

Skipped optional/hardware tests:

- CUDA: no CuPy CUDA device was installed.
- OpenEXR: optional Python bindings were not installed.
- 16-bit RGB/RGBA PNG: optional PyPNG import/export module was not installed.

The passing tests include byte-identical native/Python blended-rANS output,
real delta token emission, native SIMD correctness, Python fallback delta
references, adaptive partition validation, multi-group parallel decoding,
legacy-file decoding, and 16-bit/float high-precision reconstruction using
available I/O paths.

## CLI smoke test

A 512x320 RGB sample was compressed through the public CLI with adaptive tiles
and checkpoint interval 2, inspected, then decoded with four requested workers.
The reader reported four tiles, two independent history groups and two used
workers; the restored pixels matched the input exactly.

## Native build

Linux C++17 build command:

```bash
python build_native.py
```

Result: successful build of `libpxl_native.so`, ABI 5. Runtime dispatch on the
validation machine reported `AVX-512`. Windows/MSVC and ARM/NEON builds were
not available in this environment and must be compiled on target hardware.

## Manual checks

On Windows, build the native library from a Visual Studio Developer Command
Prompt, then launch the GUI:

```powershell
py -m pip install -r requirements.txt
py build_native.py
py pxl_gui.py
```

The GUI startup log should identify the loaded backend and SIMD path. If the
native library is unavailable, the Python fallback remains functional.
