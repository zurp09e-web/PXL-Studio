# Building PXL Studio 8.0.1

PXL Studio keeps its desktop application and image I/O layer in Python, while
its default accelerated codec backend is a custom C++17 shared library.

## Requirements

- Python 3.10 or newer
- Pillow and NumPy (listed in `requirements.txt`)
- A C++17 compiler to build the native backend
- Numba is retained for the portable Python fallback and reference kernels

## Install and build

```bash
python -m pip install -r requirements.txt
python build_native.py
python pxl_gui.py
```

`build_native.py` automatically chooses an available compiler and generates:

- Windows: `pxl_native.dll` (MSVC `cl.exe` or MinGW-w64 `g++`)
- Linux: `libpxl_native.so` (`g++` or compatible C++ compiler)
- macOS: `libpxl_native.dylib` (`clang++`)

### Windows compiler setup

Install Visual Studio Build Tools and select **Desktop development with C++**.
Open a Developer Command Prompt or ensure `cl.exe` is on `PATH`, then run:

```powershell
py -m pip install -r requirements.txt
py build_native.py
py pxl_gui.py
```

Alternatively, install MinGW-w64 and ensure `g++` is on `PATH`.

### CMake build (optional)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The Python loader looks in the project root and `native/build/` for the shared
library. For a CMake build, copy the resulting library to the project root or
`native/build/` using the platform-specific filename.

## Fallback behavior

If the native library is missing or cannot be loaded, PXL Studio uses the
compatible Python/Numba backend. The GUI startup log shows which implementation
is active. To require C++ in a script:

```python
from pxl_native import get_backend
backend = get_backend(required=True)
print(backend.path)
```

## Test

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

The C++ tests skip if the shared library has not been built. The suite includes v4/v5 fixtures, the v8 adaptive tile partition, byte-exact mixed-rANS parity, delta-block token reconstruction, parallel history groups, fallback decoding and uint16/float16/float32 round trips.

## Native source and dependencies

`native/pxl_native.cpp` and `native/pxl_v8.cpp` implement PXL filtering,
context-mixed rANS, variable-block references and runtime SIMD match scoring.
The MSVC AVX2/AVX-512 kernels live in separate translation units so the baseline
library can dispatch safely after checking CPU and OS support. ARM builds use
NEON when the compiler exposes the required intrinsics; all builds retain a
scalar path. No third-party compressor is linked into the PXL payload path.
Pillow is used only for image I/O and Codec Lab comparisons. Legacy readers
retain compatibility with older PXL files.


## Upgrading from PXL Studio 5.0–7.0

PXL 8.0 uses native C ABI version 5. Close every running PXL/Python process
that has loaded the old native library, then rebuild it from the project root:

```powershell
py build_native.py
py pxl_gui.py
```

The builder replaces `pxl_native.dll` on Windows (or the platform library on
Linux/macOS). If the linker reports that the DLL is in use, close PXL Studio
and retry. An ABI-4 library from PXL 7.0 is deliberately rejected rather than called
with the new function table. On MSVC, the builder attempts separate AVX2 and
AVX-512 translation units; unsupported compiler flags fall back to the best
available lower ISA/scalar implementation. CMake enables AVX2 by default and
has an optional `-DPXL_ENABLE_AVX512=ON` switch for AVX-512-capable toolchains.
