# PXL Studio 8.0.2 — Adaptive Lossless and High-Precision Image Codec

PXL Studio is a Python/Tkinter image-compression workbench backed by a custom C++17 codec engine and a portable Python fallback. Python provides the GUI, image I/O, batch queue and file orchestration. The native engine performs blended-context rANS model construction/decoding, variable-block search, exact/delta block coding and SIMD match scoring; the Python implementation remains available as a reference and compatibility fallback.

PXL writes its own `.pxl` format. New PXL v8 8-bit files do not use zlib/DEFLATE, Zstandard, Brotli, LZ4, JPEG, WebP or JPEG XL as an internal compression backend. PXL owns its reversible transforms, pixel and block references, predictors, token grammar, frequency models, rANS entropy coder, adaptive tile layout and tile selection. Pillow and other optional packages are used for image I/O and Codec Lab comparisons, not for PXL's internal lossless compression.

PXL v8 is experimental and is not claimed to outperform mature codecs on all images. See `benchmarks_v80/RESULTS.md` for measured results and limitations.

## Highlights

- PXL v8 container with readers for older PXL v1–v7 archives.
- Native C++17 engine for blended-context rANS, variable-block search, reversible delta tokens and reference scoring.
- Adaptive tile segmentation and independent history groups.
- Runtime SIMD match scoring with scalar fallback, x86 AVX2/AVX-512 and supported ARM NEON paths.
- High precision: exact uint16, float16 and float32 grayscale/RGB/RGBA samples in the compatible v7 outer container.
- Three effort profiles: `fast`, `balanced`, `maximum`.
- Desktop GUI, Codec Lab and multi-file batch queue.
- Optional CUDA/CuPy candidate-analysis path; CUDA is experimental and not required for the native C++ CPU backend.

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Build the C++17 backend (requires a compatible C++17 compiler):

```bash
python build_native.py
```

Launch the GUI:

```bash
python pxl_gui.py
```

Or use the Windows launcher `PXL_Studio.pyw`. If no native library is found, the program falls back to the compatible Python/Numba implementation.

## Command-line examples

```bash
python pxl.py compress input.png output.pxl --mode lossless --effort balanced --backend cpu
python pxl.py decompress output.pxl restored.png
python pxl.py info output.pxl
```

For high-precision TIFF input:

```bash
python pxl.py compress image16.tif image16.pxl --mode lossless
python pxl.py decompress image16.pxl restored16.tif
```

## Native C++ architecture

The native engine is a standalone C++17 implementation. The Python binding in `pxl_native.py` exposes a small C ABI through `ctypes`. Its main paths include fused reversible filtering, inverse filtering, custom rANS entropy coding, PixelLZ matching, block dictionaries, v8 blended rANS and variable-block search, and runtime SIMD scoring.

The Python implementation remains a compatibility fallback and a reference for differential tests. The native code is not a wrapper around a third-party compressor.

## Codec Lab and CUDA

Codec Lab compares PXL with codecs supported by the local Pillow build, including PNG, WebP, TIFF, QOI, AVIF and JPEG XL when optional plugins are installed. Lossless rows are checked by comparing decoded pixel bytes.

CUDA is optional and experimental. Install exactly one CuPy wheel matching the CUDA environment, then run:

```powershell
py -m pip install -U -r requirements-cuda12.txt
# or choose requirements-cuda13.txt
py diagnose_cuda.py
```

The CUDA path accelerates predictor/color-transform analysis only. Entropy coding, dictionaries, image I/O and v8 reconstruction remain CPU-side.

## Tests and benchmarks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python benchmarks_v80/run_benchmark.py --effort balanced --runs 3
python benchmarks_v80/run_benchmark.py --effort maximum --runs 3
```

See `benchmarks_v80/RESULTS.md` and its CSV files. Synthetic regression data is useful for repeatable testing but is not a substitute for a diverse camera-photo corpus.

## Limitations

- Maximum image area is 100 megapixels. Supported layouts are grayscale, RGB and RGBA.
- PXL preserves sample values, not the original source container or all metadata such as EXIF, ICC profiles, XMP or arbitrary TIFF tags.
- OpenEXR support is limited to single-part Y/RGB/RGBA scanline or tiled images.
- High-precision encoding is lossless-only. The 8-bit lossy DCT profile remains experimental.
- The native backend and AVX-512 path have been tested on Linux in the development environment; Windows/MSVC and macOS builds should be rebuilt on the target OS.
- CUDA remains unverified on the target GPU.

## Version compatibility

PXL Studio 8.0.2 writes PXL v8 files and reads PXL v1 through v8. High-precision files retain the v7 outer container with v8 byte-plane streams. The native C++ ABI is version 5; ABI-4 libraries from PXL Studio 7.0 must be rebuilt.
