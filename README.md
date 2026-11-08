# PXL Studio 8.0.2 — Adaptive Lossless and High-Precision Image Codec

PXL Studio is a Python/Tkinter image-compression workbench backed by a custom
C++17 codec engine and a portable Python fallback. Python provides the GUI,
image I/O, batch queue and file orchestration. The native engine now performs
the blended-context rANS model construction/decoding, variable-block search,
exact/delta block coding and SIMD match scoring; the Python implementation
remains available as a reference and compatibility fallback.

PXL writes its own `.pxl` format. New PXL v8 8-bit files do **not** use
zlib/DEFLATE, Zstandard, Brotli, LZ4, JPEG, WebP or JPEG XL as an internal
compression backend. PXL owns its reversible transforms, pixel and block
references, predictors, token grammar, frequency models, rANS entropy coder,
adaptive tile layout and tile selection. Pillow and other optional packages
are used for image I/O and Codec Lab comparisons, not for PXL's internal
lossless compression.

PXL v8 is experimental. It is not claimed to outperform mature codecs on all
images. On the five-image synthetic regression set, v8 is 300 bytes larger than
the v7 baseline in both Balanced and Maximum profiles because the new tile
records carry explicit coordinates. The added adaptive-layout sample is 4,662
bytes (1.17%) smaller than its fixed-tile v7 baseline. The delta-reference path
is exercised and round-tripped exactly, but the current near-repeat sample is
not smaller than v7. See `benchmarks_v80/RESULTS.md` for the measured results
and limitations. High-precision files retain the v7 outer container and encode
their byte planes with the v8 engine.

## Highlights

- **PXL v8 container**, with readers for older PXL v1–v7 archives.
- **Native C++17 engine** for blended context rANS, variable-block search,
  reversible delta tokens and reference scoring, alongside established
  predictor/filtering primitives.
- **Blended context model:** local 2D, per-channel and global statistics are
  combined to form byte probabilities; selected only when the completed
  payload is smaller than competing candidates.
- **Variable block references:** 8×8, 16×16 and 32×32 blocks with collision-
  checked exact references and reversible delta references for similar blocks.
- **Adaptive tile segmentation:** recursively split statistically different
  regions, store explicit tile rectangles, and validate complete non-overlap
  image coverage before decoding.
- **Independent history groups:** checkpoint reference dictionaries so groups
  can be decoded concurrently without changing pixels; tune the group interval.
- **Runtime SIMD match scoring:** scalar fallback, GCC/Clang AVX2 and AVX-512,
  separate MSVC AVX2/optional AVX-512 kernels, and ARM NEON where the compiler
  exposes the intrinsics. SIMD currently accelerates equality scoring in the
  delta-match search, not every codec operation.
- **High precision:** exact uint16, float16 and float32 grayscale/RGB/RGBA
  samples, including floating-point bit patterns, in the compatible v7 outer
  container with v8 byte-plane streams.
- **Reversible 16-bit RGB decorrelation:** identity, G/R/B-base differences
  and YCoCg-R are tested by final compressed size for integer input.
- **Three effort profiles:** `fast`, `balanced`, `maximum`.
- **Exact lossless reconstruction** for supported 8-bit and high-precision
  samples. The existing experimental 8-bit lossy profile remains available.
- **Desktop GUI, Codec Lab and multi-file batch queue** with per-job and overall
  progress, pause/resume and configurable concurrency.
- Optional CUDA/CuPy candidate-analysis path; CUDA acceleration remains
  experimental and is not required for the native C++ CPU backend.

## Quick start

### 1. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Build the C++17 backend

You need a C++17 compiler:

- **Windows:** Visual Studio Build Tools with the Desktop development with C++
  workload (`cl.exe`), or MinGW-w64 (`g++`).
- **Linux:** `g++` (or another compatible C++17 compiler).
- **macOS:** Xcode Command Line Tools (`clang++`).

From the extracted project directory:

```bash
python build_native.py
```

The script builds `pxl_native.dll`, `libpxl_native.so` or
`libpxl_native.dylib`, depending on the operating system.

### 3. Launch PXL Studio

```bash
python pxl_gui.py
```

Or on Windows, double-click `PXL_Studio.pyw` after building the native library.
The startup log identifies the active backend. If no native library is found,
the program falls back to the compatible Python/Numba implementation; run
`python build_native.py` to enable the C++ engine.

## Command-line use

Compress losslessly with balanced effort:

```bash
python pxl.py compress input.png output.pxl --mode lossless --effort balanced --backend cpu
```

Use the faster search profile:

```bash
python pxl.py compress input.png output.pxl --effort fast
```

Spend more time searching for a smaller representation:

```bash
python pxl.py compress input.png output.pxl --effort maximum
```

Compress using the experimental lossy profile:

```bash
python pxl.py compress input.png output.pxl --mode lossy --quality 88 --effort balanced
```

Decode and inspect an ordinary 8-bit image:

```bash
python pxl.py decompress output.pxl restored.png
python pxl.py info output.pxl
```

Compress and restore a 16-bit TIFF without reducing its sample precision:

```bash
python pxl.py compress image16.tif image16.pxl --mode lossless --effort balanced
python pxl.py decompress image16.pxl restored16.tif
```

For OpenEXR input/output, install the optional bindings first:

```bash
python -m pip install -r requirements-hdr.txt
python pxl.py compress render.exr render.pxl --mode lossless
python pxl.py decompress render.pxl restored.exr
```

Use four CPU workers, or all logical CPU cores (`--workers -1`):

```bash
python pxl.py compress input.png output.pxl --workers 4
python pxl.py compress input.png output.pxl --workers -1
```

## GUI and batch processing

The GUI includes single-image Encode/Decode, Preview, Codec Lab and Batch Queue
workspaces. The Batch Queue can run several files at once in separate spawned
processes. Each file displays a progress bar, current stage, elapsed time and
estimated remaining time. Overall progress is weighted by the image work size.
The queue supports pause/resume, removing queued jobs, clearing completed jobs,
an output-directory selector and an overwrite safeguard.

When using the batch queue, each file process uses one codec worker to avoid
nested process pools. Set the number of simultaneous files to use additional
CPU cores. Single-image operations can use multiple tile-analysis workers.
CUDA jobs are serialized by the batch queue to avoid competing GPU contexts.

## Compression profiles

- **Fast:** evaluates a smaller candidate set and uses larger independent entropy
  chunks. Intended for quick previews and large batch jobs.
- **Balanced:** default search breadth and chunk size; a practical size/speed
  compromise.
- **Maximum:** evaluates more reversible transforms and uses smaller entropy
  chunks, increasing search and parallel coding opportunities at higher CPU
  cost.

All three profiles remain lossless in lossless mode. They can choose different
representations, so their file sizes may differ; they never alter the source
samples. The lossy mode is explicitly approximate for color samples, while
alpha is retained exactly.

## Native C++ architecture

`native/pxl_native.cpp` is a standalone C++17 implementation using only the
standard library. The Python binding in `pxl_native.py` exposes a small C ABI
through `ctypes`.

The main native paths are:

1. **Fused filter:** reversible color transform, per-channel predictor search,
   residual emission and 17-context ID generation in one pass.
2. **Inverse filter:** residual reconstruction and inverse color transform in
   one native pass.
3. **Entropy:** custom 12-bit byte rANS frequency normalization, encoding and
   table-based decoding. Large token streams can use independently encoded
   `PXMS` chunks; the container only keeps chunked framing when the full result
   is smaller than the unchunked form.
4. **PixelLZ:** persistent open-addressed match index and rolling cross-tile
   history, with complete-pixel literals, runs and references.
5. **Block dictionary:** persistent exact block hashes with byte verification
   to reject hash collisions.
6. **PXL v8 blended rANS and variable blocks:** context construction/decoding,
   exact/delta block search and byte-equality scoring are implemented natively.
7. **SIMD scoring:** runtime AVX-512BW/AVX2/NEON selection with scalar fallback
   for byte-equality scoring in approximate block matching.

The Python implementation remains a compatibility fallback and a reference
for differential tests. The native code is not a wrapper around a third-party
compressor.

## Codec Lab

Codec Lab can compare PXL with codecs supported by the local Pillow build,
including PNG, WebP, TIFF, QOI, AVIF and JPEG XL when their optional plugins
are installed. Lossless rows are checked by comparing decoded pixel bytes.
Lossy comparisons report size and image-quality metrics, but quality numbers
from different codecs are not directly equivalent. Only codecs available in the local Pillow build are included.

## High precision and HDR

The compatible high-precision outer container introduced in PXL v7 supports **uint16**,
**float16** and **float32** samples. PXL Studio 8 keeps that v7 outer envelope
for compatibility, but its byte-plane images are now compressed using the v8
adaptive tile and independent-history-group engine. It byte-shuffles each
sample-significance plane into an 8-bit image. This preserves all
sample bits, including floating-point sign bits, signed zero and NaN payloads;
it does not tone-map, quantize, or change linear/HDR values. Supported layouts
are grayscale, RGB and RGBA. TIFF is the recommended container for multi-channel
integer and floating-point input/output. 16-bit PNG grayscale, RGB and RGBA
can also be read and written.

TIFF and 16-bit PNG support are installed through `requirements.txt`; PyPNG
handles 16-bit RGB/RGBA PNG without reducing sample depth. For
single-part OpenEXR scanline/tiled images, install the optional Python bindings:

```bash
python -m pip install -r requirements-hdr.txt
```

The high-precision path reads/writes EXR Y, RGB and RGBA channels in HALF/FLOAT formats.
Unsigned 16-bit integer samples should be exported as TIFF or PNG, not EXR.
Multipart/deep EXR, extra channels such as Z/depth, and channel subsampling are
rejected rather than silently discarded.

High-precision images are currently lossless-only. The v7 outer container stores the sample
values, not complete TIFF/EXR metadata such as camera tags, arbitrary TIFF
custom tags, ICC profiles, or layered/multipart data. OpenEXR multipart/deep
images, extra data channels, animation and image stacks are not supported by
this first HDR profile.

Install optional JPEG XL support:

```bash
python -m pip install -r requirements-jxl.txt
```

## PXL v8 codec engine

PXL v8 keeps the earlier v1-v7 readers and adds a new coordinate-based tile
layout. Its key changes are:

- **Native context mixing:** codec 26's local, per-channel and global frequency
  blending is now implemented in C++17, with the Python implementation kept
  as a byte-compatible fallback.
- **Native variable-block search:** codec 27 searches 32x32, 16x16 and 8x8
  blocks in C++, using collision-checked exact matches and reversible modular
  delta references. Delta references encode unchanged spans as zero runs and
  changed bytes as literal runs; decoded samples remain exact.
- **Adaptive tile segmentation:** tile regions may split into smaller
  rectangles where local activity differs strongly. The boundaries are aligned
  to eight pixels, and the v8 header records explicit coordinates and sizes.
- **Parallel reference reconstruction:** a configurable number of tiles share
  each cross-tile dictionary group. Group starts reset reference state, making
  groups independently decodable by multiple CPU workers. Smaller groups can
  improve decode parallelism at the cost of dictionary reuse and extra headers.
- **SIMD match scoring:** runtime-dispatched AVX-512BW/AVX2 byte-comparison
  kernels on supported x86 builds and NEON on supported ARM builds, with a
  scalar fallback. MSVC builds compile ISA-specific code separately and check
  CPU/OS support before calling it.
- **High precision retained:** the v7 outer container still preserves uint16,
  float16 and float32 sample bit patterns; its embedded byte planes now use the
  v8 codec engine.

The GUI exposes adaptive segmentation and decode-group size. The CLI exposes
`--fixed-tiles` and `--checkpoint-interval`. For normal use, keep adaptive
segmentation enabled and leave the decode-group interval at 8. Use smaller
intervals such as 1 or 2 when parallel decode matters more than maximum history
reuse.

The SIMD implementation currently accelerates delta-block candidate scoring;
it is not yet a blanket SIMD rewrite of all predictors, color transforms or
entropy routines. Adaptive segmentation uses a gradient-activity heuristic,
not a global optimal partitioner. Approximate block matching remains restricted
to the deterministic lossless delta-reference grammar.

## CUDA (optional)

The native C++ CPU backend is the default accelerated engine. CUDA remains an
optional experimental path for supported predictor/color-transform candidate
operations. Install exactly one CuPy wheel matching your CUDA installation. If CuPy warns
that multiple packages such as `cupy` and `cupy-cuda13x` are present, remove all
CuPy distributions first; the CUDA-specific wheel provides the `cupy` import:

```powershell
# Clean conflicting generic / CUDA-specific CuPy distributions:
py -m pip uninstall -y cupy cupy-cuda12x cupy-cuda13x

# Choose ONE based on your CUDA driver/runtime family:
py -m pip install -U -r requirements-cuda12.txt
# or
py -m pip install -U -r requirements-cuda13.txt

py -m pip check
# Run a real device + PXL encoder smoke test:
py diagnose_cuda.py
```

`build_native.py` builds only the C++ CPU/SIMD library; it does not compile or
install CUDA. The optional GPU path is provided by CuPy and accelerates PXL
predictor/color-transform analysis only. Entropy coding, dictionaries, image
I/O and v8 reconstruction remain CPU-side, so GPU utilization can be brief and
low on small images. Codec Lab uses CPU decoding and automatically falls back
to CPU encoding if the optional CUDA path fails, while recording the reason.

Use one matching CuPy wheel, not both CUDA variants. Current CuPy installation
documentation provides Windows wheels for CUDA 12.x/13.x and a `[ctk]` extra
that installs CUDA runtime components through Python packages, although a
compatible NVIDIA display driver is still required. See the official
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Tests and benchmarks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Reproduce the PXL v8 tests and synthetic benchmarks:

```bash
python -m pytest tests -q
python benchmarks_v80/run_benchmark.py --effort balanced --runs 3
python benchmarks_v80/run_benchmark.py --effort maximum --runs 3
```

See `benchmarks_v80/RESULTS.md` and its CSV files. They include five matched
synthetic regression images, a near-repeat/delta sample and a smooth/noisy split
sample. These are useful regression tests, not a substitute for a diverse
camera-photo corpus. Compare PXL with PNG/WebP/JPEG XL separately in Codec Lab.

## Limitations and data preservation

- Maximum image area is 100 megapixels. Supported layouts are grayscale, RGB
  and RGBA; high-precision samples are uint16, float16 and float32.
- PXL preserves sample values, not the original source container or all its
  metadata. EXIF, ICC profiles, XMP, TIFF private tags, animation and layered
  image information are not preserved.
- OpenEXR support is limited to single-part Y/RGB/RGBA scanline or tiled images;
  deep/multipart images, subsampled channels and extra channels (for example
  Z/depth) are rejected.
- The high-precision profile is lossless-only. The 8-bit lossy DCT profile
  remains experimental and its quality control is not calibrated to match
  JPEG, WebP, AVIF or JPEG XL quality values.
- Codec Lab's automatic comparison pipeline is currently limited to normalized
  8-bit images; high-precision inputs can still be compressed/decompressed via
  the main GUI or CLI, but need a dedicated HDR comparison workflow.
- Lossless guarantees exact decoded pixel samples after normalization to a
  supported sample layout; it does not guarantee identical source-file bytes
  or metadata.
- PXL has not undergone an independent security audit. Do not treat this
  experimental decoder as a hardened parser for untrusted files.
- The native backend and AVX-512 path have been built and tested on Linux in
  this environment. Windows/MSVC and macOS builds are provided but have not
  been compiled here; rebuild the native library on the target OS. CUDA also
  remains unverified on the target GPU in this run.

## Version compatibility

PXL Studio 8.0.2 writes standard PXL v8 files and reads PXL v1 through v8. The
v8 tile record stores explicit x/y coordinates and per-tile width/height,
codec, flags, payload size, decoded size and CRC32. Tile records are grouped
into independent reference-history segments. Codec 26 is blended context rANS;
codec 27 supports exact and reversible delta block references. The v8 decoder
validates the full rectangle partition before reconstruction.

High-precision files retain the v7 outer container for uint16/float16/float32
sample fidelity, with checksummed byte-plane streams encoded by the v8 8-bit
engine. Existing v1-v7 files do not require conversion.

The native C++ ABI is version 5. ABI-4 libraries from PXL Studio 7.0 must be
rebuilt. On Windows, run `py build_native.py` in a Visual Studio Developer
Command Prompt. The builder compiles AVX2 and, when supported by the installed
compiler, AVX-512 kernels in separate object files. Runtime CPU/OS checks choose
the highest supported implementation or scalar fallback.
