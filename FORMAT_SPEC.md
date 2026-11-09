# PXL v3 — Predictive eXact Lossless / Lossy Format

Status: experimental reference specification. All multi-byte integers are little-endian.

PXL v3 keeps the v2 outer tile container but introduces a native entropy layer.
New v3 encoders do **not** emit DEFLATE, zlib, Brotli, Zstandard, or another
ready-made compressor. The implementation retains a bounded zlib decoder only
for reading old v1/v2 files.

## File header

Struct: `<4sBBBBIIHBB` (20 bytes)

| Field | Type | Meaning |
|---|---|---|
| magic | 4 bytes | `PXL\0` |
| version | u8 | `3` for this revision |
| profile | u8 | `0` lossless, `1` lossy |
| channels | u8 | `1` L, `3` RGB, `4` RGBA |
| flags | u8 | Must be zero |
| width | u32 | Image width |
| height | u32 | Image height |
| tile_size | u16 | Square tile edge |
| quality | u8 | `0` lossless; `1..100` lossy |
| reserved | u8 | Must be zero |

Tiles are ordered left-to-right, top-to-bottom. Edge tile dimensions are
implied by the image dimensions and tile size.

## Tile header

Struct: `<HHBBIII` (18 bytes)

| Field | Type | Meaning |
|---|---|---|
| tile_width | u16 | Actual tile width |
| tile_height | u16 | Actual tile height |
| codec | u8 | Tile codec ID (below) |
| transform | u8 | Reversible color transform ID |
| payload_size | u32 | Bytes following this header |
| decoded_size | u32 | Raw pixel bytes after decoding |
| crc32 | u32 | CRC32 of encoded tile payload |

## v3 tile codec IDs

| ID | Name | Meaning |
|---:|---|---|
| 0 | RAW_FILTERED | Row-filter stream without entropy coding |
| 2 | PALETTE_PACKED | Palette + bit-packed indices, no entropy layer |
| 3 | DCT | Lossy DCT tile; its internal streams use PXL entropy frames |
| 4 | RANS_ORDER0 | Row-filter stream encoded by static 12-bit byte rANS |
| 5 | RANS_RUN | PXL run-token prefilter, then order-0 rANS |
| 6 | PALETTE_ENTROPY | PXL entropy frame around the packed palette stream |
| 7 | RANS_CONTEXT4 | Four-context order-1 rANS |
| 8 | RANS_RUN_CONTEXT4 | PXL run-token prefilter, then four-context rANS |
| 9 | RAW_PIXELS | Original tile pixels verbatim; used for incompressible tiles |
| 10 | RANS_CONTEXT16 | Sixteen-context order-1 rANS (previous byte upper nibble) |
| 12 | RANS_SPATIAL | Pixel-layout-aware contexts: previous residual of the same channel, grouped into four magnitude ranges; predictor bytes have a separate model |
| 13 | LZ_RANS | PXL bounded-window match tokens + run tokens, followed by order-0 rANS |
| 14 | LZ_CONTEXT | PXL bounded-window match tokens + run tokens, followed by four-context rANS |

Codec 1 (DEFLATE) is reserved for legacy v1/v2 decoding and must not be emitted
by a v3 encoder. Codec IDs not listed here must be rejected.

## Reversible color transforms

Alpha is copied unchanged. Arithmetic for transforms 0–3 is modulo 256.

- 0: identity `(R,G,B)`
- 1: `(G, R-G, B-G)`
- 2: `(R, G-R, B-R)`
- 3: `(B, R-B, G-B)`
- 4: reversible modular YCoCg lifting:

  ```text
  Co = (R - B) mod 256
  t  = (B + floor(Co / 2)) mod 256
  Cg = (G - t) mod 256
  Y  = (t + floor(Cg / 2)) mod 256
  output = (Y, Cg, Co)

  t  = (Y - floor(Cg / 2)) mod 256
  G  = (Cg + t) mod 256
  B  = (t - floor(Co / 2)) mod 256
  R  = (B + Co) mod 256
  ```

## Prediction and residual stream

Each row begins with one predictor ID byte followed by one residual byte per
sample channel. Arithmetic is modulo 256. The same predictor is used across
all channels in that row. It is selected by measuring candidate residual
entropy across the complete interleaved row.

| ID | Predictor |
|---:|---|
| 0 | None: `0` |
| 1 | Sub: left sample |
| 2 | Up: above sample |
| 3 | Average: `floor((left+up)/2)` |
| 4 | Paeth |
| 5 | Gradient: `left+up-upper_left` |
| 6 | Median edge predictor (MED) |

For RGB/RGBA, left and upper-left refer to the same component of the previous
pixel. At boundaries, missing neighbours are zero. v3 uses one predictor ID
per row; v1/v2 use the same layout.

## PXL byte rANS entropy coder

PXL v3's custom entropy coder is a byte-oriented static rANS implementation.

- alphabet: 256 byte values
- frequency scale: 12 bits, total frequency `4096`
- lower renormalization state: `L = 2^23`
- symbols are encoded in reverse source order
- final state is stored as a 4-byte little-endian integer, followed by emitted
  renormalization bytes in forward decode order
- all frequencies are positive for symbols present in a model and sum exactly
  to 4096

### Order-0 frequency table

The first table byte selects the table representation:

- `0`: 256 little-endian `u16` frequencies (512 bytes)
- `1`: sparse table: one-byte symbol count (`0` means 256), then ascending
  `(symbol:u8, frequency:unsigned-varint)` pairs

After the table comes `state:u32`, then renormalization bytes.

### Context order-1 models

The four-context model uses the upper two bits of the previous source byte.
The sixteen-context model uses the upper nibble of the previous source byte.
The first byte uses context zero. The stream begins with the context count
(`4` or `16`). Each context then stores `table_size:u16` followed by an
order-0 table (zero size means unused). The final state and renormalization
bytes follow all context tables. The decoder derives each context from already
decoded output, so there is no per-symbol side channel.

## PXL run-token prefilter

The run transform is reversible and is applied before rANS only when its final
encoded candidate is smaller.

- Non-zero literal bytes are copied directly.
- `00 00` represents one literal zero.
- `00 01 varint(length)` represents a zero run of length at least 3.
- `00 02 value varint(length)` represents a repeated non-zero byte run of
  length at least 5.
- `00 03 varint(distance) varint(length)` represents a backward match with a
  distance of 1–65535 bytes and a length of at least 5. Overlapping copies are
  permitted, as in an LZ-style sliding-window dictionary.

Unsigned varints use 7 payload bits per byte and the high bit as continuation.

The LZ token encoder uses a 64 KiB bounded hash-chain search and compares the
completed token+rANS payload against the non-dictionary alternatives.

## Palette representation

The packed palette stream is:

- palette count `u8` (`0` means 256)
- index bit width `u8` (1–8)
- `count * channels` palette bytes
- tightly packed indices, least-significant bit first

Codec 2 stores this directly. Codec 6 wraps it in a PXL entropy frame.

## PXL entropy frame used inside lossy tiles

Every DCT coefficient/alpha substream is length-prefixed by the tile structure
and begins with a frame mode byte plus decoded-size `u32`:

- `0`: raw stream follows
- `1`: order-0 rANS stream follows
- `2`: decoded size + token size + run-prefilter/order-0-rANS stream
- `3`: four-context rANS stream follows
- `4`: decoded size + token size + run-prefilter/four-context-rANS stream

PXL selects the smallest of these candidates by actual encoded size.

## Lossy profile

Lossy mode uses an 8×8 orthonormal DCT-II, quantization, zigzag ordering,
DC-delta coding, and AC zero-run coding. The custom entropy frame described
above replaces the former DEFLATE wrapper. RGBA alpha is prediction-coded and
reconstructed losslessly. Individual tiles may use any exact lossless v3 codec
when it is smaller than their DCT representation.

The current RGB-to-YCbCr conversion and chroma resampling use Pillow's image
operations; the entropy coding itself is PXL-native.

## Compatibility

The reader accepts v1, v2, and v3 files. v1/v2 DEFLATE payloads are decoded by
a size-limited legacy compatibility path. A v3 encoder emits only v3 codecs
and does not write DEFLATE streams. Old v1/v2 decoders cannot read v3 files.

## Error handling

Readers reject unsupported versions, profiles, codecs, transforms, predictors,
invalid geometry, malformed entropy tables, invalid rANS states, impossible
palette indices, oversized streams, truncated data, CRC mismatches, and
trailing bytes. All entropy-decoding output sizes are bounded by tile geometry.

## Current limits

The reference implementation handles normalized 8-bit L/RGB/RGBA pixels up
to 100 megapixels. It does not preserve EXIF/ICC/text metadata or higher-bit-
depth input. Lossless reconstruction is bit-exact for normalized pixels.


### Pixel-layout-aware model (codec 12)

The spatial model does not condition on the immediately previous byte. For a
residual byte it uses the previous residual for the same component in the same
row, grouped by its upper two bits. Each component has four contexts; the row's
predictor-ID byte uses one dedicated context. Thus the context count is
`channels * 4 + 1`. The decoder derives the context from tile geometry and
already decoded residual bytes; no side channel is written beyond the context
tables.

# PXL v4 format supplement (experimental)

PXL v4 keeps the fixed v3 file header and tile-header layouts, but advances the
version byte to 4. Existing v1-v3 files remain decoded by the legacy reader.
New v4 lossless tiles select one of several exact representations by comparing
final payload byte counts.

## v4 tile codecs

| Codec ID | Name | Description |
|---:|---|---|
| 9 | RAW_PIXELS | Exact pixel bytes |
| 20 | PRED2D | Reversible per-channel row prediction + 17-context rANS |
| 21 | PIXEL_LZ | Pixel-tuple literals/runs/back-references + PXL rANS |
| 22 | BLOCK2D | 8x8 edge-aware block dictionary references + PXL rANS |
| 23 | PALETTE | Exact tile-local palette representation |
| 3 | DCT | Existing v3 lossy tile representation (lossy profile only) |

## PRED2D

The reversible color transform ID is stored in the tile header. The predictor
stream stores one predictor selector per channel at each row start followed by
interleaved channel residuals. Entropy contexts are selected from quantized left
and upper residual bytes; the row selector positions use a dedicated context.
The context model has 17 independent normalized frequency tables and one rANS
state. Decoding derives contexts only from already decoded bytes.

## PIXEL_LZ

The token stream operates on complete pixels (1, 3 or 4 bytes), not arbitrary
byte offsets. Tokens are:

- `0`: literal pixel block, followed by pixel count varint and pixel bytes
- `1`: repeated pixel, followed by count varint and one pixel tuple
- `2`: back-reference, followed by pixel distance varint and pixel count varint

The dictionary window is 512 KiB of prior pixel data. It persists across tiles
in raster tile traversal order. References can address prior tiles or earlier
pixels in the current tile. Match overlap is supported. Token bytes are encoded
with PXL's 16-context rANS implementation.

## BLOCK2D

The tile is traversed in 8x8 blocks, with smaller edge blocks when needed. Each
block is represented as either literal block bytes (`0`) or a reference to a
previous identical block (`1` plus a block-distance varint). The block history
is global across tiles and bounded to 32,768 blocks. The block shape is part of
the match identity, and the decoder verifies the referenced shape.

## Cross-tile state and determinism

PXL v4 reference codecs require tile traversal order and are decoded
sequentially. The pixel history advances for every tile; indexing into the
pixel match table may be skipped for screened high-entropy or flat tiles using
the history-screening flag below. The block dictionary is updated for every
tile. CPU predictor analysis may run in worker processes, but global
references are committed serially in raster order.
CUDA accelerates the predictor/color-transform candidate only when CuPy is
available; tokenization, rANS and dictionary management remain on the CPU.

## Integrity and exactness

The tile header CRC32 covers the encoded payload. The decoded-size field must
equal `tile_width * tile_height * channels`. Lossless PXL v4 supports exact
8-bit L, RGB and RGBA pixel reconstruction. Metadata and original source-file
encoding are not preserved.

## Pixel-history screening flag

The high bit (`0x80`) of the tile-header transform byte is a v4-only flag for
self-contained lossless tiles (raw, palette, block-reference, or a retained
PXL-native legacy tile). When set, the decoder advances the pixel history
window with this tile but does not immediately index its pixels into the
pixel-LZ match table. A later tile may index still-retained bytes before
matching. The low seven bits remain the color-transform ID. This avoids
building expensive pixel hash chains for screened flat/high-entropy tiles while
preserving deterministic encoder/decoder state. The flag is invalid on
PIXEL_LZ and PRED2D tiles.

# PXL v5 format supplement (experimental)

PXL v5 advances the header version byte to `5`. The fixed 20-byte image header
and 18-byte tile header remain the same as the v4 layout. All v4 tile codec IDs
and their token grammars remain valid. This keeps v4-compatible tile payloads
usable while v5 introduces an optional independently coded entropy container
for long token streams.

## Native implementation note

The C++17 shared library is an implementation optimization, not a platform-
specific part of the bitstream. A v5 file can be decoded by any conforming
implementation of this specification, including the Python fallback. Native
and fallback encoders must produce valid equivalent streams; where their
candidate selection is configured identically, the reference build aims for
byte-identical output.

## PXMS independent entropy chunks

The `PIXEL_LZ` (codec 21) and `BLOCK2D` (codec 22) token payloads begin with a
little-endian u32 decoded-token length. The legacy single-stream representation
continues to store one PXL context16-rANS frame after this length.

For eligible larger streams, v5 may instead store the marker `PXMS` after the
decoded-token length:

| Offset after token length | Field | Size |
|---:|---|---:|
| 0 | ASCII marker `PXMS` | 4 bytes |
| 4 | nominal raw chunk size | u32 |
| 8 | chunk count | u32 |
| 12 | repeated chunk records | variable |

Each chunk record contains `raw_chunk_length` (u32), `encoded_chunk_length`
(u32), followed by one independent PXL context16-rANS frame. Chunks are
concatenated in original token order. Their decoded lengths must sum exactly to
the initial token length. A decoder must reject invalid sizes, too many chunks,
truncated records, trailing data and chunk totals that do not match the declared
length.

The encoder compares the complete `PXMS` serialization against the complete
single-stream representation and writes the chunked form only when it is
strictly smaller. A decoder may entropy-decode these chunks concurrently, then
concatenate them in record order before performing the sequential dictionary
expansion. Dictionary-dependent PixelLZ and block-reference reconstruction
still follows raster tile and token order; only independent entropy frames are
parallelized.

## Version compatibility

- The v5 reader accepts v4 and v5 headers and dispatches `PXMS` only when the
  actual file version is 5.
- The v4 reader does not understand `PXMS`; therefore the v5 writer must never
  write this marker into a v4 file.
- PXL v1-v3 files continue through their legacy decoding paths. The bounded
  DEFLATE compatibility code is retained solely for reading historical files;
  the v5 writer does not emit DEFLATE or call an external compression library.

## Determinism and integrity

Each tile retains the v4 tile-header CRC32 over its encoded payload and the
expected decoded byte count. The optional chunk records are contained inside
that tile payload and are consequently covered by the same CRC. Dictionary
updates and tile traversal remain deterministic. Parallel chunk completion
order must not affect the concatenated token order or file bytes.


# PXL v6 format supplement (adaptive spatial contexts)

PXL v6 increments the one-byte container version to `6`. The fixed image and
tile headers are unchanged from v4/v5. Existing codec IDs 0-23 retain their
previous payload grammars. PXL v6 adds two self-contained lossless predictor
codecs:

- `24` — signed-small-residual 2D context rANS (17 models).
- `25` — triple-neighbour residual context rANS (65 models).

Both payloads use the codec-20 predictor payload layout: a little-endian u32
filtered-stream byte count, followed by a PXL-owned rANS frame. The decoded
filtered stream has `height * (width * channels + channels)` bytes. Each row
begins with `channels` predictor-selector bytes, then the filtered residual
samples. The predictor and reversible color-transform reconstruction are the
same as codec 20; only entropy context selection changes.

## Codec 24: signed-small-residual contexts

The rANS frame's first byte is the context count, exactly `17`. Context 16 is
used for row predictor-selector bytes. For each residual byte, classify its
left and upper residual neighbours as follows:

| Residual byte | Class |
|---:|---:|
| `0` | 0 |
| `1` (`+1`) | 1 |
| `255` (`-1`) | 2 |
| all other values | 3 |

The context ID is `left_class * 4 + up_class`, in the range 0-15. Missing
neighbours at the left/top image boundary are treated as residual zero.

## Codec 25: triple-neighbour contexts

Codec 25 is used by the maximum-effort encoder when its full encoded payload is
smaller. The rANS frame's first byte is exactly `65`. Context 64 is used for
row predictor-selector bytes. The same four-class mapping as codec 24 is used
for three reconstructed residual neighbours: left, up and upper-left. The
context ID is `left_class * 16 + up_class * 4 + upper_left_class`, in the range
0-63. Missing boundary neighbours are residual zero.

The encoder may decide not to use either new codec. It compares complete
serialized payload sizes against the existing predictor, PixelLZ, block,
palette and raw candidates and selects the smallest. The signed 17-model search
is available in fast/balanced/maximum profiles with a residual-distribution
screen. The larger 65-model search is restricted to maximum effort. These are
encoder choices, not decoder heuristics: the codec ID fully identifies the
context model used by a tile.

## Compatibility

A PXL v6 reader accepts headers v4, v5 and v6. It accepts codec IDs 24 and 25
only when the file version is 6. A v4/v5 reader cannot decode those new codec
IDs and must reject them as unsupported. The C++ native library is not part of
the on-disk format; ABI version 4 adds matching decoder entry points. A
portable Python decoder remains available when the native library is absent.

PXL v6 writing remains entirely custom: it does not invoke DEFLATE, zlib,
Zstandard, Brotli, LZ4, JPEG, WebP or JPEG XL as an internal compressor.


# PXL v7 format supplement (context mixing, variable blocks and high precision)

PXL v7 increments the version byte to `7`. Existing v4/v5/v6 8-bit tile and
header layouts are retained. A v7 decoder must accept legacy codec IDs and
understand the two new tile codec IDs below. New v1-v6 files are unchanged.

## Codec 26: blended context rANS

Codec 26 uses the same predictor-filtered byte stream and row-selector layout
as codec 20. It computes a deterministic context ID for each filtered symbol
from local signed-residual classes, the channel position and a global context.
The encoder blends normalized frequency counts from these context sources,
serializes a standard PXL rANS table and payload, and selects codec 26 only
when its complete payload is smaller than every other candidate. It is an
encoder choice, not a decoder heuristic. The decoder derives identical IDs
from already reconstructed residuals.

## Codec 27: variable-size exact block references

Codec 27 is a lossless token stream with a persistent cross-tile dictionary.
It searches exact 32x32 and 16x16 regions before falling back to 8x8 literals.
A reference stores a size code and backward dictionary distance. Every hash
match is verified against the complete block bytes before encoding; hash
collisions cannot change decoded pixels. Dictionary state is updated in
raster tile order. The encoder screens for available large matches before
cloning/searching this dictionary and selects codec 27 only by complete
serialized payload size. It does not yet implement approximate/delta block
references.

## High-precision v7 outer container

High-precision files use the fixed 20-byte PXL image header (`<4sBBBBIIHBB`)
with version `7` and lossless profile. The `flags` field encodes the sample
format in its low two bits and a reversible 16-bit RGB transform in the
remaining bits:

| flags low bits | Sample representation | Bytes per sample |
|---:|---|---:|
| 1 | unsigned 16-bit integer | 2 |
| 2 | IEEE-754 binary16 (float16) | 2 |
| 3 | IEEE-754 binary32 (float32) | 4 |

For uint16 RGB/RGBA, transform IDs 0-4 mean identity, green-base differences,
red-base differences, blue-base differences, and reversible YCoCg-R. Alpha is
left unchanged. Floating-point values are never numerically transformed; their
bit patterns are preserved.

The `quality` header byte stores bytes per sample (2 or 4), and `reserved`
stores the number of byte planes (2 or 4). Immediately after the image header
are `plane_count` entries, each `<QI>`: the encoded PXL byte-plane stream
length (u64) followed by CRC32 of that decoded byte plane (u32). The embedded
PXL v7 streams follow in plane order. A byte-plane is formed by taking the
same significance byte from every sample and arranging it as an 8-bit L, RGB
or RGBA image. Each plane is an independently decodable ordinary 8-bit PXL
v7 stream. After decoding, the plane CRC is checked, byte planes are
interleaved back into sample bytes, and any reversible uint16 colour transform
is inverted. The result is bit-exact for supported samples.

This profile currently supports grayscale, RGB and RGBA; uint16, float16 and
float32; and images within the common PXL dimension/pixel safety limits. The
container stores pixel samples, not source metadata. EXR I/O is an optional
implementation feature and does not alter the PXL bitstream.

## Compatibility and implementation notes

A v7 reader accepts v1-v7 files. A v4-v6 reader cannot decode codec IDs 26/27
or the high-precision outer-container flags. The PXL native C++ ABI is not a
part of the format; codec 26/27 candidate generation and high-precision
byte-plane orchestration are currently implemented in portable Python, while
the established C++ filtering and entropy primitives remain available to the
8-bit engine.

# PXL v8 format supplement (adaptive rectangles and independent history groups)

PXL v8 increments the ordinary 8-bit image-container version byte to `8`.
The 20-byte image header remains the existing `<4sBBBBIIHBB>` layout. PXL v1-v7
files and tile records are unchanged and remain readable. The v8 writer uses a
new 26-byte tile record, `<IIHHBBIII>`:

| Field | Type | Meaning |
|---|---|---|
| `x`, `y` | u32, u32 | Tile origin in image pixels |
| `width`, `height` | u16, u16 | Tile rectangle dimensions |
| `codec` | u8 | Tile codec ID |
| `transform_flags` | u8 | History/pixel-index flags and low-six-bit color transform |
| `payload_size` | u32 | Encoded payload byte count |
| `decoded_size` | u32 | Expected decoded byte count |
| `crc32` | u32 | CRC32 of the encoded payload |

Each tile rectangle is bounded by the image dimensions and configured maximum
tile size. Internal tile origins are aligned to eight pixels; edge rectangles
may have smaller or non-aligned extents. The decoder validates that all
rectangles form a complete image partition with no gaps or overlaps before it
reconstructs pixels.

## Tile transform flags

- Bits 0-5: reversible color-transform ID (currently 0-4).
- Bit 6 (`0x40`): this tile starts a new independent reference-history group.
- Bit 7 (`0x80`): the encoder advanced pixel-history bookkeeping without
  inserting this tile into the PixelLZ pixel index.

The first tile must set bit 6. By default, a new group starts every eight tiles;
the encoder may store a different cadence implicitly through these marker
bits. At each group boundary, PixelLZ, exact-block and variable-block reference
histories are reset. Tiles remain sequential within a group, but independent
groups can be decoded concurrently. This preserves cross-tile references
inside each group while allowing bounded parallel reconstruction. A decoder
must not infer group boundaries from tile count; it must read bit 6.

## Codec 26: native blended-context rANS

Codec 26 uses the predictor-filtered stream layout of codec 20: a little-endian
u32 filtered byte count followed by a PXL rANS frame. For each symbol, the
context ID is derived from already reconstructed left/up residual classes,
channel position and global residual statistics. The normalized model blends
local two-dimensional, channel-specific and global frequency contributions.
The encoder compares the complete payload with the other exact tile
representations and uses codec 26 only when it is smaller. Native C++ and
portable Python implementations use the same deterministic normalization and
produce byte-identical streams.

## Codec 27: variable blocks with exact and delta references

Codec 27 is a sequence of block tokens. Tokens begin with an operation byte
and a block-size code (`0` = 8x8, `1` = 16x16, `2` = 32x32), followed by the
operation-specific data:

- Operation `0`: literal block bytes. The size code is 8x8; partial image-edge
  literals use their actual edge dimensions.
- Operation `1`: exact reference, followed by a backward dictionary distance
  encoded as an unsigned base-128 varint.
- Operation `2`: reversible delta reference, followed by the same backward
  distance and a delta stream.

The delta stream alternates unsigned varint zero-difference run lengths and
changed-byte run lengths. Each changed byte is the modulo-256 difference
`current_byte - reference_byte`. Reconstruction adds that difference modulo
256 to the referenced block. This is exact for all byte values. The search
checks candidate geometry, uses SIMD-assisted byte-equality scoring as a
prefilter, and keeps a delta candidate only when its serialized token is
smaller than the literal. Hash matches for exact references are verified
against full block bytes. The dictionary is deterministic and ordered by tile
and block traversal.

## Native acceleration and portability

The v8 native ABI is version 5; it is an implementation detail and does not
change the file format. Native code performs blended-context rANS model
construction/decoding and variable-block exact/delta candidate search. The
match-scoring path selects among AVX-512BW, AVX2, ARM NEON or scalar code at
runtime where compiled and supported. SIMD currently accelerates equality
scoring used by delta candidate screening; it is not a blanket SIMD rewrite of
all prediction, entropy or image-processing kernels. A portable Python path
remains available and is covered by differential round-trip tests.

## High-precision outer container

High-precision sample files retain the PXL v7 outer high-precision envelope
and flags for compatibility with the established sample representation. New
PXL Studio 8 writers encode each 8-bit byte plane with ordinary PXL v8 streams;
older files may contain v7 byte-plane streams. The reader accepts the embedded
stream versions supported by the main decoder. The outer uint16/float16/float32
sample layout, reversible integer color transforms, byte-plane CRCs and
bit-exact floating-point reconstruction are unchanged.

## Compatibility

A PXL v8 reader accepts v1-v8 standard files. A v1-v7 decoder cannot decode
the new coordinate-bearing v8 tile records. V8 readers continue to decode
legacy fixed-grid tile records for v4-v7. Codec IDs 26 and 27 retain their
previous v7 meaning for v7 files; the version byte selects the v8 token
extensions and tile framing. No older file is rewritten or upgraded in place.
