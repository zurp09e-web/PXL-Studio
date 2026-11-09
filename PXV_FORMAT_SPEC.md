# PXV v1 binary format specification

Status: experimental, first implementation in PXL Studio 8.1.0.

PXV v1 is an indexed frame container whose frame payloads are independent PXL
v8 lossless image archives. It defines no temporal prediction syntax.
Readers must treat each embedded PXL payload as an opaque PXL archive and use
the PXL reader to decode it.

## Scope and guarantees

- Unsigned 8-bit samples only.
- Channels: 1 (gray), 3 (RGB), or 4 (RGBA).
- One fixed width, height, channel count and nominal frame rate per file.
- Presentation timestamps are signed microseconds, monotonically non-decreasing.
- Every frame is independently decodable.
- Header, index and every embedded PXL archive are protected by CRC32.
- Frame index offsets are absolute byte offsets from the start of the file.
- The current writer emits PXL v8 lossless frames.

PXV v1 does not store audio, subtitles, chapters, ICC/HDR metadata, source
codec information or the source container. The video-file frontend uses
FFmpeg to produce normalized 8-bit gray/RGB/RGBA frames. The decode_video
frontend exports FFV1 in Matroska at the stored nominal frame rate; it does not
currently recreate variable-frame-rate packet timing. Original PTS values
remain available from the PXV index.

CRC32 detects accidental corruption; it is not a cryptographic authenticity
or tamper-proofing mechanism.

## Integer and byte order

All integer fields are little-endian. Packed structures contain no padding;
writers must emit exactly the specified byte counts.

## File layout

    +-------------------------------+
    | PXV file header               | 48 bytes
    +-------------------------------+
    | Frame record 0                |
    |   frame record header         | 32 bytes
    |   PXL v8 payload              | payload_size bytes
    +-------------------------------+
    | Frame record 1                |
    |   ...                         |
    +-------------------------------+
    | Frame record N-1              |
    +-------------------------------+
    | Index header                  | 20 bytes
    +-------------------------------+
    | Index entry 0                 | 28 bytes
    | Index entry 1                 |
    | ...                           |
    | Index entry N-1               |
    +-------------------------------+

The index_offset in the file header points to the index header. The index
must be the final structure in the file. Frame records are contiguous, in
frame-number order, and must end exactly at index_offset.

## File header

Python struct format: <4sHHIIHHIIQQI> (48 bytes).

| Offset | Size | Field | Required value / meaning |
|---:|---:|---|---|
| 0 | 4 | magic | PXV followed by NUL |
| 4 | 2 | version | 1 |
| 6 | 2 | flags | 0 |
| 8 | 4 | width | 1..100,000; width × height ≤100,000,000 |
| 12 | 4 | height | 1..100,000; width × height ≤100,000,000 |
| 16 | 2 | channels | 1, 3, or 4 |
| 18 | 2 | reserved | 0 |
| 20 | 4 | fps_num | non-zero nominal frame-rate numerator |
| 24 | 4 | fps_den | non-zero nominal frame-rate denominator |
| 28 | 8 | frame_count | 1..1,000,000 |
| 36 | 8 | index_offset | absolute start offset of index header |
| 44 | 4 | header_crc32 | CRC32 of bytes 0..43 |

The CRC is computed over the first 44 bytes, excluding the CRC field itself.

## Frame record header

Python struct format: <4sQqQI> (32 bytes).

| Field | Meaning |
|---|---|
| magic | FRM1 |
| frame_number | zero-based unsigned 64-bit frame index |
| timestamp_us | signed 64-bit presentation timestamp in microseconds |
| payload_size | embedded PXL archive length in bytes |
| payload_crc32 | CRC32 of exactly the embedded PXL archive bytes |

The frame header is followed immediately by payload_size bytes of a PXL v8
lossless archive. Its PXL geometry, channel count and lossless profile must
match the enclosing PXV header.

## Index header

Python struct format: <4sQII> (20 bytes).

| Field | Meaning |
|---|---|
| magic | IDX1 |
| frame_count | must equal the PXV header frame_count |
| entry_size | must equal 28 |
| entries_crc32 | CRC32 of concatenated index entries only |

## Index entry

Python struct format: <qQQI> (28 bytes).

| Field | Meaning |
|---|---|
| timestamp_us | same signed timestamp as matching frame record |
| frame_offset | absolute offset of frame record header |
| payload_size | embedded PXL archive length |
| payload_crc32 | same CRC as frame record header |

Entry i describes frame i. The first frame offset equals the file header size.
Each frame record must end exactly at the next frame offset; the final frame
record must end at index_offset. Timestamps must be non-decreasing. The index
CRC protects the complete table; frame header and payload CRCs are checked
again before decode.

## Safety limits

- At most 1,000,000 frames.
- At most 100,000,000 pixels per frame.
- Dimensions at most 100,000 per axis.
- At most 512 MiB per embedded PXL frame archive.
- A per-frame archive ceiling derived from decoded pixel bytes, tile count,
  row framing overhead and a fixed allowance.
- Exact contiguous offsets and a terminal index.
- No unknown flags, unsupported versions, invalid channels, decreasing
  timestamps, malformed frame headers or mismatched embedded PXL geometry.

## Random access

A reader validates the header and entire index (including the index CRC) when
opening the file. To read frame i:

1. Read index entry i.
2. Seek to frame_offset.
3. Validate the frame record header against the index entry.
4. Read exactly payload_size bytes and validate payload_crc32.
5. Inspect the embedded PXL header and require matching lossless profile,
   dimensions and channels.
6. Decode that PXL archive with the PXL v8 decoder.

No preceding PXV frame needs to be decoded.

## Future versions

Do not add temporal prediction, multiple reference frames, 10/12/16-bit
samples, subsampling, audio or subtitle packets to v1. Those require a new
version with explicit feature flags, sample-format/chroma metadata, per-stream
indexes and compatibility rules.
