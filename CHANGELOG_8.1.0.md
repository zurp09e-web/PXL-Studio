# PXL Studio 8.1.0 — PXV v1

## Added

- New .pxv indexed lossless video container.
- PXV v1 stores independently decodable PXL v8 lossless frame archives.
- Frame-level microsecond timestamps and random-access index.
- CRC32 integrity checks for the PXV header, index and each embedded PXL frame archive.
- pxv command-line tool: encode, decode, info, and verify.
- Python API: encode_frames, encode_video, decode_frame, iter_frames,
  decode_video, inspect, and verify.
- FFmpeg/ffprobe import and FFV1 Matroska export.
- New PXL Studio PXV Video tab with encode, decode, inspect and verification.

## Limitations (intentional v1 scope)

- 8-bit gray/RGB/RGBA only.
- Intra-frame coding only: no temporal residuals, motion estimation or reference frames yet.
- No audio, subtitles, chapters, HDR metadata or 10/12/16-bit formats.
- FFmpeg import converts the source to normalized 8-bit frames; PXV guarantees
  exact reconstruction of those stored decoded samples, not the original
  source bitstream or source-container metadata.
- PTS values are indexed in PXV. Current FFV1 export uses the nominal frame
  rate and does not yet reconstruct variable-frame-rate timing.

## Validation

See TESTING_PXV_1.0.md. The PR adds frame round-trip, random access,
grayscale/RGBA, corruption, timestamp and optional FFmpeg integration tests.
The full CI result is tracked by the associated GitHub Actions workflow; no
test pass is claimed before that workflow completes.
