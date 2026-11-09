# PXV v1 testing

## Install

    python -m pip install -r requirements-dev.txt
    python build_native.py

For video-file integration tests and GUI video import/export, also install
FFmpeg and ffprobe and ensure both are on PATH.

## Run

    python -m pytest tests/test_pxv.py -q
    python -m pytest tests -q

## Coverage

- PXL v8 frame round-trip exactness.
- Random access to a later frame without decoding preceding frames.
- Sequential frame iterator.
- Gray, RGB and RGBA channel preservation.
- Header, index and payload corruption rejection.
- Monotonic timestamp enforcement and empty-stream rejection.
- Index and deep frame verification.
- Optional FFmpeg/FFV1 encode/decode round trip (skipped when FFmpeg is unavailable).

A passing inspect() or shallow verify() validates the PXV index and frame
payload checksums but does not decode all embedded PXL frames. Use
verify(path, deep=True) to decode each frame as well.
