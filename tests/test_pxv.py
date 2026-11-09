from pathlib import Path
import shutil
import struct

import numpy as np
import pytest

import pxv


def _encode_sample(path: Path, channels=3):
    rng = np.random.default_rng(1234)
    shape = (23, 29) if channels == 1 else (23, 29, channels)
    frames = [rng.integers(0, 256, size=shape, dtype=np.uint8) for _ in range(3)]
    timestamps = [0, 41_708, 83_417]
    pxv.encode_frames(
        list(zip(frames, timestamps)), path, width=29, height=23,
        fps_num=24_000, fps_den=1_001, channels=channels,
        tile_size=64, effort="fast",
    )
    return frames, timestamps


def test_pxv_lossless_roundtrip_and_random_access(tmp_path):
    archive = tmp_path / "sample.pxv"
    frames, timestamps = _encode_sample(archive)
    info = pxv.inspect(archive)

    assert info["format"] == "PXV"
    assert info["version"] == 1
    assert info["frame_count"] == 3
    assert info["channels"] == 3
    assert info["temporal_compression"] is False
    assert info["audio"] is False
    assert info["fps"] == "24000/1001"

    # Seeking to frame 2 must not require decoding frame 0 or frame 1.
    selected = pxv.decode_frame(archive, 2)
    assert np.array_equal(np.asarray(selected), frames[2])

    decoded = list(pxv.iter_frames(archive))
    assert len(decoded) == len(frames)
    for expected, actual in zip(frames, decoded):
        assert np.array_equal(np.asarray(actual), expected)

    assert pxv.verify(archive)["verified"] is True
    assert pxv.verify(archive, deep=True)["deep"] is True


def test_pxv_preserves_rgba_alpha_and_grayscale(tmp_path):
    rgba_path = tmp_path / "rgba.pxv"
    rgba_frames, _ = _encode_sample(rgba_path, channels=4)
    assert np.array_equal(np.asarray(pxv.decode_frame(rgba_path, 1)), rgba_frames[1])

    gray_path = tmp_path / "gray.pxv"
    gray_frames, _ = _encode_sample(gray_path, channels=1)
    assert np.array_equal(np.asarray(pxv.decode_frame(gray_path, 0)), gray_frames[0])


def test_pxv_rejects_header_corruption(tmp_path):
    archive = tmp_path / "bad-header.pxv"
    _encode_sample(archive)
    data = bytearray(archive.read_bytes())
    data[8] ^= 1
    archive.write_bytes(data)
    with pytest.raises(ValueError, match="header checksum"):
        pxv.inspect(archive)


def test_pxv_rejects_index_checksum_corruption(tmp_path):
    archive = tmp_path / "bad-index.pxv"
    _encode_sample(archive)
    data = bytearray(archive.read_bytes())
    header = pxv.HEADER.unpack_from(data)
    index_offset = header[-2]
    data[index_offset + pxv.INDEX_HEADER.size] ^= 1
    archive.write_bytes(data)
    with pytest.raises(ValueError, match="index checksum"):
        pxv.inspect(archive)


def test_pxv_rejects_frame_payload_corruption(tmp_path):
    archive = tmp_path / "bad-frame.pxv"
    _encode_sample(archive)
    data = bytearray(archive.read_bytes())
    header = pxv.HEADER.unpack_from(data)
    index_offset = header[-2]
    entry = pxv.INDEX_ENTRY.unpack_from(
        data, index_offset + pxv.INDEX_HEADER.size
    )
    _timestamp, frame_offset, payload_size, _crc = entry
    payload_offset = frame_offset + pxv.FRAME_HEADER.size
    assert payload_size > 0
    data[payload_offset + payload_size // 2] ^= 0x40
    archive.write_bytes(data)
    with pytest.raises(ValueError, match="frame archive checksum"):
        pxv.decode_frame(archive, 0)


def test_pxv_rejects_non_monotonic_timestamps_without_leaving_output(tmp_path):
    archive = tmp_path / "invalid.pxv"
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
    ]
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        pxv.encode_frames(
            [(frames[0], 10), (frames[1], 9)], archive,
            width=8, height=8, channels=3, tile_size=64, effort="fast",
        )
    assert not archive.exists()


def test_pxv_empty_stream_is_rejected(tmp_path):
    archive = tmp_path / "empty.pxv"
    with pytest.raises(ValueError, match="No frames"):
        pxv.encode_frames(
            [], archive, width=8, height=8, channels=3,
            tile_size=64, effort="fast",
        )
    assert not archive.exists()


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg/ffprobe are optional runtime dependencies",
)
def test_ffmpeg_video_frontend_roundtrip(tmp_path):
    source = tmp_path / "source.mkv"
    archive = tmp_path / "source.pxv"
    restored = tmp_path / "restored.mkv"
    # Generate two RGB frames via FFV1 to keep the source fixture deterministic.
    raw = bytes([255, 0, 0] * (16 * 16)) + bytes([0, 255, 0] * (16 * 16))
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", "16x16", "-framerate", "2",
        "-i", "pipe:0", "-an", "-c:v", "ffv1", "-pix_fmt", "bgr0",
        str(source),
    ], input=raw, check=True, capture_output=True)

    encoded = pxv.encode_video(source, archive, tile_size=64, effort="fast")
    assert encoded["frame_count"] == 2
    assert pxv.verify(archive, deep=True)["verified"] is True
    decoded = pxv.decode_video(archive, restored)
    assert decoded["frame_count"] == 2

    result = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(restored),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], check=True, capture_output=True)
    assert result.stdout == raw


def test_pxv_rejects_timestamp_outside_signed_64bit_range(tmp_path):
    archive = tmp_path / "bad-timestamp.pxv"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="signed 64-bit range"):
        pxv.encode_frames(
            [(frame, 1 << 63)], archive,
            width=8, height=8, channels=3, tile_size=64, effort="fast",
        )
    assert not archive.exists()
