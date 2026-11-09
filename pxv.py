#!/usr/bin/env python3
"""PXV v1: indexed, lossless video built from PXL v8 lossless frames.

PXV v1 stores each decoded 8-bit frame as an independent PXL archive. This
provides exact frame reconstruction and random access without temporal coding.
FFmpeg/ffprobe are optional runtime tools used only for video file I/O.
"""
from __future__ import annotations

import argparse
import binascii
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
from PIL import Image

import pxl


MAGIC = b"PXV\x00"
VERSION = 1
FRAME_MAGIC = b"FRM1"
INDEX_MAGIC = b"IDX1"

# magic, version, flags, width, height, channels, reserved, fps_num,
# fps_den, frame_count, index_offset, header_crc32
HEADER = struct.Struct("<4sHHIIHHIIQQI")
# frame magic, frame number, timestamp in microseconds, PXL length, PXL CRC32
FRAME_HEADER = struct.Struct("<4sQqQI")
# index magic, frame count, entry size, entries CRC32
INDEX_HEADER = struct.Struct("<4sQII")
# timestamp_us, frame record offset, PXL archive length, PXL archive CRC32
INDEX_ENTRY = struct.Struct("<qQQI")

MAX_FRAME_COUNT = 1_000_000
MAX_FRAME_BYTES = 512 * 1024 * 1024
MAX_DIMENSION = 100_000
MAX_PIXELS = 100_000_000
COPY_CHUNK = 1024 * 1024
DEFAULT_FPS_NUM = 25
DEFAULT_FPS_DEN = 1

Progress = Callable[[int, str], None]


def _report(progress: Progress | None, percent: int, message: str) -> None:
    if progress is not None:
        progress(max(0, min(100, int(percent))), message)


def _which_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"{name} was not found. Install FFmpeg and make sure {name} is on PATH."
        )
    return path


def _parse_rate(value: str | None) -> tuple[int, int]:
    if not value or value in ("N/A", "0/0", "0"):
        return DEFAULT_FPS_NUM, DEFAULT_FPS_DEN
    try:
        rate = Fraction(value).limit_denominator(1_000_000)
        if rate.numerator <= 0 or rate.denominator <= 0:
            raise ValueError
        if rate.numerator > 0xFFFFFFFF or rate.denominator > 0xFFFFFFFF:
            raise ValueError
        return rate.numerator, rate.denominator
    except (ValueError, ZeroDivisionError, TypeError):
        return DEFAULT_FPS_NUM, DEFAULT_FPS_DEN


def _probe_video(path: Path) -> dict:
    ffprobe = _which_tool("ffprobe")
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,pix_fmt,nb_frames",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8",
            errors="replace",
        )
        document = json.loads(completed.stdout)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"ffprobe could not read the input video: {detail}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid stream metadata") from exc

    streams = document.get("streams") or []
    if not streams:
        raise ValueError("The input contains no video stream")
    stream = streams[0]
    width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
    if width < 1 or height < 1 or width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ValueError("Video dimensions are outside PXV safety limits")
    if width * height > MAX_PIXELS:
        raise ValueError("Video frame exceeds the 100-megapixel PXV safety limit")
    rate_value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps_num, fps_den = _parse_rate(rate_value)
    pix_fmt = str(stream.get("pix_fmt") or "").lower()
    alpha = (
        pix_fmt.startswith(("rgba", "bgra", "argb", "abgr", "yuva", "gbrap", "ya"))
        or pix_fmt.endswith("a")
    )
    try:
        frame_count = int(stream.get("nb_frames"))
        if frame_count < 1:
            frame_count = None
    except (TypeError, ValueError):
        frame_count = None
    return {
        "width": width, "height": height, "fps_num": fps_num,
        "fps_den": fps_den, "pix_fmt": pix_fmt, "channels": 4 if alpha else 3,
        "frame_count": frame_count,
    }


def _read_exact(stream, size: int) -> bytes | None:
    parts = []
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, COPY_CHUNK))
        if not chunk:
            if not parts:
                return None
            raise ValueError("Video decoder ended in the middle of a frame")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _max_archive_size(width: int, height: int, channels: int) -> int:
    raw_size = width * height * channels
    # PXL's v8 tile stream has bounded per-row predictor and per-tile framing
    # overhead. Use the smallest legal tile to form a conservative upper bound.
    tile_count = math.ceil(width / 64) * math.ceil(height / 64)
    overhead = height * 2 + tile_count * 64 + 1024 * 1024
    return min(MAX_FRAME_BYTES, raw_size + overhead)


def _frame_to_bytes(frame, width: int, height: int, channels: int | None):
    if isinstance(frame, Image.Image):
        target_mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels)
        if target_mode is None:
            if frame.mode in ("L", "RGB", "RGBA"):
                target_mode = frame.mode
            elif frame.mode == "LA" or "transparency" in frame.info:
                target_mode = "RGBA"
            else:
                target_mode = "RGB"
        image = frame if frame.mode == target_mode else frame.convert(target_mode)
        if image.size != (width, height):
            raise ValueError("Every PXV frame must have identical dimensions")
        resolved_channels = len(image.getbands())
        return image.tobytes(), resolved_channels

    if isinstance(frame, np.ndarray):
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            raise ValueError("PXV v1 accepts uint8 frames only; high bit depth is planned for PXV 2.0")
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3 or array.shape[:2] != (height, width):
            raise ValueError("Frame array must have shape (height, width, channels)")
        resolved_channels = int(array.shape[2])
        if resolved_channels not in (1, 3, 4):
            raise ValueError("PXV frames must have 1, 3 or 4 channels")
        if channels is not None and resolved_channels != channels:
            raise ValueError("Every PXV frame must use the same channel count")
        return np.ascontiguousarray(array).tobytes(), resolved_channels

    if isinstance(frame, (bytes, bytearray, memoryview)):
        if channels not in (1, 3, 4):
            raise ValueError("Specify channels when supplying raw frame bytes")
        data = bytes(frame)
        expected = width * height * channels
        if len(data) != expected:
            raise ValueError("Raw frame byte length does not match its dimensions")
        return data, channels

    raise TypeError("Frames must be PIL images, uint8 NumPy arrays, or raw bytes")


def _copy_and_crc(source: Path, destination, size: int) -> int:
    crc = 0
    copied = 0
    with source.open("rb") as src:
        while copied < size:
            chunk = src.read(min(COPY_CHUNK, size - copied))
            if not chunk:
                raise ValueError("Truncated temporary PXL frame archive")
            destination.write(chunk)
            crc = zlib.crc32(chunk, crc)
            copied += len(chunk)
        if src.read(1):
            raise ValueError("Unexpected trailing data in temporary PXL frame archive")
    return crc & 0xFFFFFFFF


def encode_frames(
    frames: Iterable,
    output_path: str | os.PathLike,
    width: int,
    height: int,
    fps_num: int = DEFAULT_FPS_NUM,
    fps_den: int = DEFAULT_FPS_DEN,
    channels: int | None = None,
    tile_size: int = 256,
    effort: str = "balanced",
    progress: Progress | None = None,
    overwrite: bool = False,
    total_frames: int | None = None,
) -> dict:
    """Encode an iterable of frames. A frame may be (image, timestamp_us)."""
    dst = Path(output_path).expanduser()
    width, height = int(width), int(height)
    fps_num, fps_den = int(fps_num), int(fps_den)
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
        raise ValueError("Invalid PXV frame dimensions")
    if width * height > MAX_PIXELS:
        raise ValueError("Frame exceeds the 100-megapixel PXV safety limit")
    if fps_num <= 0 or fps_den <= 0 or fps_num > 0xFFFFFFFF or fps_den > 0xFFFFFFFF:
        raise ValueError("Invalid frame rate")
    if channels is not None and channels not in (1, 3, 4):
        raise ValueError("PXV frames must have 1, 3 or 4 channels")
    if tile_size < 64 or tile_size > 1024 or tile_size % 8:
        raise ValueError("PXL tile size must be a multiple of 8 between 64 and 1024")
    if effort not in ("fast", "balanced", "maximum"):
        raise ValueError("effort must be fast, balanced, or maximum")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=dst.name + ".", suffix=".pxv-tmp", dir=str(dst.parent)
    )
    os.close(fd)
    temporary_output = Path(temporary_name)

    frame_count = 0
    last_timestamp = None
    first_timestamp = None
    default_step = max(1, int(round(1_000_000 * fps_den / fps_num)))
    index_crc = 0
    index_offset = 0

    try:
        with tempfile.TemporaryDirectory(prefix="pxv-frame-") as temp_dir:
            temp_root = Path(temp_dir)
            png_path = temp_root / "frame.png"
            pxl_path = temp_root / "frame.pxl"
            index_path = temp_root / "index.bin"

            with index_path.open("wb") as index_stream, temporary_output.open("w+b") as output:
                output.write(bytes(HEADER.size))
                for item in frames:
                    if frame_count >= MAX_FRAME_COUNT:
                        raise ValueError("Video exceeds the PXV maximum frame count")
                    if isinstance(item, tuple) and len(item) == 2:
                        frame, timestamp_value = item
                        timestamp_us = int(timestamp_value)
                    else:
                        frame = item
                        timestamp_us = (
                            0 if last_timestamp is None else last_timestamp + default_step
                        )

                    frame_data, frame_channels = _frame_to_bytes(
                        frame, width, height, channels
                    )
                    if channels is None:
                        channels = frame_channels
                    if frame_channels != channels:
                        raise ValueError("PXV channel count changed between frames")
                    expected_bytes = width * height * channels
                    if len(frame_data) != expected_bytes:
                        raise ValueError("Frame byte count does not match PXV geometry")
                    if not (-(1 << 63) <= timestamp_us < (1 << 63)):
                        raise ValueError("Frame timestamp is outside the PXV signed 64-bit range")
                    if last_timestamp is not None and timestamp_us < last_timestamp:
                        raise ValueError("Frame timestamps must be monotonically non-decreasing")
                    last_timestamp = timestamp_us
                    if first_timestamp is None:
                        first_timestamp = timestamp_us

                    mode = {1: "L", 3: "RGB", 4: "RGBA"}[channels]
                    Image.frombytes(mode, (width, height), frame_data).save(
                        png_path, format="PNG"
                    )
                    pxl.compress(
                        png_path, pxl_path, mode="lossless", tile_size=tile_size,
                        effort=effort, backend="cpu", workers=1,
                        adaptive_tiles=True, checkpoint_interval=8,
                    )
                    payload_size = pxl_path.stat().st_size
                    if payload_size < 1 or payload_size > _max_archive_size(
                        width, height, channels
                    ):
                        raise ValueError("PXL frame archive exceeds PXV size limits")
                    frame_offset = output.tell()
                    frame_crc = _file_crc32(pxl_path)
                    output.write(FRAME_HEADER.pack(
                        FRAME_MAGIC, frame_count, timestamp_us, payload_size, frame_crc
                    ))
                    copied_crc = _copy_and_crc(pxl_path, output, payload_size)
                    if copied_crc != frame_crc:
                        raise ValueError("Temporary PXL frame changed while writing PXV")
                    entry = INDEX_ENTRY.pack(
                        timestamp_us, frame_offset, payload_size, frame_crc
                    )
                    index_stream.write(entry)
                    index_crc = zlib.crc32(entry, index_crc)
                    frame_count += 1
                    if total_frames and total_frames > 0:
                        percent = min(99, int(frame_count * 99 / total_frames))
                    else:
                        percent = min(99, int(99 * frame_count / (frame_count + 300)))
                    _report(progress, percent, f"Encoded frame {frame_count}")

                if frame_count < 1:
                    raise ValueError("No frames were supplied to the PXV encoder")
                index_stream.flush()
                index_offset = output.tell()
                output.write(INDEX_HEADER.pack(
                    INDEX_MAGIC, frame_count, INDEX_ENTRY.size, index_crc & 0xFFFFFFFF
                ))
                with index_path.open("rb") as index_stream:
                    shutil.copyfileobj(index_stream, output, length=COPY_CHUNK)

                header_without_crc = HEADER.pack(
                    MAGIC, VERSION, 0, width, height, channels, 0,
                    fps_num, fps_den, frame_count, index_offset, 0,
                )
                header_crc = zlib.crc32(header_without_crc[:-4]) & 0xFFFFFFFF
                final_header = HEADER.pack(
                    MAGIC, VERSION, 0, width, height, channels, 0,
                    fps_num, fps_den, frame_count, index_offset, header_crc,
                )
                output.seek(0)
                output.write(final_header)
                output.flush()
                os.fsync(output.fileno())

        if dst.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {dst}")
        os.replace(temporary_output, dst)
    except Exception:
        try:
            temporary_output.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    output_bytes = dst.stat().st_size
    _report(progress, 100, f"PXV complete · {frame_count} frames")
    duration = 0.0
    if frame_count > 1 and first_timestamp is not None and last_timestamp is not None:
        duration = max(0, last_timestamp - first_timestamp) / 1_000_000
    duration += fps_den / fps_num
    return {
        "output_bytes": output_bytes, "width": width, "height": height,
        "channels": channels, "frame_count": frame_count,
        "fps": f"{fps_num}/{fps_den}", "duration_seconds": duration,
        "codec": "PXV v1 · PXL v8 intra-frame lossless",
    }


def _file_crc32(path: Path) -> int:
    crc = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(COPY_CHUNK)
            if not chunk:
                break
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def _validate_open_file(stream) -> dict:
    stream.seek(0, os.SEEK_END)
    file_size = stream.tell()
    stream.seek(0)
    raw_header = stream.read(HEADER.size)
    if len(raw_header) != HEADER.size:
        raise ValueError("Truncated PXV header")
    (
        magic, version, flags, width, height, channels, reserved,
        fps_num, fps_den, frame_count, index_offset, header_crc,
    ) = HEADER.unpack(raw_header)

    if magic != MAGIC:
        raise ValueError("Invalid PXV magic signature")
    if version != VERSION:
        raise ValueError(f"Unsupported PXV version: {version}")
    if flags != 0 or reserved != 0:
        raise ValueError("Unsupported PXV flags or reserved fields")
    if zlib.crc32(raw_header[:-4]) & 0xFFFFFFFF != header_crc:
        raise ValueError("PXV header checksum failed")
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
        raise ValueError("PXV dimensions exceed safety limits")
    if width * height > MAX_PIXELS or channels not in (1, 3, 4):
        raise ValueError("Invalid PXV frame geometry or channel count")
    if fps_num < 1 or fps_den < 1:
        raise ValueError("Invalid PXV frame rate")
    if frame_count < 1 or frame_count > MAX_FRAME_COUNT:
        raise ValueError("PXV frame count exceeds safety limits")

    index_size = INDEX_HEADER.size + frame_count * INDEX_ENTRY.size
    if index_offset < HEADER.size or index_offset + index_size != file_size:
        raise ValueError("Invalid PXV index offset or length")
    stream.seek(index_offset)
    raw_index_header = stream.read(INDEX_HEADER.size)
    if len(raw_index_header) != INDEX_HEADER.size:
        raise ValueError("Truncated PXV index header")
    index_magic, index_count, entry_size, expected_index_crc = INDEX_HEADER.unpack(
        raw_index_header
    )
    if (
        index_magic != INDEX_MAGIC or index_count != frame_count
        or entry_size != INDEX_ENTRY.size
    ):
        raise ValueError("Invalid PXV index header")

    stream.seek(index_offset + INDEX_HEADER.size)
    expected_offset = HEADER.size
    previous_timestamp = None
    index_crc = 0
    max_payload = _max_archive_size(width, height, channels)
    for frame_number in range(frame_count):
        raw_entry = stream.read(INDEX_ENTRY.size)
        if len(raw_entry) != INDEX_ENTRY.size:
            raise ValueError("Truncated PXV frame index")
        index_crc = zlib.crc32(raw_entry, index_crc)
        timestamp_us, frame_offset, payload_size, _payload_crc = INDEX_ENTRY.unpack(
            raw_entry
        )
        if frame_offset != expected_offset:
            raise ValueError("PXV frame index contains a gap, overlap, or invalid offset")
        if payload_size < 1 or payload_size > max_payload:
            raise ValueError("PXV frame archive length exceeds safety limits")
        if previous_timestamp is not None and timestamp_us < previous_timestamp:
            raise ValueError("PXV frame timestamps are not monotonic")
        previous_timestamp = timestamp_us
        expected_offset = frame_offset + FRAME_HEADER.size + payload_size
        if expected_offset > index_offset:
            raise ValueError("PXV frame payload overlaps the index")

    if expected_offset != index_offset:
        raise ValueError("PXV frame data does not end at the index")
    if index_crc & 0xFFFFFFFF != expected_index_crc:
        raise ValueError("PXV index checksum failed")

    return {
        "path": None, "version": version, "width": width, "height": height,
        "channels": channels, "fps_num": fps_num, "fps_den": fps_den,
        "frame_count": frame_count, "index_offset": index_offset,
        "index_size": index_size, "file_bytes": file_size,
        "max_payload": max_payload,
    }


def _read_index_entry(stream, info: dict, frame_number: int) -> tuple[int, int, int, int]:
    if not 0 <= frame_number < info["frame_count"]:
        raise IndexError("PXV frame index is out of range")
    position = (
        info["index_offset"] + INDEX_HEADER.size
        + frame_number * INDEX_ENTRY.size
    )
    stream.seek(position)
    raw = stream.read(INDEX_ENTRY.size)
    if len(raw) != INDEX_ENTRY.size:
        raise ValueError("Truncated PXV frame index entry")
    return INDEX_ENTRY.unpack(raw)


def _pxl_dimensions(metadata: dict) -> tuple[int, int]:
    dimensions = str(metadata.get("dimensions", ""))
    try:
        width_text, height_text = dimensions.lower().replace(" ", "").split("x", 1)
        return int(width_text), int(height_text)
    except (ValueError, TypeError):
        raise ValueError("Embedded PXL frame has invalid dimensions") from None


def _decode_frame_at(stream, info: dict, frame_number: int, temp_root: Path) -> Image.Image:
    timestamp_us, frame_offset, payload_size, payload_crc = _read_index_entry(
        stream, info, frame_number
    )
    stream.seek(frame_offset)
    raw_frame_header = stream.read(FRAME_HEADER.size)
    if len(raw_frame_header) != FRAME_HEADER.size:
        raise ValueError("Truncated PXV frame header")
    marker, stored_number, stored_timestamp, stored_size, stored_crc = FRAME_HEADER.unpack(
        raw_frame_header
    )
    if (
        marker != FRAME_MAGIC or stored_number != frame_number
        or stored_timestamp != timestamp_us or stored_size != payload_size
        or stored_crc != payload_crc
    ):
        raise ValueError("PXV frame record does not match its index entry")

    archive_path = temp_root / "frame.pxl"
    image_path = temp_root / "frame.png"
    crc = 0
    remaining = payload_size
    with archive_path.open("wb") as archive:
        while remaining:
            chunk = stream.read(min(COPY_CHUNK, remaining))
            if not chunk:
                raise ValueError("Truncated embedded PXL frame archive")
            archive.write(chunk)
            crc = zlib.crc32(chunk, crc)
            remaining -= len(chunk)
    if crc & 0xFFFFFFFF != payload_crc:
        raise ValueError("PXV frame archive checksum failed")

    metadata = pxl.inspect(archive_path)
    if (
        metadata.get("profile") != "lossless"
        or metadata.get("channels") != info["channels"]
        or _pxl_dimensions(metadata) != (info["width"], info["height"])
    ):
        raise ValueError("Embedded PXL frame geometry or profile does not match PXV")
    pxl.decompress(archive_path, image_path, backend="cpu", workers=1)
    with Image.open(image_path) as image:
        expected_mode = {1: "L", 3: "RGB", 4: "RGBA"}[info["channels"]]
        if image.size != (info["width"], info["height"]) or image.mode != expected_mode:
            raise ValueError("Decoded PXL frame does not match PXV frame geometry")
        return image.copy()


def inspect(path: str | os.PathLike) -> dict:
    file_path = Path(path)
    with file_path.open("rb") as stream:
        info = _validate_open_file(stream)
    fps = Fraction(info["fps_num"], info["fps_den"])
    return {
        "format": "PXV", "version": info["version"],
        "dimensions": f"{info['width']} x {info['height']}",
        "width": info["width"], "height": info["height"],
        "channels": info["channels"],
        "pixel_format": {1: "gray8", 3: "rgb24", 4: "rgba"}[info["channels"]],
        "fps": f"{fps.numerator}/{fps.denominator}",
        "frame_count": info["frame_count"],
        "duration_seconds": round(info["frame_count"] / float(fps), 6),
        "frame_coding": "independent PXL v8 lossless frames",
        "temporal_compression": False,
        "audio": False, "file_bytes": info["file_bytes"],
        "index_bytes": info["index_size"],
    }


def decode_frame(path: str | os.PathLike, frame_number: int) -> Image.Image:
    """Decode one frame by its index without decoding preceding frames."""
    with Path(path).open("rb") as stream:
        info = _validate_open_file(stream)
        with tempfile.TemporaryDirectory(prefix="pxv-decode-") as temp_dir:
            return _decode_frame_at(stream, info, int(frame_number), Path(temp_dir))


def iter_frames(path: str | os.PathLike) -> Iterator[Image.Image]:
    """Yield decoded frames in display order, keeping only one PXL scratch file."""
    with Path(path).open("rb") as stream:
        info = _validate_open_file(stream)
        with tempfile.TemporaryDirectory(prefix="pxv-decode-") as temp_dir:
            temp_root = Path(temp_dir)
            for frame_number in range(info["frame_count"]):
                yield _decode_frame_at(stream, info, frame_number, temp_root)


def verify(path: str | os.PathLike, deep: bool = False,
           progress: Progress | None = None) -> dict:
    """Verify index and every frame payload; deep=True also decodes every frame."""
    with Path(path).open("rb") as stream:
        info = _validate_open_file(stream)
        with tempfile.TemporaryDirectory(prefix="pxv-verify-") as temp_dir:
            temp_root = Path(temp_dir)
            for frame_number in range(info["frame_count"]):
                timestamp_us, frame_offset, payload_size, payload_crc = _read_index_entry(
                    stream, info, frame_number
                )
                stream.seek(frame_offset)
                raw_frame_header = stream.read(FRAME_HEADER.size)
                if len(raw_frame_header) != FRAME_HEADER.size:
                    raise ValueError("Truncated PXV frame header")
                marker, stored_number, stored_timestamp, stored_size, stored_crc = FRAME_HEADER.unpack(
                    raw_frame_header
                )
                if (
                    marker != FRAME_MAGIC or stored_number != frame_number
                    or stored_timestamp != timestamp_us or stored_size != payload_size
                    or stored_crc != payload_crc
                ):
                    raise ValueError("PXV frame record does not match its index entry")
                crc = 0
                remaining = payload_size
                while remaining:
                    chunk = stream.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise ValueError("Truncated PXV frame payload")
                    crc = zlib.crc32(chunk, crc)
                    remaining -= len(chunk)
                if crc & 0xFFFFFFFF != payload_crc:
                    raise ValueError("PXV frame archive checksum failed")
                if deep:
                    _decode_frame_at(stream, info, frame_number, temp_root)
                _report(
                    progress,
                    int((frame_number + 1) * 100 / info["frame_count"]),
                    f"Verified frame {frame_number + 1}/{info['frame_count']}",
                )
    return {
        "verified": True, "deep": bool(deep),
        "frame_count": info["frame_count"],
        "file_bytes": info["file_bytes"],
    }


def encode_video(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    tile_size: int = 256,
    effort: str = "balanced",
    progress: Progress | None = None,
    overwrite: bool = False,
) -> dict:
    """Decode a video with FFmpeg and encode each RGB/RGBA frame through PXL."""
    src, dst = Path(input_path).expanduser(), Path(output_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(src)
    if src.resolve() == dst.resolve():
        raise ValueError("Input and output paths must be different")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {dst}")

    ffmpeg = _which_tool("ffmpeg")
    metadata = _probe_video(src)
    width, height = metadata["width"], metadata["height"]
    channels = metadata["channels"]
    pixel_format = "rgba" if channels == 4 else "rgb24"
    frame_bytes = width * height * channels
    nominal_step = max(1, int(round(
        1_000_000 * metadata["fps_den"] / metadata["fps_num"]
    )))

    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, staged_name = tempfile.mkstemp(
        prefix=dst.name + ".", suffix=".pxv-stage", dir=str(dst.parent)
    )
    os.close(fd)
    staged_output = Path(staged_name)
    staged_output.unlink(missing_ok=True)

    ffprobe = _which_tool("ffprobe")
    timestamp_command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0", str(src),
    ]
    timestamp_process = None
    decode_process = None
    decode_command = [
        ffmpeg, "-nostdin", "-v", "error", "-noautorotate", "-i", str(src),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0",
        "-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1",
    ]
    def frames():
        index = 0
        previous_timestamp = None
        while True:
            raw = _read_exact(decode_process.stdout, frame_bytes)
            if raw is None:
                break
            line = timestamp_process.stdout.readline() if timestamp_process.stdout else ""
            timestamp = None
            if line:
                candidate = line.strip().split(",", 1)[0]
                if candidate and candidate not in ("N/A", "nan"):
                    try:
                        timestamp = int(round(float(candidate) * 1_000_000))
                    except ValueError:
                        timestamp = None
            if timestamp is None:
                timestamp = (
                    0 if previous_timestamp is None
                    else previous_timestamp + nominal_step
                )
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError("Input video has non-monotonic presentation timestamps")
            previous_timestamp = timestamp
            yield raw, timestamp
            index += 1

    try:
        timestamp_process = subprocess.Popen(
            timestamp_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="ascii", errors="replace",
        )
        decode_process = subprocess.Popen(
            decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        result = encode_frames(
            frames(), staged_output, width, height,
            fps_num=metadata["fps_num"], fps_den=metadata["fps_den"],
            channels=channels, tile_size=tile_size, effort=effort,
            progress=progress, overwrite=False,
            total_frames=metadata["frame_count"],
        )
        decode_error = decode_process.stderr.read().decode("utf-8", "replace").strip()
        decode_return = decode_process.wait()
        probe_error = timestamp_process.stderr.read().strip()
        probe_return = timestamp_process.wait()
        if decode_return != 0:
            raise RuntimeError(f"FFmpeg video decoding failed: {decode_error}")
        if probe_return != 0:
            raise RuntimeError(f"ffprobe timestamp extraction failed: {probe_error}")
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {dst}")
        os.replace(staged_output, dst)
    except Exception:
        staged_output.unlink(missing_ok=True)
        for process in (decode_process, timestamp_process):
            if process is None:
                continue
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
        raise
    result["output_bytes"] = dst.stat().st_size
    result["input_file"] = str(src)
    result["source_pixel_format"] = metadata["pix_fmt"] or "unknown"
    return result


def decode_video(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    progress: Progress | None = None,
    overwrite: bool = False,
) -> dict:
    """Decode PXV to a Matroska file with the lossless FFV1 video codec."""
    src, dst = Path(input_path).expanduser(), Path(output_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(src)
    if src.resolve() == dst.resolve():
        raise ValueError("Input and output paths must be different")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {dst}")
    ffmpeg = _which_tool("ffmpeg")
    with src.open("rb") as stream:
        info = _validate_open_file(stream)

    input_format = {1: "gray", 3: "rgb24", 4: "rgba"}[info["channels"]]
    output_format = {1: "gray", 3: "bgr0", 4: "bgra"}[info["channels"]]
    fps = f"{info['fps_num']}/{info['fps_den']}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=dst.stem + ".", suffix=(dst.suffix or ".mkv"),
        dir=str(dst.parent),
    )
    os.close(fd)
    temporary_output = Path(temporary_name)

    command = [
        ffmpeg, "-y", "-nostdin", "-v", "error",
        "-f", "rawvideo", "-pixel_format", input_format,
        "-video_size", f"{info['width']}x{info['height']}",
        "-framerate", fps, "-i", "pipe:0",
        "-an", "-c:v", "ffv1", "-level", "3", "-g", "1",
        "-slicecrc", "1", "-pix_fmt", output_format, "-f", "matroska",
        str(temporary_output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    completed = 0
    try:
        assert process.stdin is not None
        for frame in iter_frames(src):
            process.stdin.write(frame.tobytes())
            completed += 1
            _report(
                progress,
                int(completed * 100 / info["frame_count"]),
                f"Decoded frame {completed}/{info['frame_count']}",
            )
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", "replace").strip() if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg FFV1 muxing failed: {stderr}")
        if completed != info["frame_count"]:
            raise ValueError("Decoded frame count does not match PXV index")
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {dst}")
        os.replace(temporary_output, dst)
    except Exception:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()
        temporary_output.unlink(missing_ok=True)
        raise

    _report(progress, 100, f"FFV1 export complete · {completed} frames")
    return {
        "output_bytes": dst.stat().st_size, "frame_count": completed,
        "width": info["width"], "height": info["height"],
        "codec": "FFV1 lossless · Matroska",
        "note": "PXV timestamps are retained in the source index; this export uses the stored nominal frame rate.",
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="pxv", description="PXV lossless video codec using PXL v8 frame compression"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode", help="Encode a video into .pxv")
    encode.add_argument("input")
    encode.add_argument("output")
    encode.add_argument("--tile", type=int, default=256)
    encode.add_argument("--effort", choices=("fast", "balanced", "maximum"), default="balanced")
    encode.add_argument("--overwrite", action="store_true")
    decode = commands.add_parser("decode", help="Decode .pxv to lossless FFV1 Matroska")
    decode.add_argument("input")
    decode.add_argument("output")
    decode.add_argument("--overwrite", action="store_true")
    info = commands.add_parser("info", help="Inspect PXV metadata and index")
    info.add_argument("input")
    check = commands.add_parser("verify", help="Verify frame checksums")
    check.add_argument("input")
    check.add_argument("--deep", action="store_true", help="Decode every frame as well")

    args = parser.parse_args()
    try:
        if args.command == "encode":
            result = encode_video(
                args.input, args.output, tile_size=args.tile,
                effort=args.effort, overwrite=args.overwrite,
            )
        elif args.command == "decode":
            result = decode_video(args.input, args.output, overwrite=args.overwrite)
        elif args.command == "info":
            result = inspect(args.input)
        else:
            result = verify(args.input, deep=args.deep)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"PXV error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def cli_entry() -> None:
    raise SystemExit(_cli())


if __name__ == "__main__":
    raise SystemExit(_cli())
