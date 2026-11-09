"""PXL v7 high-precision outer container with v8 byte-plane compression.

The compatible v7 outer container is lossless and stores uint16, float16 or
float32 sample planes. New writers encode each byte plane as a PXL v8 8-bit
image; the reader also accepts older embedded v7 streams. Floating-point
payloads are preserved bit-for-bit, including signed zero and NaN payloads.
PXL does not apply a tone map or change colour space.
"""
from __future__ import annotations

import binascii
import math
import struct
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

import pxl as core

VERSION = 7
SAMPLE_UINT16 = 1
SAMPLE_FLOAT16 = 2
SAMPLE_FLOAT32 = 3
PLANE_ENTRY = struct.Struct("<QI")  # encoded PXL stream length, raw plane CRC32
# Embedded byte-plane archives are at most their decoded byte count plus
# container/tile-record overhead. Bound this allowance across all planes.
MAX_EMBEDDED_ARCHIVE_OVERHEAD = 16 * 1024 * 1024


def _read_tiff(path: Path):
    try:
        import tifffile
    except Exception as exc:
        raise RuntimeError("High-precision TIFF support requires tifffile; install it with `python -m pip install tifffile`") from exc
    try:
        array = tifffile.imread(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot read TIFF image: {exc}") from exc
    if array.ndim == 3:
        # tifffile's normal RGB layout is YXS. Accommodate common planar CYX.
        if array.shape[-1] in (3, 4):
            pass
        elif array.shape[0] in (3, 4):
            array = np.moveaxis(array, 0, -1)
        else:
            raise ValueError("PXL v7 expects a single 2D grayscale or H×W×(RGB/RGBA) image")
    elif array.ndim != 2:
        raise ValueError("PXL v7 does not yet support image stacks or multidimensional TIFF series")
    return np.asarray(array)


def _read_exr(path: Path):
    try:
        import OpenEXR
    except Exception as exc:
        raise RuntimeError("OpenEXR input requires the optional OpenEXR package; install it with `python -m pip install -r requirements-hdr.txt`") from exc
    try:
        with OpenEXR.File(str(path)) as exr:
            parts = getattr(exr, "parts", None)
            if parts is not None and len(parts) != 1:
                raise ValueError("PXL v7 supports single-part OpenEXR images only")
            channels = exr.channels()
            recognized = [name for name in ("RGBA", "RGB", "Y") if name in channels]
            if len(recognized) == 1:
                name = recognized[0]
                unexpected = set(channels) - {name}
                if unexpected:
                    raise ValueError("PXL v7 OpenEXR input cannot discard extra channels: " + ", ".join(sorted(unexpected)))
                return np.asarray(channels[name].pixels).copy()
            if all(name in channels for name in ("R", "G", "B")):
                extras = set(channels) - {"R", "G", "B", "A"}
                if extras:
                    raise ValueError("PXL v7 OpenEXR input cannot discard extra channels: " + ", ".join(sorted(extras)))
                names = ["R", "G", "B"] + (["A"] if "A" in channels else [])
                return np.stack([np.asarray(channels[name].pixels) for name in names], axis=-1)
            raise ValueError("PXL v7 OpenEXR needs Y, RGB or RGBA channels")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot read OpenEXR file: {exc}") from exc


def _write_exr(path: Path, array: np.ndarray, channels: int):
    try:
        import OpenEXR
    except Exception as exc:
        raise RuntimeError("Writing OpenEXR requires the optional `OpenEXR` package; install it with `python -m pip install OpenEXR`") from exc
    if channels == 1:
        channel_data = {"Y": np.ascontiguousarray(array)}
    else:
        channel_data = {"RGB" if channels == 3 else "RGBA": np.ascontiguousarray(array)}
    header = {"type": OpenEXR.scanlineimage, "compression": OpenEXR.ZIP_COMPRESSION}
    with OpenEXR.File(header, channel_data) as output:
        output.write(str(path))


def _read_16bit_png(path: Path):
    """Read 16-bit PNG with PyPNG when available, retaining RGB/RGBA precision."""
    try:
        import png as pypng
    except Exception:
        return None
    reader = pypng.Reader(filename=str(path))
    width, height, rows, info = reader.read()
    if int(info.get("bitdepth", 0)) != 16:
        return None
    planes = int(info.get("planes", 1))
    if planes not in (1, 3, 4):
        raise ValueError("PXL v7 supports 16-bit grayscale, RGB and RGBA PNG; grayscale-alpha PNG is not yet supported")
    flat = np.fromiter((sample for row in rows for sample in row), dtype=np.uint16,
                       count=width * height * planes)
    array = flat.reshape(height, width, planes)
    return array[..., 0] if planes == 1 else array


def _png_ihdr_bit_depth(path: Path):
    try:
        with path.open("rb") as f:
            header = f.read(29)
        if len(header) >= 29 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            return header[24]
    except OSError:
        pass
    return None


def probe_high_precision(path: str | Path):
    """Inspect high-precision image metadata without decoding full TIFF/PNG arrays."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".png":
        try:
            with p.open("rb") as stream:
                head = stream.read(29)
        except OSError:
            return None
        if len(head) < 29 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
            return None
        bit_depth, color_type = head[24], head[25]
        if bit_depth != 16:
            return None
        channel_count = {0: 1, 2: 3, 6: 4}.get(color_type)
        if channel_count is None:
            raise ValueError("PXL v7 supports 16-bit grayscale, RGB and RGBA PNG only")
        width, height = struct.unpack(">II", head[16:24])
        return {"width": width, "height": height, "channels": channel_count,
                "sample_format": "uint16", "dtype": "uint16"}
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
        except Exception as exc:
            raise RuntimeError("High-precision TIFF support requires tifffile; install it with `python -m pip install tifffile`") from exc
        with tifffile.TiffFile(str(p)) as tif:
            if not tif.series:
                return None
            series = tif.series[0]
            shape = tuple(series.shape)
            dtype = np.dtype(series.dtype)
        if dtype == np.dtype("uint8"):
            return None
        if dtype.kind == "u" and dtype.itemsize == 2:
            sample_format = "uint16"
        elif dtype.kind == "f" and dtype.itemsize == 2:
            sample_format = "float16"
        elif dtype.kind == "f" and dtype.itemsize == 4:
            sample_format = "float32"
        elif dtype.kind == "i" and dtype.itemsize <= 4:
            sample_format = "uint16"
        else:
            raise ValueError(f"Unsupported high-precision TIFF sample type: {dtype}")
        if len(shape) == 2:
            height, width = shape
            channels = 1
        elif len(shape) == 3 and shape[-1] in (3, 4):
            height, width, channels = shape
        elif len(shape) == 3 and shape[0] in (3, 4):
            channels, height, width = shape
        else:
            raise ValueError("PXL v7 expects a single 2D grayscale or H×W×(RGB/RGBA) TIFF image")
        return {"width": int(width), "height": int(height), "channels": int(channels),
                "sample_format": sample_format, "dtype": str(dtype)}
    if suffix == ".exr":
        array = _read_exr(p)
        dtype = array.dtype
        if dtype.kind == "f" and dtype.itemsize in (2, 4):
            sample_format = "float16" if dtype.itemsize == 2 else "float32"
        else:
            raise ValueError(f"Unsupported OpenEXR sample type: {dtype}; use HALF or FLOAT channels")
        channels = 1 if array.ndim == 2 else int(array.shape[2])
        return {"width": int(array.shape[1]), "height": int(array.shape[0]),
                "channels": channels, "sample_format": sample_format, "dtype": str(dtype)}
    return None


def load_high_precision(path: str | Path):
    """Return (array, sample_code, dtype, channels) or None for ordinary 8-bit input."""
    p = Path(path)
    array = None
    if p.suffix.lower() == ".exr":
        array = _read_exr(p)
    elif p.suffix.lower() == ".png" and _png_ihdr_bit_depth(p) == 16:
        array = _read_16bit_png(p)
    elif p.suffix.lower() in (".tif", ".tiff"):
        array = _read_tiff(p)
    if array is None:
        try:
            with Image.open(p) as image:
                if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                    array = np.asarray(image).copy()
                elif p.suffix.lower() == ".png" and _png_ihdr_bit_depth(p) == 16:
                    raise ValueError("This 16-bit PNG needs PyPNG to read RGB/RGBA precision; install pypng or convert it to a 16-bit TIFF")
        except ValueError:
            raise
        except Exception:
            return None
    if array is None:
        return None
    if array.ndim == 2:
        channels = 1
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        channels = int(array.shape[2])
    else:
        raise ValueError("PXL v7 supports grayscale, RGB and RGBA high-precision images")
    if array.dtype.kind == "u" and array.dtype.itemsize == 2:
        code, target_dtype = SAMPLE_UINT16, np.dtype("<u2")
    elif array.dtype.kind == "f" and array.dtype.itemsize == 2:
        code, target_dtype = SAMPLE_FLOAT16, np.dtype("<f2")
    elif array.dtype.kind == "f" and array.dtype.itemsize == 4:
        code, target_dtype = SAMPLE_FLOAT32, np.dtype("<f4")
    elif array.dtype.kind == "i" and array.dtype.itemsize <= 4:
        if array.size and (int(array.min()) < 0 or int(array.max()) > 65535):
            raise ValueError("PXL v7 integer input must be unsigned 16-bit samples")
        code, target_dtype = SAMPLE_UINT16, np.dtype("<u2")
    elif array.dtype == np.uint8:
        return None
    else:
        raise ValueError(f"Unsupported high-precision sample type: {array.dtype}; use uint16, float16 or float32")
    array = np.asarray(array, dtype=target_dtype)
    if array.shape[0] > core.MAX_DIMENSION or array.shape[1] > core.MAX_DIMENSION or array.shape[0] * array.shape[1] > core.MAX_PIXELS:
        raise ValueError("Image exceeds PXL's 100-megapixel safety limit")
    return array, code, target_dtype, channels


def _plane_image(plane: np.ndarray, channels: int) -> Image.Image:
    if channels == 1:
        return Image.fromarray(np.ascontiguousarray(plane.reshape(plane.shape[0], plane.shape[1])), "L")
    return Image.fromarray(np.ascontiguousarray(plane), "RGB" if channels == 3 else "RGBA")


def _forward_uint16_color(array: np.ndarray, channels: int, transform: int) -> np.ndarray:
    if transform == 0 or channels == 1:
        return array
    src = np.asarray(array, dtype=np.uint32)
    out = src.copy()
    r, g, b = src[..., 0], src[..., 1], src[..., 2]
    if transform == 1:  # Green base
        out[..., 0] = g
        out[..., 1] = (r - g) & 65535
        out[..., 2] = (b - g) & 65535
    elif transform == 2:  # Red base
        out[..., 0] = r
        out[..., 1] = (g - r) & 65535
        out[..., 2] = (b - r) & 65535
    elif transform == 3:  # Blue base
        out[..., 0] = b
        out[..., 1] = (r - b) & 65535
        out[..., 2] = (g - b) & 65535
    elif transform == 4:  # Reversible YCoCg-R lifting
        co = (r - b) & 65535
        t = (b + (co >> 1)) & 65535
        cg = (g - t) & 65535
        yy = (t + (cg >> 1)) & 65535
        out[..., 0], out[..., 1], out[..., 2] = yy, cg, co
    else:
        raise ValueError("Unknown PXL v7 16-bit colour transform")
    return out.astype(np.uint16)


def _inverse_uint16_color(array: np.ndarray, channels: int, transform: int) -> np.ndarray:
    if transform == 0 or channels == 1:
        return array
    src = np.asarray(array, dtype=np.uint32)
    out = src.copy()
    x, y, z = src[..., 0], src[..., 1], src[..., 2]
    if transform == 1:
        out[..., 0] = (y + x) & 65535
        out[..., 1] = x
        out[..., 2] = (z + x) & 65535
    elif transform == 2:
        out[..., 0] = x
        out[..., 1] = (y + x) & 65535
        out[..., 2] = (z + x) & 65535
    elif transform == 3:
        out[..., 0] = (y + x) & 65535
        out[..., 1] = (z + x) & 65535
        out[..., 2] = x
    elif transform == 4:
        yy, cg, co = x, y, z
        t = (yy - (cg >> 1)) & 65535
        g = (cg + t) & 65535
        b = (t - (co >> 1)) & 65535
        r = (b + co) & 65535
        out[..., 0], out[..., 1], out[..., 2] = r, g, b
    else:
        raise ValueError("Unknown PXL v7 16-bit colour transform")
    if channels == 4:
        out[..., 3] = src[..., 3]
    return out.astype(np.uint16)


def compress_high_precision(input_path, output_path, loaded, tile_size=256, progress=None,
                            effort="balanced", backend="auto", workers=None):
    src, dst = Path(input_path), Path(output_path)
    array, sample_code, dtype, channels = loaded
    height, width = array.shape[:2]
    bytes_per_sample = dtype.itemsize
    # Search reversible 16-bit colour transforms by complete resulting size.
    # Float samples are intentionally left bitwise untouched.
    if sample_code == SAMPLE_UINT16 and channels >= 3:
        transforms = [0, 4] if effort in ("balanced", "maximum") else [0]
        if effort == "maximum":
            transforms = [0, 1, 2, 3, 4]
    else:
        transforms = [0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    from pxl4_engine import compress as compress_v6
    best = None
    plane_reports = []
    plane_backend = backend
    with tempfile.TemporaryDirectory(prefix="pxl7_planes_") as td:
        td = Path(td)
        for transform in transforms:
            working = _forward_uint16_color(array, channels, transform) if sample_code == SAMPLE_UINT16 else array
            sample_shape = (height, width, channels) if channels > 1 else (height, width, 1)
            samples = np.ascontiguousarray(working.reshape(sample_shape), dtype=dtype)
            interleaved = samples.view(np.uint8).reshape(height, width, channels, bytes_per_sample)
            streams, raw_crcs = [], []
            for plane_index in range(bytes_per_sample):
                plane = interleaved[..., plane_index]
                image_path = td / f"t{transform}_plane_{plane_index}.png"
                archive_path = td / f"t{transform}_plane_{plane_index}.pxl"
                _plane_image(plane, channels).save(image_path, format="PNG")
                def plane_progress(percent, message, index=plane_index, tr=transform):
                    if progress:
                        progress(int(((transforms.index(tr) + (index + percent / 100.0) / bytes_per_sample) * 100) / len(transforms)),
                                 f"PXL v7 transform {tr}, byte plane {index + 1}/{bytes_per_sample}: {message}")
                plane_report = compress_v6(image_path, archive_path, mode="lossless", tile_size=tile_size,
                                           progress=plane_progress, effort=effort, backend=plane_backend, workers=workers)
                plane_reports.append(plane_report)
                if plane_backend == "auto" and plane_report.get("cuda_fallback"):
                    # Do not re-trigger the same failed CUDA initialization/kernel
                    # for each subsequent high-precision byte plane.
                    plane_backend = "cpu"
                streams.append(archive_path.read_bytes())
                raw_crcs.append(binascii.crc32(np.ascontiguousarray(plane).tobytes()) & 0xFFFFFFFF)
            candidate_size = core.HEADER.size + bytes_per_sample * PLANE_ENTRY.size + sum(map(len, streams))
            if best is None or candidate_size < best[0]:
                best = (candidate_size, transform, streams, raw_crcs)
        _, selected_transform, streams, raw_crcs = best
        encoded_flags = sample_code | (selected_transform << 2)
        with dst.open("wb") as out:
            out.write(core.HEADER.pack(core.MAGIC, VERSION, core.PROFILE_LOSSLESS,
                                       channels, encoded_flags, width, height,
                                       tile_size, bytes_per_sample, bytes_per_sample))
            for payload, crc in zip(streams, raw_crcs):
                out.write(PLANE_ENTRY.pack(len(payload), crc))
            for payload in streams:
                out.write(payload)
    cuda_tiles_used = sum(int(r.get("cuda_tiles_used", 0)) for r in plane_reports)
    cuda_fallbacks = [r.get("cuda_fallback") for r in plane_reports if r.get("cuda_fallback")]
    if cuda_tiles_used and cuda_fallbacks:
        backend_used = "cuda+cpu-fallback"
    elif cuda_tiles_used:
        backend_used = "cuda"
    else:
        backend_used = "cpu"
    return {"input_bytes": src.stat().st_size, "output_bytes": dst.stat().st_size,
            "width": width, "height": height, "channels": channels,
            "backend": backend_used, "cuda_tiles_used": cuda_tiles_used,
            "cuda_fallback": "; ".join(cuda_fallbacks) if cuda_fallbacks else None,
            "mode": "lossless", "sample_format": {1: "uint16", 2: "float16", 3: "float32"}[sample_code],
            "color_transform": {0: "none", 1: "green", 2: "red", 3: "blue", 4: "YCoCg-R"}[selected_transform],
            "byte_planes": bytes_per_sample, "tiles_per_plane": math.ceil(width / tile_size) * math.ceil(height / tile_size),
            "version": VERSION, "ratio": dst.stat().st_size / max(1, src.stat().st_size)}


def inspect(path):
    p = Path(path)
    with p.open("rb") as f:
        header = f.read(core.HEADER.size)
        if len(header) != core.HEADER.size:
            raise ValueError("Truncated PXL v7 header")
        magic, version, profile, channels, flags, width, height, tile_size, bytes_per_sample, plane_count = core.HEADER.unpack(header)
        sample_code, transform = flags & 0x03, flags >> 2
        if magic != core.MAGIC or version != VERSION or profile != core.PROFILE_LOSSLESS:
            raise ValueError("Not a supported PXL v7 high-precision file")
        if transform not in (0, 1, 2, 3, 4) or (transform and sample_code != SAMPLE_UINT16) or (transform and channels < 3):
            raise ValueError("Invalid PXL v7 sample transform")
        expected_bytes = {SAMPLE_UINT16: 2, SAMPLE_FLOAT16: 2, SAMPLE_FLOAT32: 4}.get(sample_code)
        if expected_bytes is None or bytes_per_sample != expected_bytes or plane_count != expected_bytes:
            raise ValueError("Invalid PXL v7 sample format")
        if channels not in (1, 3, 4) or width < 1 or height < 1 or width > core.MAX_DIMENSION or height > core.MAX_DIMENSION or width * height > core.MAX_PIXELS:
            raise ValueError("Invalid PXL v7 image geometry")
        if not (core.MIN_TILE <= tile_size <= core.MAX_TILE and tile_size % 8 == 0):
            raise ValueError("Invalid PXL v7 tile size")
        entries = []
        for _ in range(plane_count):
            raw = f.read(PLANE_ENTRY.size)
            if len(raw) != PLANE_ENTRY.size:
                raise ValueError("Truncated PXL v7 plane table")
            entries.append(PLANE_ENTRY.unpack(raw))
        payload_bytes = sum(length for length, _ in entries)
        if payload_bytes != p.stat().st_size - core.HEADER.size - plane_count * PLANE_ENTRY.size:
            raise ValueError("PXL v7 plane sizes do not match file length")
        decoded_sample_bytes = width * height * channels * bytes_per_sample
        max_embedded_bytes = decoded_sample_bytes + MAX_EMBEDDED_ARCHIVE_OVERHEAD
        if any(length < 1 for length, _ in entries) or payload_bytes > max_embedded_bytes:
            raise ValueError("PXL v7 aggregate embedded planes exceed the decoded-image safety limit")
    return {"version": 7, "profile": "lossless", "width": width, "height": height,
            "channels": channels, "sample_format": {1: "uint16", 2: "float16", 3: "float32"}[sample_code],
            "color_transform": {0: "none", 1: "green", 2: "red", 3: "blue", 4: "YCoCg-R"}[transform],
            "tile_size": tile_size, "byte_planes": plane_count, "file_size": p.stat().st_size,
            "format": "PXL v7 outer high-precision container with PXL v8 byte-plane streams"}


def decompress(input_path, output_path, progress=None, backend="auto", workers=None):
    src, dst = Path(input_path), Path(output_path)
    info = inspect(src)
    with tempfile.TemporaryDirectory(prefix="pxl7_decode_") as td:
        td = Path(td)
        from pxl4_engine import decompress as decompress_v6
        with src.open("rb") as f:
            header = f.read(core.HEADER.size)
            if len(header) != core.HEADER.size:
                raise ValueError("Truncated PXL v7 header")
            _, _, _, channels, flags, width, height, tile_size, bytes_per_sample, plane_count = core.HEADER.unpack(header)
            sample_code, color_transform = flags & 0x03, flags >> 2
            if (width, height, channels) != (info["width"], info["height"], info["channels"]):
                raise ValueError("PXL v7 header changed during decode")
            shuffled = np.empty((height, width, channels, bytes_per_sample), dtype=np.uint8)
            for i in range(plane_count):
                entry = f.read(PLANE_ENTRY.size)
                if len(entry) != PLANE_ENTRY.size:
                    raise ValueError("Truncated PXL v7 plane table")
                length, expected_crc = PLANE_ENTRY.unpack(entry)
                max_plane_archive = width * height * channels + MAX_EMBEDDED_ARCHIVE_OVERHEAD
                if length < 1 or length > max_plane_archive:
                    raise ValueError("PXL v7 embedded plane exceeds safety limit")

                # Copy one embedded archive to disk in bounded chunks. Keeping
                # every compressed byte-plane in a Python list amplified peak
                # memory use by up to the complete high-precision image size.
                archive = td / f"plane_{i}.pxl"
                restored = td / f"plane_{i}.png"
                remaining = length
                with archive.open("wb") as plane_stream:
                    while remaining:
                        chunk = f.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Truncated PXL v7 embedded plane")
                        plane_stream.write(chunk)
                        remaining -= len(chunk)

                decompress_v6(archive, restored, progress=None, backend=backend, workers=workers)
                with Image.open(restored) as image:
                    if image.size != (width, height):
                        raise ValueError("Decoded PXL v7 byte-plane dimensions do not match header")
                    plane = np.asarray(image, dtype=np.uint8)
                    if channels == 1:
                        plane = plane.reshape(height, width, 1)
                    elif plane.ndim != 3 or plane.shape[2] != channels:
                        raise ValueError("Decoded PXL v7 byte-plane channel count mismatch")
                    plane = np.ascontiguousarray(plane)
                crc = binascii.crc32(plane.tobytes()) & 0xFFFFFFFF
                if crc != expected_crc:
                    raise ValueError(f"PXL v7 byte-plane {i} checksum failed")
                shuffled[..., i] = plane
                if progress:
                    progress(int((i + 1) * 100 / plane_count), f"PXL v7 reconstructed byte plane {i + 1}/{plane_count}")
            if f.read(1):
                raise ValueError("Trailing bytes after PXL v7 planes")
    dtype = {SAMPLE_UINT16: np.dtype("<u2"), SAMPLE_FLOAT16: np.dtype("<f2"), SAMPLE_FLOAT32: np.dtype("<f4")}[sample_code]
    result = shuffled.reshape(height, width, channels * bytes_per_sample).view(dtype)
    result = result.reshape((height, width, channels))
    if sample_code == SAMPLE_UINT16 and color_transform:
        result = _inverse_uint16_color(result, channels, color_transform)
    if channels == 1:
        result = result[..., 0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() == ".png":
        if sample_code != SAMPLE_UINT16:
            raise ValueError("Floating-point HDR output must use .tif/.tiff or .exr; PNG stores integer samples")
        if channels == 1:
            Image.fromarray(result.astype(np.uint16, copy=False)).save(dst)
        else:
            try:
                import png as pypng
            except Exception as exc:
                raise RuntimeError("Writing 16-bit RGB/RGBA PNG requires pypng; install it with `python -m pip install pypng`") from exc
            writer = pypng.Writer(width=width, height=height, bitdepth=16,
                                  greyscale=False, alpha=(channels == 4))
            rows = (np.asarray(result[y]).reshape(-1).tolist() for y in range(height))
            with dst.open("wb") as stream:
                writer.write(stream, rows)
    elif dst.suffix.lower() == ".exr":
        if sample_code == SAMPLE_UINT16:
            raise ValueError("OpenEXR export is for float16/float32 samples; use .tif/.tiff or .png for uint16")
        _write_exr(dst, result, channels)
    elif dst.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
        except Exception as exc:
            raise RuntimeError("Writing 16-bit RGB/RGBA and floating-point HDR output requires tifffile; install it with `python -m pip install tifffile`") from exc
        kwargs = {"photometric": "rgb" if channels in (3, 4) else "minisblack", "metadata": None}
        tifffile.imwrite(str(dst), result, **kwargs)
    else:
        raise ValueError("Choose .png for uint16 samples, .tif/.tiff for high-precision samples, or .exr for floating-point HDR")
    return {"output_bytes": dst.stat().st_size, "width": width, "height": height,
            "mode": "lossless", "sample_format": info["sample_format"], "channels": channels,
            "version": 7, "byte_planes": plane_count}
