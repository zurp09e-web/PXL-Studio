#!/usr/bin/env python3
"""PXL v8 public API with legacy v1-v7 readers and a custom image codec.

PXL is a custom, tiled container and transform pipeline. Pillow handles image
I/O; NumPy accelerates the block-DCT lossy profile.
"""
from __future__ import annotations

import argparse
import binascii
import math
import multiprocessing
import os
import struct
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

try:
    from numba import njit as _numba_njit
except Exception:  # optional JIT; pure Python remains fully supported
    _numba_njit = None

MAGIC = b"PXL\0"
VERSION = 3  # Version used internally by the retained v3 codec implementation.
CURRENT_VERSION = 8
LEGACY_V2 = 2
# magic, version, profile, channels, flags, width, height, tile_size, quality, reserved
HEADER = struct.Struct("<4sBBBBIIHBB")
# Legacy v1 structures are retained for backwards decoding.
HEADER_V1 = struct.Struct("<4sBBIIIBBH")
TILE_HEADER_V1 = struct.Struct("<IHHBBIII")
# v2 tile width, tile height, codec, transform, payload size, decoded pixel bytes, CRC32
TILE_HEADER = struct.Struct("<HHBBIII")

PROFILE_LOSSLESS = 0
PROFILE_LOSSY = 1

CODEC_RAW = 0
CODEC_DEFLATE = 1
CODEC_PALETTE = 2
CODEC_DCT = 3
CODEC_RANS = 4
CODEC_RANS_ZERO = 5
CODEC_PALETTE_RANS = 6
CODEC_RANS_CONTEXT = 7
CODEC_RANS_ZERO_CONTEXT = 8
CODEC_RAW_PIXELS = 9
CODEC_RANS_CONTEXT16 = 10
CODEC_RANS_SPATIAL = 12
CODEC_LZ_RANS = 13
CODEC_LZ_CONTEXT = 14

P_NONE, P_SUB, P_UP, P_AVG, P_PAETH, P_GRADIENT, P_MED = range(7)
TRANSFORM_NONE = 0
TRANSFORM_G = 1
TRANSFORM_R = 2
TRANSFORM_B = 3
TRANSFORM_YCOCG = 4

MIN_DIMENSION = 1
MAX_DIMENSION = 100_000
MIN_TILE = 64
MAX_TILE = 1024
MAX_PIXELS = 100_000_000
MAX_TILE_PAYLOAD = 256 * 1024 * 1024

Progress = Optional[Callable[[int, str], None]]
_CUPY = None
_CUPY_CHECKED = False
_CUPY_ERROR = None


def _get_cupy(required: bool = False, refresh: bool = False):
    """Return CuPy only after a real allocation, kernel launch and sync succeed.

    A failed probe can be retried with refresh=True after the user installs a
    wheel, updates a driver, or fixes CUDA_PATH without restarting Python.
    """
    global _CUPY, _CUPY_CHECKED, _CUPY_ERROR
    if refresh:
        _CUPY = None
        _CUPY_CHECKED = False
        _CUPY_ERROR = None
    if not _CUPY_CHECKED:
        _CUPY_CHECKED = True
        try:
            import cupy as cp  # optional dependency; CPU installs stay lightweight
            count = int(cp.cuda.runtime.getDeviceCount())
            if count < 1:
                raise RuntimeError("CuPy imported, but the CUDA driver reported zero usable devices")
            # getDeviceCount alone is not enough: it can succeed even when the
            # first context creation, NVRTC load, or actual kernel launch fails.
            with cp.cuda.Device(0):
                probe = cp.asarray([1, 2, 3], dtype=cp.int32)
                total = int(cp.sum(probe).get())
                cp.cuda.get_current_stream().synchronize()
            if total != 6:
                raise RuntimeError(f"CUDA smoke test returned {total}, expected 6")
            _CUPY = cp
            _CUPY_ERROR = None
        except Exception as exc:
            _CUPY = None
            _CUPY_ERROR = f"{type(exc).__name__}: {exc}"
    if required and _CUPY is None:
        detail = _CUPY_ERROR or "CuPy/CUDA is unavailable"
        raise RuntimeError(
            "CUDA backend requested, but the CUDA smoke test failed: " + detail +
            ". Install a CuPy wheel matching your driver (see README), then use Recheck CUDA."
        )
    return _CUPY


def _cuda_version_string(value):
    try:
        value = int(value)
        return f"{value // 1000}.{(value % 1000) // 10}"
    except Exception:
        return "unknown"


def cuda_status(refresh: bool = False) -> dict:
    """Return detailed CUDA diagnostics after a small real GPU smoke test."""
    cp = _get_cupy(False, refresh=refresh)
    if cp is None:
        return {"available": False, "device": None,
                "reason": _CUPY_ERROR or "CuPy/CUDA device unavailable"}
    try:
        device_id = int(cp.cuda.runtime.getDevice())
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props.get("name", b"NVIDIA GPU")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        major, minor = props.get("major"), props.get("minor")
        return {
            "available": True,
            "device": str(name),
            "device_id": device_id,
            "compute_capability": f"{major}.{minor}" if major is not None and minor is not None else "unknown",
            "cupy_version": getattr(cp, "__version__", "unknown"),
            "driver_version": _cuda_version_string(cp.cuda.runtime.driverGetVersion()),
            "runtime_version": _cuda_version_string(cp.cuda.runtime.runtimeGetVersion()),
            "smoke_test": "PASS",
        }
    except Exception as exc:
        return {"available": False, "device": None,
                "reason": f"{type(exc).__name__}: {exc}"}


def _resolve_backend(backend: str, workers: Optional[int]) -> tuple[str, int]:
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")
    if backend == "auto":
        backend = "cuda" if _get_cupy(False) is not None else "cpu"
    if backend == "cuda":
        _get_cupy(required=True)
        return "cuda", 1
    count = os.cpu_count() or 1
    if workers == -1:
        worker_count = count
    else:
        worker_count = min(8, max(1, count - 1)) if workers in (None, 0) else int(workers)
    if worker_count < 1 or worker_count > count:
        raise ValueError(f"workers must be between 1 and {count} (or -1 for all cores)")
    return "cpu", worker_count


def _report(cb: Progress, pct: int, message: str) -> None:
    if cb:
        cb(max(0, min(100, int(pct))), message)


def _normalise_image(image: Image.Image) -> Image.Image:
    """Normalize to a supported 8-bit layout while preserving displayed pixels.

    PXL stores raster samples, not EXIF/ICC/text metadata. High-bit-depth source
    images are rejected instead of silently down-converted.
    """
    if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        raise ValueError("The standard 8-bit image path cannot normalize this sample type; high-precision input must use the sample-preserving PXL v7/v8 path.")
    if image.mode == "1":
        return image.convert("L")
    if image.mode == "L":
        return image.copy()
    if image.mode == "LA":
        return image.convert("RGBA")
    if image.mode == "RGBA":
        return image.copy()
    if "A" in image.getbands() or "transparency" in image.info:
        return image.convert("RGBA")
    return image.convert("RGB")


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _predictor_value(pred: int, left: int, up: int, upper_left: int) -> int:
    if pred == P_NONE:
        return 0
    if pred == P_SUB:
        return left
    if pred == P_UP:
        return up
    if pred == P_AVG:
        return (left + up) // 2
    if pred == P_PAETH:
        return _paeth(left, up, upper_left)
    if pred == P_GRADIENT:
        return left + up - upper_left
    if pred == P_MED:
        lo, hi = min(left, up), max(left, up)
        if upper_left >= hi:
            return lo
        if upper_left <= lo:
            return hi
        return left + up - upper_left
    raise ValueError(f"Unknown predictor {pred}")


def _entropy_cost_array(values: np.ndarray) -> float:
    """Shannon coding estimate for an 8-bit NumPy vector."""
    hist = np.bincount(values, minlength=256)
    nz = hist[hist != 0].astype(np.float64)
    n = values.size
    return float(n * math.log2(n) - np.sum(nz * np.log2(nz))) if n else 0.0


def _forward_color_cuda(data: bytes, channels: int, transform: int):
    cp = _get_cupy(required=True)
    if channels == 1:
        return cp.asarray(np.frombuffer(data, dtype=np.uint8))
    if transform == TRANSFORM_NONE:
        return cp.asarray(np.frombuffer(data, dtype=np.uint8).reshape(-1, channels))
    src = cp.asarray(np.frombuffer(data, dtype=np.uint8).reshape(-1, channels))
    out = cp.empty_like(src)
    a, b, c = src[:, 0].astype(cp.int16), src[:, 1].astype(cp.int16), src[:, 2].astype(cp.int16)
    if transform == TRANSFORM_G:
        out[:, 0], out[:, 1], out[:, 2] = b, (a-b) & 255, (c-b) & 255
    elif transform == TRANSFORM_R:
        out[:, 0], out[:, 1], out[:, 2] = a, (b-a) & 255, (c-a) & 255
    elif transform == TRANSFORM_B:
        out[:, 0], out[:, 1], out[:, 2] = c, (a-c) & 255, (b-c) & 255
    elif transform == TRANSFORM_YCOCG:
        co = (a - c) & 255
        t = (c + (co >> 1)) & 255
        cg = (b - t) & 255
        y = (t + (cg >> 1)) & 255
        out[:, 0], out[:, 1], out[:, 2] = y, cg, co
    else:
        raise ValueError("Unknown PXL color transform")
    if channels == 4:
        out[:, 3] = src[:, 3]
    return out


def _filter_rows_cuda(data, width: int, height: int, channels: int) -> bytes:
    """GPU predictor search. Final PXL-native entropy coding remains on CPU.

    CUDA ranks row predictors by circular residual magnitude, a fast GPU-side
    proxy for entropy. The decoder is format-identical and does not need CUDA.
    """
    cp = _get_cupy(required=True)
    stride = width * channels
    if isinstance(data, cp.ndarray):
        rows = data.reshape(height, stride).astype(cp.uint8, copy=False)
    else:
        rows = cp.asarray(np.frombuffer(data, dtype=np.uint8).reshape(height, stride))
    wide = rows.astype(cp.int16)
    left = cp.zeros_like(wide)
    if stride > channels:
        left[:, channels:] = wide[:, :-channels]
    up = cp.zeros_like(wide)
    if height > 1:
        up[1:, :] = wide[:-1, :]
    upper_left = cp.zeros_like(wide)
    if height > 1 and stride > channels:
        upper_left[1:, channels:] = wide[:-1, :-channels]
    paeth_p = left + up - upper_left
    pa = cp.abs(paeth_p - left)
    pb = cp.abs(paeth_p - up)
    pc = cp.abs(paeth_p - upper_left)
    paeth = cp.where((pa <= pb) & (pa <= pc), left,
                     cp.where(pb <= pc, up, upper_left))
    predictors = cp.stack((cp.zeros_like(wide), left, up,
                           (left + up) // 2, paeth), axis=0)
    residuals = ((wide[None, :, :] - predictors) & 255).astype(cp.uint8)
    residual_i = residuals.astype(cp.int16)
    magnitude = cp.minimum(residual_i, 256 - residual_i)
    scores = cp.sum(magnitude, axis=2, dtype=cp.int64)
    choices = cp.argmin(scores, axis=0).astype(cp.uint8)
    chosen = residuals[choices, cp.arange(height)]
    packed = cp.concatenate((choices[:, None], chosen), axis=1).reshape(-1)
    return cp.asnumpy(packed).tobytes()



if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _filter_rows_kernel_numba(source, width, height, channels):
        stride = width * channels
        output = np.empty(height * (stride + 1), dtype=np.uint8)
        current = np.empty(stride, dtype=np.uint8)
        best_residual = np.empty(stride, dtype=np.uint8)
        candidate = np.empty(stride, dtype=np.uint8)
        for y in range(height):
            best_score = 1.0e300
            best_predictor = 0
            for pred in range(7):
                hist = np.zeros(256, dtype=np.int64)
                for i in range(stride):
                    value = int(source[y*stride+i])
                    left = int(source[y*stride+i-channels]) if i >= channels else 0
                    up = int(source[(y-1)*stride+i]) if y > 0 else 0
                    upper_left = int(source[(y-1)*stride+i-channels]) if y > 0 and i >= channels else 0
                    if pred == 0:
                        prediction = 0
                    elif pred == 1:
                        prediction = left
                    elif pred == 2:
                        prediction = up
                    elif pred == 3:
                        prediction = (left + up) // 2
                    elif pred == 4:
                        pp = left + up - upper_left
                        pa = abs(pp-left); pb = abs(pp-up); pc = abs(pp-upper_left)
                        if pa <= pb and pa <= pc:
                            prediction = left
                        elif pb <= pc:
                            prediction = up
                        else:
                            prediction = upper_left
                    elif pred == 5:
                        prediction = left + up - upper_left
                    else:
                        lo = min(left, up); hi = max(left, up)
                        if upper_left >= hi:
                            prediction = lo
                        elif upper_left <= lo:
                            prediction = hi
                        else:
                            prediction = left + up - upper_left
                    residual = (value - prediction) & 255
                    candidate[i] = residual
                    hist[residual] += 1
                score = stride * math.log2(stride)
                for symbol in range(256):
                    count = hist[symbol]
                    if count:
                        score -= count * math.log2(count)
                if score < best_score:
                    best_score = score
                    best_predictor = pred
                    for i in range(stride):
                        best_residual[i] = candidate[i]
            offset = y * (stride + 1)
            output[offset] = best_predictor
            for i in range(stride):
                output[offset+1+i] = best_residual[i]
        return output

def _filter_rows(data: bytes, width: int, height: int, channels: int) -> bytes:
    if _numba_njit is not None and len(data) >= 4096:
        source = np.frombuffer(data, dtype=np.uint8)
        return _filter_rows_kernel_numba(source, width, height, channels).tobytes()
    stride = width * channels
    out = bytearray()
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        row = np.frombuffer(data, dtype=np.uint8, count=stride, offset=y*stride)
        wide = row.astype(np.int16)
        left = np.zeros(stride, dtype=np.uint8)
        left[channels:] = row[:-channels]
        upper_left = np.zeros(stride, dtype=np.uint8)
        upper_left[channels:] = prev[:-channels]
        left_i = left.astype(np.int16)
        up_i = prev.astype(np.int16)
        ul_i = upper_left.astype(np.int16)
        paeth_p = left_i + up_i - ul_i
        pa = np.abs(paeth_p-left_i)
        pb = np.abs(paeth_p-up_i)
        pc = np.abs(paeth_p-ul_i)
        paeth = np.where((pa <= pb) & (pa <= pc), left_i,
                         np.where(pb <= pc, up_i, ul_i))
        gradient = left_i + up_i - ul_i
        lo, hi = np.minimum(left_i, up_i), np.maximum(left_i, up_i)
        med = np.where(ul_i >= hi, lo, np.where(ul_i <= lo, hi, gradient))
        predictors = (
            np.zeros(stride, dtype=np.int16), left_i, up_i,
            (left_i + up_i) // 2, paeth, gradient, med
        )
        best_score = None
        best_pred = 0
        best_residual = None
        for pred_id, prediction in enumerate(predictors):
            residual = ((wide - prediction) & 255).astype(np.uint8)
            score = _entropy_cost_array(residual)
            if best_score is None or score < best_score:
                best_score, best_pred, best_residual = score, pred_id, residual
        out.append(best_pred)
        out.extend(best_residual.tobytes())
        prev = row
    return bytes(out)


def _filter_rows_v3(data: bytes, width: int, height: int, channels: int) -> bytes:
    """V3 channel-adaptive row filtering: one predictor choice per component."""
    stride = width * channels
    out = bytearray()
    prev = np.zeros((width, channels), dtype=np.uint8)
    for y in range(height):
        row = np.frombuffer(data, dtype=np.uint8, count=stride, offset=y*stride).reshape(width, channels)
        residuals = np.empty_like(row)
        for ch in range(channels):
            current = row[:, ch]
            wide = current.astype(np.int16)
            left = np.zeros(width, dtype=np.int16)
            up = prev[:, ch].astype(np.int16)
            upper_left = np.zeros(width, dtype=np.int16)
            if width > 1:
                left[1:] = wide[:-1]
                upper_left[1:] = up[:-1]
            paeth_p = left + up - upper_left
            pa = np.abs(paeth_p-left)
            pb = np.abs(paeth_p-up)
            pc = np.abs(paeth_p-upper_left)
            paeth = np.where((pa <= pb) & (pa <= pc), left,
                             np.where(pb <= pc, up, upper_left))
            gradient = left + up - upper_left
            lo, hi = np.minimum(left, up), np.maximum(left, up)
            med = np.where(upper_left >= hi, lo,
                           np.where(upper_left <= lo, hi, gradient))
            predictors = (np.zeros(width, dtype=np.int16), left, up,
                          (left+up)//2, paeth, gradient, med)
            best_score = None
            best_id = 0
            best_residual = None
            for pred_id, prediction in enumerate(predictors):
                residual = ((wide - prediction) & 255).astype(np.uint8)
                entropy_score = _entropy_cost_array(residual)
                residual_i = residual.astype(np.int16)
                magnitude_score = int(np.minimum(residual_i, 256-residual_i).sum())
                score = (entropy_score, magnitude_score)
                if best_score is None or score < best_score:
                    best_score, best_id, best_residual = score, pred_id, residual
            out.append(best_id)
            residuals[:, ch] = best_residual
        out.extend(residuals.tobytes())
        prev = row
    return bytes(out)


def _unfilter_rows_v3(payload: bytes, width: int, height: int, channels: int) -> bytes:
    stride = width * channels
    expected = height * (stride + channels)
    if len(payload) != expected:
        raise ValueError("Invalid PXL v3 channel-predictive stream length")
    output = bytearray(width * height * channels)
    pos = 0
    prev = bytearray(stride)
    for y in range(height):
        predictors = payload[pos:pos+channels]
        pos += channels
        residual = payload[pos:pos+stride]
        pos += stride
        row = bytearray(stride)
        for x in range(width):
            base = x * channels
            for ch in range(channels):
                i = base + ch
                pred = predictors[ch]
                if pred > P_MED:
                    raise ValueError("Invalid PXL v3 predictor")
                left = row[i-channels] if x else 0
                up = prev[i]
                ul = prev[i-channels] if x else 0
                row[i] = (residual[i] + _predictor_value(pred, left, up, ul)) & 255
        output[y*stride:(y+1)*stride] = row
        prev = row
    return bytes(output)


def _filter_rows_cuda_v3(data, width: int, height: int, channels: int) -> bytes:
    """CUDA equivalent of v3 per-channel predictor selection."""
    cp = _get_cupy(required=True)
    if isinstance(data, cp.ndarray):
        rows = data.reshape(height, width, channels).astype(cp.uint8, copy=False)
    else:
        rows = cp.asarray(np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels))
    wide = rows.astype(cp.int16)
    predictor_ids = cp.empty((height, channels), dtype=cp.uint8)
    residuals_out = cp.empty((height, width, channels), dtype=cp.uint8)
    for ch in range(channels):
        component = wide[:, :, ch]
        left = cp.zeros_like(component)
        if width > 1:
            left[:, 1:] = component[:, :-1]
        up = cp.zeros_like(component)
        if height > 1:
            up[1:, :] = component[:-1, :]
        upper_left = cp.zeros_like(component)
        if height > 1 and width > 1:
            upper_left[1:, 1:] = component[:-1, :-1]
        pp = left + up - upper_left
        pa, pb, pc = cp.abs(pp-left), cp.abs(pp-up), cp.abs(pp-upper_left)
        paeth = cp.where((pa <= pb) & (pa <= pc), left,
                         cp.where(pb <= pc, up, upper_left))
        gradient = left + up - upper_left
        lo, hi = cp.minimum(left, up), cp.maximum(left, up)
        med = cp.where(upper_left >= hi, lo, cp.where(upper_left <= lo, hi, gradient))
        preds = cp.stack((cp.zeros_like(component), left, up, (left+up)//2,
                          paeth, gradient, med), axis=0)
        residuals = ((component[None, :, :] - preds) & 255).astype(cp.uint8)
        signed = residuals.astype(cp.int16)
        magnitude = cp.minimum(signed, 256-signed)
        scores = cp.sum(magnitude, axis=2, dtype=cp.int64)
        choices = cp.argmin(scores, axis=0).astype(cp.uint8)
        residuals_out[:, :, ch] = residuals[choices, cp.arange(height)]
        predictor_ids[:, ch] = choices
    packed = cp.concatenate((predictor_ids, residuals_out.reshape(height, width*channels)), axis=1)
    return cp.asnumpy(packed).tobytes()


if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _unfilter_rows_kernel_numba(source, width, height, channels):
        stride = width * channels
        row_size = stride + 1
        out = np.empty(width * height * channels, dtype=np.uint8)
        for y in range(height):
            source_base = y * row_size
            if source[source_base] > P_MED:
                return np.empty(0, dtype=np.uint8)
            pred = int(source[source_base])
            for i in range(stride):
                x_channel = i % channels
                left = int(out[y * stride + i - channels]) if i >= channels else 0
                up = int(out[(y - 1) * stride + i]) if y else 0
                ul = int(out[(y - 1) * stride + i - channels]) if y and i >= channels else 0
                if pred == P_NONE:
                    prediction = 0
                elif pred == P_SUB:
                    prediction = left
                elif pred == P_UP:
                    prediction = up
                elif pred == P_AVG:
                    prediction = (left + up) // 2
                elif pred == P_PAETH:
                    pp = left + up - ul
                    pa = abs(pp - left); pb = abs(pp - up); pc = abs(pp - ul)
                    if pa <= pb and pa <= pc:
                        prediction = left
                    elif pb <= pc:
                        prediction = up
                    else:
                        prediction = ul
                elif pred == P_GRADIENT:
                    prediction = left + up - ul
                else:
                    lo = min(left, up); hi = max(left, up)
                    if ul >= hi:
                        prediction = lo
                    elif ul <= lo:
                        prediction = hi
                    else:
                        prediction = left + up - ul
                out[y * stride + i] = (int(source[source_base + 1 + i]) + prediction) & 255
        return out


def _unfilter_rows(payload: bytes, width: int, height: int, channels: int) -> bytes:
    stride = width * channels
    expected = height * (stride + 1)
    if len(payload) != expected:
        raise ValueError("Invalid predictive tile stream length")
    if _numba_njit is not None and len(payload) >= 1024:
        decoded = _unfilter_rows_kernel_numba(np.frombuffer(payload, dtype=np.uint8),
                                              width, height, channels)
        if decoded.size != width * height * channels:
            raise ValueError("Invalid predictor in PXL tile")
        return decoded.tobytes()
    out = bytearray(width * height * channels)
    pos = 0
    prev = b""
    for y in range(height):
        pred = payload[pos]
        pos += 1
        residual = payload[pos:pos+stride]
        pos += stride
        if pred not in (P_NONE, P_SUB, P_UP, P_AVG, P_PAETH, P_GRADIENT, P_MED):
            raise ValueError("Invalid predictor in PXL tile")
        row = bytearray(stride)
        for i, value in enumerate(residual):
            left = row[i-channels] if i >= channels else 0
            up = prev[i] if prev else 0
            ul = prev[i-channels] if prev and i >= channels else 0
            row[i] = (value + _predictor_value(pred, left, up, ul)) & 255
        out[y*stride:(y+1)*stride] = row
        prev = bytes(row)
    return bytes(out)


def _forward_color(data: bytes, channels: int, transform: int) -> bytes:
    if channels == 1 or transform == TRANSFORM_NONE:
        return data
    src = np.frombuffer(data, dtype=np.uint8).reshape(-1, channels)
    out = np.empty_like(src)
    a, b, c = src[:, 0].astype(np.int16), src[:, 1].astype(np.int16), src[:, 2].astype(np.int16)
    if transform == TRANSFORM_G:
        out[:, 0] = b
        out[:, 1] = ((a-b) & 255).astype(np.uint8)
        out[:, 2] = ((c-b) & 255).astype(np.uint8)
    elif transform == TRANSFORM_R:
        out[:, 0] = a
        out[:, 1] = ((b-a) & 255).astype(np.uint8)
        out[:, 2] = ((c-a) & 255).astype(np.uint8)
    elif transform == TRANSFORM_B:
        out[:, 0] = c
        out[:, 1] = ((a-c) & 255).astype(np.uint8)
        out[:, 2] = ((b-c) & 255).astype(np.uint8)
    elif transform == TRANSFORM_YCOCG:
        co = (a-c) & 255
        t = (c + (co >> 1)) & 255
        cg = (b-t) & 255
        y = (t + (cg >> 1)) & 255
        out[:, 0] = y.astype(np.uint8)
        out[:, 1] = cg.astype(np.uint8)
        out[:, 2] = co.astype(np.uint8)
    else:
        raise ValueError("Unknown PXL color transform")
    if channels == 4:
        out[:, 3] = src[:, 3]
    return out.tobytes()


def _inverse_color(data: bytes, channels: int, transform: int) -> bytes:
    if channels == 1 or transform == TRANSFORM_NONE:
        return data
    src = np.frombuffer(data, dtype=np.uint8).reshape(-1, channels)
    out = np.empty_like(src)
    x, y, z = src[:, 0].astype(np.int16), src[:, 1].astype(np.int16), src[:, 2].astype(np.int16)
    if transform == TRANSFORM_G:
        out[:, 0] = ((y+x) & 255).astype(np.uint8)
        out[:, 1] = x
        out[:, 2] = ((z+x) & 255).astype(np.uint8)
    elif transform == TRANSFORM_R:
        out[:, 0] = x
        out[:, 1] = ((y+x) & 255).astype(np.uint8)
        out[:, 2] = ((z+x) & 255).astype(np.uint8)
    elif transform == TRANSFORM_B:
        out[:, 0] = ((y+x) & 255).astype(np.uint8)
        out[:, 1] = ((z+x) & 255).astype(np.uint8)
        out[:, 2] = x
    elif transform == TRANSFORM_YCOCG:
        yy, cg, co = x, y, z
        t = (yy - (cg >> 1)) & 255
        g = (cg + t) & 255
        b = (t - (co >> 1)) & 255
        r = (b + co) & 255
        out[:, 0] = r.astype(np.uint8)
        out[:, 1] = g.astype(np.uint8)
        out[:, 2] = b.astype(np.uint8)
    else:
        raise ValueError("Unknown PXL color transform")
    if channels == 4:
        out[:, 3] = src[:, 3]
    return out.tobytes()


def _pack_palette(data: bytes, channels: int) -> Optional[bytes]:
    """Encode a tile with <=256 distinct pixels using tightly packed indices."""
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(-1, channels)
    # A small sample can prove that the tile has >256 unique colours without
    # sorting every pixel. If it cannot prove that, fall back to exact counting.
    if pixels.shape[0] > 256:
        sample_step = max(1, pixels.shape[0] // 2048)
        sample = pixels[::sample_step]
        sample_records = sample.view(np.dtype((np.void, channels))).reshape(-1)
        if np.unique(sample_records).size > 256:
            return None
    # Treat each pixel as one fixed-width record. This avoids NumPy's slower
    # axis-wise unique machinery while preserving lexicographic palette order.
    records = pixels.view(np.dtype((np.void, channels))).reshape(-1)
    palette_records, inverse = np.unique(records, return_inverse=True)
    count = len(palette_records)
    palette = palette_records.view(np.uint8).reshape(count, channels)
    if count > 256:
        return None
    bits = max(1, (count-1).bit_length())
    positions = np.arange(inverse.size, dtype=np.uint64) * bits
    byte_indices = (positions >> 3).astype(np.intp)
    shifts = (positions & 7).astype(np.uint8)
    values = inverse.astype(np.uint16)
    packed = np.zeros((inverse.size*bits+7)//8, dtype=np.uint8)
    np.bitwise_or.at(packed, byte_indices, ((values << shifts) & 255).astype(np.uint8))
    cross = shifts.astype(np.uint16) + bits > 8
    if np.any(cross):
        np.bitwise_or.at(packed, byte_indices[cross]+1,
                         (values[cross] >> (8-shifts[cross])).astype(np.uint8))
    return bytes((count & 255, bits)) + palette.tobytes() + packed.tobytes()


def _unpack_palette(payload: bytes, width: int, height: int, channels: int) -> bytes:
    if len(payload) < 2:
        raise ValueError("Truncated PXL palette tile")
    count = payload[0] or 256
    bits = payload[1]
    if bits not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError("Invalid PXL palette bit depth")
    if count > (1 << bits):
        raise ValueError("Invalid PXL palette size")
    pal_end = 2 + count*channels
    pixel_count = width*height
    packed_len = (pixel_count*bits+7)//8
    if len(payload) != pal_end+packed_len:
        raise ValueError("Invalid PXL palette payload length")
    palette = np.frombuffer(payload, dtype=np.uint8, count=count*channels, offset=2).reshape(count, channels)
    packed = np.frombuffer(payload, dtype=np.uint8, count=packed_len, offset=pal_end)
    positions = np.arange(pixel_count, dtype=np.uint64)*bits
    byte_indices = (positions >> 3).astype(np.intp)
    shifts = (positions & 7).astype(np.uint8)
    values = (packed[byte_indices].astype(np.uint16) >> shifts)
    cross = shifts.astype(np.uint16)+bits > 8
    if np.any(cross):
        values[cross] |= packed[byte_indices[cross]+1].astype(np.uint16) << (8-shifts[cross])
    indices = (values & ((1 << bits)-1)).astype(np.intp)
    if np.any(indices >= count):
        raise ValueError("PXL palette index out of range")
    return palette[indices].tobytes()


# PXL-Native Entropy Layer ---------------------------------------------------
# Static byte rANS with a compact sparse/dense frequency table. This is our
# own entropy coder; it does not call zlib, DEFLATE, Brotli, Zstandard, etc.
_RANS_SCALE_BITS = 12
_RANS_TOTAL = 1 << _RANS_SCALE_BITS
_RANS_L = 1 << 23


def _normalise_frequency_counts(counts: np.ndarray) -> list[int]:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    frequencies = [0] * 256
    if total == 0:
        raise ValueError("PXL rANS cannot encode an empty context")
    present = [int(x) for x in np.flatnonzero(counts)]
    remainders = []
    for symbol in present:
        q, r = divmod(int(counts[symbol]) * _RANS_TOTAL, total)
        frequencies[symbol] = max(1, q)
        remainders.append((r, symbol))
    delta = _RANS_TOTAL - sum(frequencies)
    if delta > 0:
        remainders.sort(reverse=True)
        for i in range(delta):
            frequencies[remainders[i % len(remainders)][1]] += 1
    elif delta < 0:
        need = -delta
        rem = {sym: r for r, sym in remainders}
        while need:
            candidates = [sym for sym in present if frequencies[sym] > 1]
            if not candidates:
                raise ValueError("Unable to normalise PXL rANS frequencies")
            candidates.sort(key=lambda sym: (rem[sym], -frequencies[sym]))
            for sym in candidates:
                if not need:
                    break
                frequencies[sym] -= 1
                need -= 1
    if sum(frequencies) != _RANS_TOTAL or any(frequencies[x] == 0 for x in present):
        raise AssertionError("Invalid PXL rANS frequency table")
    return frequencies


def _normalise_frequencies(data: bytes) -> list[int]:
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    return _normalise_frequency_counts(counts)


def _uvarint_bytes(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7f) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_uvarint_bytes(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for i in range(5):
        if pos >= len(data):
            raise ValueError("Truncated PXL entropy table varint")
        b = data[pos]
        pos += 1
        # This field is a uint32 varint: only four low bits are usable
        # in its fifth byte.
        if i == 4 and b > 0x0F:
            raise ValueError("Overflowing PXL entropy table varint")
        value |= (b & 0x7f) << (i * 7)
        if not (b & 0x80):
            return value, pos
    raise ValueError("Oversized PXL entropy table varint")


def _rans_encode_kernel_python(data: np.ndarray, frequencies: np.ndarray, cumulative: np.ndarray):
    state = _RANS_L
    emitted = bytearray()
    for i in range(data.size - 1, -1, -1):
        symbol = int(data[i])
        f = int(frequencies[symbol])
        x_max = ((_RANS_L >> _RANS_SCALE_BITS) << 8) * f
        while state >= x_max:
            emitted.append(state & 0xff)
            state >>= 8
        state = ((state // f) << _RANS_SCALE_BITS) + (state % f) + int(cumulative[symbol])
    return state, bytes(reversed(emitted))


if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _rans_encode_kernel_numba(data, frequencies, cumulative):
        state = np.uint64(_RANS_L)
        emitted = np.empty(data.size * 2 + 16, dtype=np.uint8)
        used = 0
        for i in range(data.size - 1, -1, -1):
            symbol = int(data[i])
            f = np.uint64(frequencies[symbol])
            x_max = np.uint64(((_RANS_L >> _RANS_SCALE_BITS) << 8)) * f
            while state >= x_max:
                emitted[used] = np.uint8(state & np.uint64(255))
                used += 1
                state >>= np.uint64(8)
            state = ((state // f) << np.uint64(_RANS_SCALE_BITS)) + (state % f) + np.uint64(cumulative[symbol])
        return state, emitted[:used][::-1].copy()

    @_numba_njit(cache=True)
    def _rans_decode_kernel(blob, start_pos, output_size, frequencies, cumulative, decode_table):
        state = np.uint64(0)
        for j in range(4):
            state |= np.uint64(blob[start_pos+j]) << np.uint64(8*j)
        if state < np.uint64(_RANS_L):
            return np.empty(output_size, dtype=np.uint8), -2
        pos = start_pos + 4
        output = np.empty(output_size, dtype=np.uint8)
        for i in range(output_size):
            slot = int(state & np.uint64(_RANS_TOTAL-1))
            symbol = int(decode_table[slot])
            f = np.uint64(frequencies[symbol])
            if f == 0:
                return output, -2
            state = f * (state >> np.uint64(_RANS_SCALE_BITS)) + np.uint64(slot - int(cumulative[symbol]))
            while state < np.uint64(_RANS_L):
                if pos >= blob.size:
                    return output, -1
                state = (state << np.uint64(8)) | np.uint64(blob[pos])
                pos += 1
            output[i] = symbol
        return output, pos

    @_numba_njit(cache=True)
    def _rans_run_encode_kernel(data):
        out = np.empty(data.size * 2 + 16, dtype=np.uint8)
        used = 0
        i = 0
        while i < data.size:
            value = int(data[i])
            j = i + 1
            while j < data.size and int(data[j]) == value:
                j += 1
            run = j - i
            if value == 0 and run >= 3:
                out[used] = 0; used += 1
                out[used] = 1; used += 1
                v = run
                while v >= 128:
                    out[used] = (v & 127) | 128; used += 1; v >>= 7
                out[used] = v; used += 1
            elif value != 0 and run >= 5:
                out[used] = 0; used += 1
                out[used] = 2; used += 1
                out[used] = value; used += 1
                v = run
                while v >= 128:
                    out[used] = (v & 127) | 128; used += 1; v >>= 7
                out[used] = v; used += 1
            elif value == 0:
                for _ in range(run):
                    out[used] = 0; used += 1
                    out[used] = 0; used += 1
            else:
                for k in range(i, j):
                    out[used] = data[k]; used += 1
            i = j
        return out[:used].copy()

    @_numba_njit(cache=True)
    def _rans_context_encode_kernel(data, frequencies, cumulative):
        state = np.uint64(_RANS_L)
        emitted = np.empty(data.size * 2 + 16, dtype=np.uint8)
        used = 0
        for i in range(data.size - 1, -1, -1):
            ctx = 0
            if i > 0:
                ctx = int(data[i-1]) // 64
            symbol = int(data[i])
            f = np.uint64(frequencies[ctx, symbol])
            x_max = np.uint64(((_RANS_L >> _RANS_SCALE_BITS) << 8)) * f
            while state >= x_max:
                emitted[used] = np.uint8(state & np.uint64(255))
                used += 1
                state >>= np.uint64(8)
            state = ((state // f) << np.uint64(_RANS_SCALE_BITS)) + (state % f) + np.uint64(cumulative[ctx, symbol])
        return state, emitted[:used][::-1].copy()

    @_numba_njit(cache=True)
    def _rans_context_decode_kernel(blob, start_pos, output_size, frequencies, cumulative, decode_tables):
        state = np.uint64(0)
        for j in range(4):
            state |= np.uint64(blob[start_pos+j]) << np.uint64(8*j)
        pos = start_pos + 4
        if state < np.uint64(_RANS_L):
            return np.empty(output_size, dtype=np.uint8), -2
        output = np.empty(output_size, dtype=np.uint8)
        for i in range(output_size):
            ctx = 0
            if i > 0:
                ctx = int(output[i-1]) // 64
            slot = int(state & np.uint64(_RANS_TOTAL-1))
            symbol = int(decode_tables[ctx, slot])
            f = np.uint64(frequencies[ctx, symbol])
            if f == 0:
                return output, -2
            state = f * (state >> np.uint64(_RANS_SCALE_BITS)) + np.uint64(slot - int(cumulative[ctx, symbol]))
            while state < np.uint64(_RANS_L):
                if pos >= blob.size:
                    return output, -1
                state = (state << np.uint64(8)) | np.uint64(blob[pos])
                pos += 1
            output[i] = symbol
        return output, pos


def _rans_encode(data: bytes) -> bytes:
    """Encode bytes using a tile-local, static 12-bit byte rANS model."""
    if not data:
        raise ValueError("Cannot encode empty PXL entropy stream")
    freq = _normalise_frequencies(data)
    cumulative = [0] * 257
    for i in range(256):
        cumulative[i+1] = cumulative[i] + freq[i]

    source = np.frombuffer(data, dtype=np.uint8)
    freq_array = np.asarray(freq, dtype=np.uint64)
    cumulative_array = np.asarray(cumulative, dtype=np.uint64)
    if _numba_njit is not None:
        state, emitted_array = _rans_encode_kernel_numba(source, freq_array, cumulative_array)
        state = int(state)
        emitted = emitted_array.tobytes()
    else:
        state, emitted = _rans_encode_kernel_python(source, freq_array, cumulative_array)

    dense = b"".join(int(x).to_bytes(2, "little") for x in freq)
    nonzero = [i for i, f in enumerate(freq) if f]
    sparse = bytearray((len(nonzero) & 0xff,))  # 0 denotes all 256 symbols
    for symbol in nonzero:
        sparse.append(symbol)
        sparse.extend(_uvarint_bytes(freq[symbol]))
    if len(sparse) < len(dense):
        table = b"\x01" + bytes(sparse)
    else:
        table = b"\x00" + dense
    return table + state.to_bytes(4, "little") + emitted


def _serialize_rans_table(freq: list[int]) -> bytes:
    dense = b"".join(int(x).to_bytes(2, "little") for x in freq)
    nonzero = [i for i, f in enumerate(freq) if f]
    sparse = bytearray((len(nonzero) & 0xff,))
    for symbol in nonzero:
        sparse.append(symbol)
        sparse.extend(_uvarint_bytes(freq[symbol]))
    if len(sparse) < len(dense):
        return b"\x01" + bytes(sparse)
    return b"\x00" + dense


def _parse_rans_table(blob: bytes, pos: int) -> tuple[list[int], int]:
    if pos >= len(blob):
        raise ValueError("Truncated PXL rANS table")
    mode = blob[pos]
    pos += 1
    freq = [0] * 256
    if mode == 0:
        if pos + 512 > len(blob):
            raise ValueError("Truncated PXL dense rANS table")
        for symbol in range(256):
            freq[symbol] = int.from_bytes(blob[pos+2*symbol:pos+2*symbol+2], "little")
        pos += 512
    elif mode == 1:
        if pos >= len(blob):
            raise ValueError("Truncated PXL sparse rANS table")
        count = blob[pos] or 256
        pos += 1
        previous = -1
        for _ in range(count):
            if pos >= len(blob):
                raise ValueError("Truncated PXL sparse rANS symbol")
            symbol = blob[pos]
            pos += 1
            if symbol <= previous:
                raise ValueError("Unsorted or duplicate PXL rANS symbols")
            previous = symbol
            value, pos = _read_uvarint_bytes(blob, pos)
            if value < 1 or value > _RANS_TOTAL:
                raise ValueError("Invalid PXL rANS frequency")
            freq[symbol] = value
    else:
        raise ValueError("Unknown PXL rANS table mode")
    if sum(freq) != _RANS_TOTAL:
        raise ValueError("Invalid PXL rANS frequency total")
    return freq, pos


def _rans_context_encode(data: bytes) -> bytes:
    """Four-context order-1 rANS. Context = top two bits of previous byte."""
    if not data:
        raise ValueError("Cannot encode empty PXL context stream")
    arr = np.frombuffer(data, dtype=np.uint8)
    context_ids = np.empty(arr.size, dtype=np.uint8)
    context_ids[0] = 0
    context_ids[1:] = arr[:-1] >> 6
    frequencies = np.zeros((4, 256), dtype=np.uint64)
    cumulative = np.zeros((4, 256), dtype=np.uint64)
    table = bytearray((4,))
    histogram = np.bincount(context_ids.astype(np.int64) * 256 + arr.astype(np.int64),
                            minlength=4 * 256).reshape(4, 256)
    for ctx in range(4):
        if not histogram[ctx].any():
            table.extend(b"\x00\x00")
            continue
        freq = _normalise_frequency_counts(histogram[ctx])
        frequencies[ctx] = np.asarray(freq, dtype=np.uint64)
        c = 0
        for symbol in range(256):
            cumulative[ctx, symbol] = c
            c += freq[symbol]
        encoded_table = _serialize_rans_table(freq)
        table.extend(len(encoded_table).to_bytes(2, "little"))
        table.extend(encoded_table)
    state, emitted = _rans_context_encode_kernel(arr, frequencies, cumulative) if _numba_njit is not None else _rans_context_encode_python(arr, frequencies, cumulative)
    return bytes(table) + int(state).to_bytes(4, "little") + (emitted.tobytes() if isinstance(emitted, np.ndarray) else emitted)


def _rans_context_encode_python(data, frequencies, cumulative):
    state = _RANS_L
    emitted = bytearray()
    for i in range(data.size - 1, -1, -1):
        ctx = 0 if i == 0 else int(data[i-1]) >> 6
        symbol = int(data[i])
        f = int(frequencies[ctx, symbol])
        x_max = ((_RANS_L >> _RANS_SCALE_BITS) << 8) * f
        while state >= x_max:
            emitted.append(state & 255)
            state >>= 8
        state = ((state // f) << _RANS_SCALE_BITS) + (state % f) + int(cumulative[ctx, symbol])
    return state, bytes(reversed(emitted))


def _rans_context_decode(blob: bytes, output_size: int) -> bytes:
    if len(blob) < 5 or blob[0] != 4:
        raise ValueError("Invalid PXL context-rANS header")
    pos = 1
    frequencies = np.zeros((4, 256), dtype=np.uint64)
    cumulative = np.zeros((4, 256), dtype=np.uint64)
    decode_tables = np.zeros((4, _RANS_TOTAL), dtype=np.uint8)
    for ctx in range(4):
        if pos + 2 > len(blob):
            raise ValueError("Truncated PXL context table length")
        table_size = int.from_bytes(blob[pos:pos+2], "little")
        pos += 2
        if table_size == 0:
            continue
        if pos + table_size > len(blob):
            raise ValueError("Truncated PXL context frequency table")
        freq, end = _parse_rans_table(blob[:pos+table_size], pos)
        if end != pos + table_size:
            raise ValueError("Invalid PXL context table length")
        pos += table_size
        frequencies[ctx] = np.asarray(freq, dtype=np.uint64)
        c = 0
        for symbol in range(256):
            cumulative[ctx, symbol] = c
            f = freq[symbol]
            if f:
                decode_tables[ctx, c:c+f] = symbol
            c += f
    if pos + 4 > len(blob):
        raise ValueError("Truncated PXL context-rANS state")
    if output_size > MAX_TILE_PAYLOAD:
        raise ValueError("PXL context-rANS output exceeds safety limit")
    if _numba_njit is not None:
        output, end_pos = _rans_context_decode_kernel(np.frombuffer(blob, dtype=np.uint8), pos, output_size,
                                                      frequencies, cumulative, decode_tables)
        if end_pos == -1:
            raise ValueError("Truncated PXL context-rANS renormalisation data")
        if end_pos == -2:
            raise ValueError("Invalid PXL context-rANS state/model")
        if end_pos != len(blob):
            raise ValueError("Trailing data in PXL context-rANS stream")
        return output.tobytes()
    state = int.from_bytes(blob[pos:pos+4], "little")
    pos += 4
    if state < _RANS_L:
        raise ValueError("Invalid PXL context-rANS initial state")
    output = bytearray(output_size)
    for i in range(output_size):
        ctx = 0 if i == 0 else output[i-1] >> 6
        slot = state & (_RANS_TOTAL-1)
        symbol = int(decode_tables[ctx, slot])
        f = int(frequencies[ctx, symbol])
        state = f * (state >> _RANS_SCALE_BITS) + slot - int(cumulative[ctx, symbol])
        while state < _RANS_L:
            if pos >= len(blob):
                raise ValueError("Truncated PXL context-rANS stream")
            state = (state << 8) | blob[pos]
            pos += 1
        output[i] = symbol
    if pos != len(blob):
        raise ValueError("Trailing data in PXL context-rANS stream")
    return bytes(output)



if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _rans_context16_encode_kernel(data, frequencies, cumulative):
        state = np.uint64(_RANS_L)
        emitted = np.empty(data.size * 2 + 16, dtype=np.uint8)
        used = 0
        for i in range(data.size - 1, -1, -1):
            ctx = np.int64(0)
            if i > 0:
                ctx = np.int64(data[i-1]) >> 4
            symbol = int(data[i])
            f = np.uint64(frequencies[ctx, symbol])
            x_max = np.uint64(((_RANS_L >> _RANS_SCALE_BITS) << 8)) * f
            while state >= x_max:
                emitted[used] = np.uint8(state & np.uint64(255))
                used += 1
                state >>= np.uint64(8)
            state = ((state // f) << np.uint64(_RANS_SCALE_BITS)) + (state % f) + np.uint64(cumulative[ctx, symbol])
        return state, emitted[:used][::-1].copy()

    @_numba_njit(cache=True)
    def _rans_context16_decode_kernel(blob, start_pos, output_size, frequencies, cumulative, decode_tables):
        state = np.uint64(0)
        for j in range(4):
            state |= np.uint64(blob[start_pos+j]) << np.uint64(8*j)
        if state < np.uint64(_RANS_L):
            return np.empty(output_size, dtype=np.uint8), -2
        pos = start_pos + 4
        output = np.empty(output_size, dtype=np.uint8)
        for i in range(output_size):
            ctx = np.int64(0)
            if i > 0:
                ctx = np.int64(output[i-1]) >> 4
            slot = int(state & np.uint64(_RANS_TOTAL-1))
            symbol = int(decode_tables[ctx, slot])
            f = np.uint64(frequencies[ctx, symbol])
            if f == 0:
                return output, -2
            state = f * (state >> np.uint64(_RANS_SCALE_BITS)) + np.uint64(slot - int(cumulative[ctx, symbol]))
            while state < np.uint64(_RANS_L):
                if pos >= blob.size:
                    return output, -1
                state = (state << np.uint64(8)) | np.uint64(blob[pos])
                pos += 1
            output[i] = symbol
        return output, pos


def _rans_context16_encode(data: bytes) -> bytes:
    """Sixteen-context static rANS; context is previous symbol's upper nibble."""
    if not data:
        raise ValueError("Cannot encode empty PXL context16 stream")
    arr = np.frombuffer(data, dtype=np.uint8)
    context_ids = np.empty(arr.size, dtype=np.uint8)
    context_ids[0] = 0
    context_ids[1:] = arr[:-1] >> 4
    frequencies = np.zeros((16, 256), dtype=np.uint64)
    cumulative = np.zeros((16, 256), dtype=np.uint64)
    table = bytearray((16,))
    histogram = np.bincount(context_ids.astype(np.int64) * 256 + arr.astype(np.int64),
                            minlength=16 * 256).reshape(16, 256)
    for ctx in range(16):
        if not histogram[ctx].any():
            table.extend(b"\x00\x00")
            continue
        freq = _normalise_frequency_counts(histogram[ctx])
        frequencies[ctx] = np.asarray(freq, dtype=np.uint64)
        c = 0
        for symbol in range(256):
            cumulative[ctx, symbol] = c
            c += freq[symbol]
        encoded_table = _serialize_rans_table(freq)
        table.extend(len(encoded_table).to_bytes(2, "little"))
        table.extend(encoded_table)
    native = None
    try:
        from pxl_native import get_backend as _get_native_backend
        native = _get_native_backend(False)
    except Exception:
        native = None
    if native is not None:
        state, emitted_bytes = native.rans_encode_context16(data, frequencies, cumulative)
    elif _numba_njit is not None:
        state, emitted = _rans_context16_encode_kernel(arr, frequencies, cumulative)
        emitted_bytes = emitted.tobytes()
    else:
        state = _RANS_L
        emitted_buf = bytearray()
        for i in range(arr.size - 1, -1, -1):
            ctx = 0 if i == 0 else int(arr[i-1]) >> 4
            symbol = int(arr[i])
            f = int(frequencies[ctx, symbol])
            x_max = ((_RANS_L >> _RANS_SCALE_BITS) << 8) * f
            while state >= x_max:
                emitted_buf.append(state & 255)
                state >>= 8
            state = ((state // f) << _RANS_SCALE_BITS) + (state % f) + int(cumulative[ctx, symbol])
        emitted_bytes = bytes(reversed(emitted_buf))
    return bytes(table) + int(state).to_bytes(4, "little") + emitted_bytes


def _rans_context16_decode(blob: bytes, output_size: int) -> bytes:
    if len(blob) < 5 or blob[0] != 16:
        raise ValueError("Invalid PXL context16-rANS header")
    if output_size < 0 or output_size > MAX_TILE_PAYLOAD:
        raise ValueError("PXL context16-rANS output exceeds safety limit")
    pos = 1
    frequencies = np.zeros((16, 256), dtype=np.uint64)
    cumulative = np.zeros((16, 256), dtype=np.uint64)
    decode_tables = np.zeros((16, _RANS_TOTAL), dtype=np.uint8)
    for ctx in range(16):
        if pos + 2 > len(blob):
            raise ValueError("Truncated PXL context16 table length")
        table_size = int.from_bytes(blob[pos:pos+2], "little")
        pos += 2
        if table_size == 0:
            continue
        if pos + table_size > len(blob):
            raise ValueError("Truncated PXL context16 frequency table")
        freq, end = _parse_rans_table(blob[:pos+table_size], pos)
        if end != pos + table_size:
            raise ValueError("Invalid PXL context16 table length")
        pos += table_size
        frequencies[ctx] = np.asarray(freq, dtype=np.uint64)
        c = 0
        for symbol in range(256):
            cumulative[ctx, symbol] = c
            f = freq[symbol]
            if f:
                decode_tables[ctx, c:c+f] = symbol
            c += f
    if pos + 4 > len(blob):
        raise ValueError("Truncated PXL context16-rANS state")
    native = None
    try:
        from pxl_native import get_backend as _get_native_backend
        native = _get_native_backend(False)
    except Exception:
        native = None
    if native is not None:
        output, end_pos = native.rans_decode_context16(blob, pos, output_size,
                                                       frequencies, cumulative, decode_tables)
        if end_pos != len(blob):
            raise ValueError("Trailing data in PXL context16-rANS stream")
        return output
    if _numba_njit is not None:
        output, end_pos = _rans_context16_decode_kernel(np.frombuffer(blob, dtype=np.uint8), pos, output_size,
                                                        frequencies, cumulative, decode_tables)
        if end_pos == -1:
            raise ValueError("Truncated PXL context16-rANS renormalisation data")
        if end_pos == -2:
            raise ValueError("Invalid PXL context16-rANS state/model")
        if end_pos != len(blob):
            raise ValueError("Trailing data in PXL context16-rANS stream")
        return output.tobytes()
    state = int.from_bytes(blob[pos:pos+4], "little")
    pos += 4
    if state < _RANS_L:
        raise ValueError("Invalid PXL context16-rANS initial state")
    output = bytearray(output_size)
    for i in range(output_size):
        ctx = 0 if i == 0 else output[i-1] >> 4
        slot = state & (_RANS_TOTAL-1)
        symbol = int(decode_tables[ctx, slot])
        f = int(frequencies[ctx, symbol])
        if f == 0:
            raise ValueError("Invalid PXL context16-rANS model symbol")
        state = f * (state >> _RANS_SCALE_BITS) + slot - int(cumulative[ctx, symbol])
        while state < _RANS_L:
            if pos >= len(blob):
                raise ValueError("Truncated PXL context16-rANS data")
            state = (state << 8) | blob[pos]
            pos += 1
        output[i] = symbol
    if pos != len(blob):
        raise ValueError("Trailing data in PXL context16-rANS stream")
    return bytes(output)


if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _rans_context_ids_encode_kernel(data, context_ids, frequencies, cumulative):
        state = np.uint64(_RANS_L)
        emitted = np.empty(data.size * 2 + 16, dtype=np.uint8)
        used = 0
        for i in range(data.size - 1, -1, -1):
            ctx = int(context_ids[i])
            symbol = int(data[i])
            f = np.uint64(frequencies[ctx, symbol])
            x_max = np.uint64(((_RANS_L >> _RANS_SCALE_BITS) << 8)) * f
            while state >= x_max:
                emitted[used] = np.uint8(state & np.uint64(255))
                used += 1
                state >>= np.uint64(8)
            state = ((state // f) << np.uint64(_RANS_SCALE_BITS)) + (state % f) + np.uint64(cumulative[ctx, symbol])
        return state, emitted[:used][::-1].copy()

    @_numba_njit(cache=True)
    def _rans_spatial_decode_kernel(blob, start_pos, output_size, width, height, channels,
                                    frequencies, cumulative, decode_tables):
        state = np.uint64(0)
        for j in range(4):
            state |= np.uint64(blob[start_pos+j]) << np.uint64(8*j)
        if state < np.uint64(_RANS_L):
            return np.empty(output_size, dtype=np.uint8), -2
        pos = start_pos + 4
        output = np.empty(output_size, dtype=np.uint8)
        rowbytes = width * channels + 1
        predictor_context = channels * 4
        for i in range(output_size):
            col = i % rowbytes
            if col == 0:
                ctx = predictor_context
            else:
                sample_pos = col - 1
                ch = sample_pos % channels
                previous = 0
                if sample_pos >= channels:
                    previous = int(output[i-channels])
                ctx = ch * 4 + (previous >> 6)
            slot = int(state & np.uint64(_RANS_TOTAL-1))
            symbol = int(decode_tables[ctx, slot])
            f = np.uint64(frequencies[ctx, symbol])
            if f == 0:
                return output, -2
            state = f * (state >> np.uint64(_RANS_SCALE_BITS)) + np.uint64(slot - int(cumulative[ctx, symbol]))
            while state < np.uint64(_RANS_L):
                if pos >= blob.size:
                    return output, -1
                state = (state << np.uint64(8)) | np.uint64(blob[pos])
                pos += 1
            output[i] = symbol
        return output, pos


def _spatial_context_ids(data: np.ndarray, width: int, height: int, channels: int) -> np.ndarray:
    rowbytes = width * channels + 1
    positions = np.arange(data.size, dtype=np.int64)
    columns = positions % rowbytes
    sample_positions = columns - 1
    contexts = np.full(data.size, channels * 4, dtype=np.uint8)
    samples = columns != 0
    sample_channels = np.zeros(data.size, dtype=np.int64)
    sample_channels[samples] = sample_positions[samples] % channels
    previous = np.zeros(data.size, dtype=np.uint8)
    has_left = samples & (sample_positions >= channels)
    previous[has_left] = data[positions[has_left] - channels]
    contexts[samples] = (sample_channels[samples] * 4 + (previous[samples] >> 6)).astype(np.uint8)
    return contexts


def _rans_spatial_encode(data: bytes, width: int, height: int, channels: int) -> bytes:
    """Context model tied to PXL's row/pixel layout, not just byte adjacency."""
    try:
        from pxl_native import get_backend as _get_native_backend
        native = _get_native_backend(False)
    except Exception:
        native = None
    if native is not None:
        return native.rans_encode_spatial(data, width, height, channels)
    if not data:
        raise ValueError("Cannot encode empty PXL spatial context stream")
    expected = height * (width * channels + 1)
    if len(data) != expected:
        raise ValueError("Invalid PXL spatial context input geometry")
    arr = np.frombuffer(data, dtype=np.uint8)
    context_ids = _spatial_context_ids(arr, width, height, channels)
    context_count = channels * 4 + 1
    frequencies = np.zeros((context_count, 256), dtype=np.uint64)
    cumulative = np.zeros((context_count, 256), dtype=np.uint64)
    table = bytearray((context_count,))
    histogram = np.bincount(context_ids.astype(np.int64) * 256 + arr.astype(np.int64),
                            minlength=context_count * 256).reshape(context_count, 256)
    for ctx in range(context_count):
        if not histogram[ctx].any():
            table.extend(b"\x00\x00")
            continue
        freq = _normalise_frequency_counts(histogram[ctx])
        frequencies[ctx] = np.asarray(freq, dtype=np.uint64)
        c = 0
        for symbol in range(256):
            cumulative[ctx, symbol] = c
            c += freq[symbol]
        encoded_table = _serialize_rans_table(freq)
        table.extend(len(encoded_table).to_bytes(2, "little"))
        table.extend(encoded_table)
    if _numba_njit is not None:
        state, emitted = _rans_context_ids_encode_kernel(arr, context_ids, frequencies, cumulative)
        emitted_bytes = emitted.tobytes()
    else:
        state = _RANS_L
        emitted_buf = bytearray()
        for i in range(arr.size - 1, -1, -1):
            ctx = int(context_ids[i])
            symbol = int(arr[i])
            f = int(frequencies[ctx, symbol])
            x_max = ((_RANS_L >> _RANS_SCALE_BITS) << 8) * f
            while state >= x_max:
                emitted_buf.append(state & 255)
                state >>= 8
            state = ((state // f) << _RANS_SCALE_BITS) + (state % f) + int(cumulative[ctx, symbol])
        emitted_bytes = bytes(reversed(emitted_buf))
    return bytes(table) + int(state).to_bytes(4, "little") + emitted_bytes


def _rans_spatial_decode(blob: bytes, output_size: int, width: int, height: int, channels: int) -> bytes:
    try:
        from pxl_native import get_backend as _get_native_backend
        native = _get_native_backend(False)
    except Exception:
        native = None
    if native is not None:
        return native.rans_decode_spatial(blob, output_size, width, height, channels)
    context_count = channels * 4 + 1
    if len(blob) < 5 or blob[0] != context_count:
        raise ValueError("Invalid PXL spatial-context rANS header")
    if output_size != height * (width * channels + 1) or output_size > MAX_TILE_PAYLOAD:
        raise ValueError("Invalid PXL spatial-context output geometry")
    pos = 1
    frequencies = np.zeros((context_count, 256), dtype=np.uint64)
    cumulative = np.zeros((context_count, 256), dtype=np.uint64)
    decode_tables = np.zeros((context_count, _RANS_TOTAL), dtype=np.uint8)
    for ctx in range(context_count):
        if pos + 2 > len(blob):
            raise ValueError("Truncated PXL spatial-context table length")
        table_size = int.from_bytes(blob[pos:pos+2], "little")
        pos += 2
        if table_size == 0:
            continue
        if pos + table_size > len(blob):
            raise ValueError("Truncated PXL spatial-context frequency table")
        freq, end = _parse_rans_table(blob[:pos+table_size], pos)
        if end != pos + table_size:
            raise ValueError("Invalid PXL spatial-context table length")
        pos += table_size
        frequencies[ctx] = np.asarray(freq, dtype=np.uint64)
        c = 0
        for symbol in range(256):
            cumulative[ctx, symbol] = c
            f = freq[symbol]
            if f:
                decode_tables[ctx, c:c+f] = symbol
            c += f
    if pos + 4 > len(blob):
        raise ValueError("Truncated PXL spatial-context state")
    if _numba_njit is not None:
        output, end_pos = _rans_spatial_decode_kernel(np.frombuffer(blob, dtype=np.uint8), pos, output_size,
                                                      width, height, channels, frequencies, cumulative, decode_tables)
        if end_pos == -1:
            raise ValueError("Truncated PXL spatial-context renormalisation data")
        if end_pos == -2:
            raise ValueError("Invalid PXL spatial-context state/model")
        if end_pos != len(blob):
            raise ValueError("Trailing data in PXL spatial-context stream")
        return output.tobytes()
    state = int.from_bytes(blob[pos:pos+4], "little")
    pos += 4
    if state < _RANS_L:
        raise ValueError("Invalid PXL spatial-context initial state")
    output = bytearray(output_size)
    rowbytes = width * channels + 1
    for i in range(output_size):
        col = i % rowbytes
        if col == 0:
            ctx = channels * 4
        else:
            sample_pos = col - 1
            ch = sample_pos % channels
            previous = output[i-channels] if sample_pos >= channels else 0
            ctx = ch * 4 + (previous >> 6)
        slot = state & (_RANS_TOTAL-1)
        symbol = int(decode_tables[ctx, slot])
        f = int(frequencies[ctx, symbol])
        if f == 0:
            raise ValueError("Invalid PXL spatial-context model symbol")
        state = f * (state >> _RANS_SCALE_BITS) + slot - int(cumulative[ctx, symbol])
        while state < _RANS_L:
            if pos >= len(blob):
                raise ValueError("Truncated PXL spatial-context stream")
            state = (state << 8) | blob[pos]
            pos += 1
        output[i] = symbol
    if pos != len(blob):
        raise ValueError("Trailing data in PXL spatial-context stream")
    return bytes(output)

def _rans_decode(blob: bytes, output_size: int) -> bytes:
    """Decode one bounded PXL rANS stream and reject malformed tables/states."""
    if output_size < 0 or output_size > MAX_TILE_PAYLOAD:
        raise ValueError("Invalid PXL entropy output size")
    if len(blob) < 6:
        raise ValueError("Truncated PXL rANS stream")
    mode = blob[0]
    pos = 1
    freq = [0] * 256
    if mode == 0:
        if len(blob) < pos + 512 + 4:
            raise ValueError("Truncated PXL dense rANS table")
        for symbol in range(256):
            freq[symbol] = int.from_bytes(blob[pos+2*symbol:pos+2*symbol+2], "little")
        pos += 512
    elif mode == 1:
        count = blob[pos] or 256
        pos += 1
        previous = -1
        for _ in range(count):
            if pos >= len(blob):
                raise ValueError("Truncated PXL sparse rANS table")
            symbol = blob[pos]
            pos += 1
            if symbol <= previous:
                raise ValueError("Unsorted or duplicate PXL rANS symbols")
            previous = symbol
            value, pos = _read_uvarint_bytes(blob, pos)
            if value < 1 or value > _RANS_TOTAL:
                raise ValueError("Invalid PXL rANS frequency")
            freq[symbol] = value
    else:
        raise ValueError("Unknown PXL entropy table mode")
    if sum(freq) != _RANS_TOTAL:
        raise ValueError("Invalid PXL rANS frequency total")
    if pos + 4 > len(blob):
        raise ValueError("Truncated PXL rANS state")
    state = int.from_bytes(blob[pos:pos+4], "little")
    pos += 4
    if state < _RANS_L:
        raise ValueError("Invalid PXL rANS initial state")

    cumulative = [0] * 257
    for i in range(256):
        cumulative[i+1] = cumulative[i] + freq[i]
    decode_table = np.zeros(_RANS_TOTAL, dtype=np.uint8)
    for symbol, f in enumerate(freq):
        if f:
            decode_table[cumulative[symbol]:cumulative[symbol+1]] = symbol

    if _numba_njit is not None:
        output, end_pos = _rans_decode_kernel(np.frombuffer(blob, dtype=np.uint8), pos-4, output_size,
                                               np.asarray(freq, dtype=np.uint64),
                                               np.asarray(cumulative[:-1], dtype=np.uint64), decode_table)
        if end_pos == -1:
            raise ValueError("Truncated PXL rANS renormalisation data")
        if end_pos == -2:
            raise ValueError("Invalid PXL rANS state/model")
        if end_pos != len(blob):
            raise ValueError("Trailing bytes in PXL rANS stream")
        return output.tobytes()

    output = bytearray(output_size)
    for i in range(output_size):
        slot = state & (_RANS_TOTAL - 1)
        symbol = int(decode_table[slot])
        f = freq[symbol]
        if f == 0:
            raise ValueError("Invalid PXL rANS symbol")
        state = f * (state >> _RANS_SCALE_BITS) + slot - cumulative[symbol]
        while state < _RANS_L:
            if pos >= len(blob):
                raise ValueError("Truncated PXL rANS renormalisation data")
            state = (state << 8) | blob[pos]
            pos += 1
        output[i] = symbol
    if pos != len(blob):
        raise ValueError("Trailing bytes in PXL rANS stream")
    return bytes(output)



if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _lz_run_encode_kernel(data):
        """PXL LZ-token prefilter: bounded hash-chain matches + zero/repeat runs."""
        n = data.size
        out = np.empty(n * 2 + 32, dtype=np.uint8)
        head = np.full(1 << 16, -1, dtype=np.int32)
        previous = np.full(n, -1, dtype=np.int32)
        used = 0
        i = 0
        while i < n:
            value = int(data[i])
            j = i + 1
            while j < n and int(data[j]) == value:
                j += 1
            run = j - i
            # Run tokens are simpler and cheaper than dictionary matches.
            if (value == 0 and run >= 3) or (value != 0 and run >= 5):
                if value == 0:
                    out[used] = 0; used += 1
                    out[used] = 1; used += 1
                else:
                    out[used] = 0; used += 1
                    out[used] = 2; used += 1
                    out[used] = value; used += 1
                v = run
                while v >= 128:
                    out[used] = (v & 127) | 128; used += 1; v >>= 7
                out[used] = v; used += 1
                # Keep the hash chain current for positions swallowed by a run.
                for p in range(i, j):
                    if p + 2 < n:
                        h = ((int(data[p]) * 251 + int(data[p+1])) * 251 + int(data[p+2])) & 65535
                        previous[p] = head[h]
                        head[h] = p
                i = j
                continue

            best_length = 0
            best_distance = 0
            best_candidate = -1
            h = -1
            if i + 2 < n:
                h = ((value * 251 + int(data[i+1])) * 251 + int(data[i+2])) & 65535
                candidate_pos = int(head[h])
                attempts = 0
                while candidate_pos >= 0 and i - candidate_pos <= 65535 and attempts < 8:
                    length = 0
                    limit = n - i
                    while length < limit and int(data[candidate_pos+length]) == int(data[i+length]):
                        length += 1
                    if length > best_length:
                        best_length = length
                        best_distance = i - candidate_pos
                        best_candidate = candidate_pos
                        if length >= 128:
                            break
                    candidate_pos = int(previous[candidate_pos])
                    attempts += 1
                previous[i] = head[h]
                head[h] = i

            if best_length >= 5:
                out[used] = 0; used += 1
                out[used] = 3; used += 1
                v = best_distance
                while v >= 128:
                    out[used] = (v & 127) | 128; used += 1; v >>= 7
                out[used] = v; used += 1
                v = best_length
                while v >= 128:
                    out[used] = (v & 127) | 128; used += 1; v >>= 7
                out[used] = v; used += 1
                end = i + best_length
                for p in range(i+1, end):
                    if p + 2 < n:
                        hp = ((int(data[p]) * 251 + int(data[p+1])) * 251 + int(data[p+2])) & 65535
                        previous[p] = head[hp]
                        head[hp] = p
                i = end
            else:
                if value == 0:
                    out[used] = 0; used += 1
                    out[used] = 0; used += 1
                else:
                    out[used] = value; used += 1
                i += 1
        return out[:used].copy()

def _run_encode(data: bytes) -> bytes:
    """PXL token prefilter: zero runs and repeated-byte runs, fully reversible."""
    if _numba_njit is not None and len(data) >= 512:
        return _rans_run_encode_kernel(np.frombuffer(data, dtype=np.uint8)).tobytes()
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        value = data[i]
        j = i + 1
        while j < n and data[j] == value and (j - i) < 0x7fffffff:
            j += 1
        run = j - i
        if value == 0 and run >= 3:
            out.extend((0, 1))
            out.extend(_uvarint_bytes(run))
        elif value != 0 and run >= 5:
            out.extend((0, 2, value))
            out.extend(_uvarint_bytes(run))
        elif value == 0:
            out.extend((0, 0))
            if run == 2:
                out.extend((0, 0))
        else:
            out.extend(data[i:j])
        i = j
    return bytes(out)



def _lz_run_encode(data: bytes) -> bytes:
    """PXL-owned LZ+run tokenization with JIT and pure-Python implementations."""
    if len(data) < 512:
        return _run_encode(data)
    if _numba_njit is not None:
        return _lz_run_encode_kernel(np.frombuffer(data, dtype=np.uint8)).tobytes()

    out = bytearray()
    heads: dict[int, list[int]] = {}
    n = len(data)

    def insert_position(pos: int) -> None:
        if pos + 2 >= n:
            return
        h = ((data[pos] * 251 + data[pos+1]) * 251 + data[pos+2]) & 65535
        chain = heads.setdefault(h, [])
        chain.append(pos)
        if len(chain) > 8:
            del chain[:-8]

    i = 0
    while i < n:
        value = data[i]
        j = i + 1
        while j < n and data[j] == value:
            j += 1
        run = j - i
        if value == 0 and run >= 3:
            out.extend((0, 1)); out.extend(_uvarint_bytes(run))
            for pos in range(i, j):
                insert_position(pos)
            i = j
            continue
        if value != 0 and run >= 5:
            out.extend((0, 2, value)); out.extend(_uvarint_bytes(run))
            for pos in range(i, j):
                insert_position(pos)
            i = j
            continue

        best_length = 0
        best_distance = 0
        h = -1
        if i + 2 < n:
            h = ((data[i] * 251 + data[i+1]) * 251 + data[i+2]) & 65535
            for candidate in reversed(heads.get(h, ())):
                distance = i - candidate
                if distance < 1 or distance > 65535:
                    continue
                length = 0
                limit = n - i
                while length < limit and data[candidate+length] == data[i+length]:
                    length += 1
                if length > best_length:
                    best_length, best_distance = length, distance
                    if length >= 128:
                        break
            insert_position(i)
        if best_length >= 5:
            out.extend((0, 3))
            out.extend(_uvarint_bytes(best_distance))
            out.extend(_uvarint_bytes(best_length))
            end = i + best_length
            for pos in range(i+1, end):
                insert_position(pos)
            i = end
        else:
            if value == 0:
                out.extend((0, 0))
            else:
                out.append(value)
            i += 1
    return bytes(out)

if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _run_read_varint_numba(tokens, pos):
        value = 0
        shift = 0
        for i in range(10):
            if pos >= tokens.size:
                return 0, pos, -1
            b = int(tokens[pos]); pos += 1
            # Reject overflow before a shift by 63 can lose high bits.
            if i == 9 and b > 1:
                return 0, pos, -2
            value |= (b & 127) << shift
            if (b & 128) == 0:
                return value, pos, 0
            shift += 7
        return 0, pos, -2

    @_numba_njit(cache=True)
    def _run_decode_kernel_numba(tokens, expected_size):
        out = np.empty(expected_size, dtype=np.uint8)
        pos = 0
        used = 0
        while pos < tokens.size:
            value = int(tokens[pos]); pos += 1
            if value != 0:
                if used >= expected_size:
                    return np.empty(0, dtype=np.uint8)
                out[used] = value; used += 1
                continue
            if pos >= tokens.size:
                return np.empty(0, dtype=np.uint8)
            op = int(tokens[pos]); pos += 1
            if op == 0:
                if used >= expected_size:
                    return np.empty(0, dtype=np.uint8)
                out[used] = 0; used += 1
            elif op == 1:
                run, pos, err = _run_read_varint_numba(tokens, pos)
                if err != 0 or run < 3 or used + run > expected_size:
                    return np.empty(0, dtype=np.uint8)
                out[used:used + run] = 0
                used += run
            elif op == 2:
                if pos >= tokens.size:
                    return np.empty(0, dtype=np.uint8)
                repeated = int(tokens[pos]); pos += 1
                run, pos, err = _run_read_varint_numba(tokens, pos)
                if err != 0 or repeated == 0 or run < 5 or used + run > expected_size:
                    return np.empty(0, dtype=np.uint8)
                out[used:used + run] = repeated
                used += run
            elif op == 3:
                distance, pos, err = _run_read_varint_numba(tokens, pos)
                if err != 0:
                    return np.empty(0, dtype=np.uint8)
                run, pos, err = _run_read_varint_numba(tokens, pos)
                if err != 0 or distance < 1 or distance > used or run < 5 or used + run > expected_size:
                    return np.empty(0, dtype=np.uint8)
                for _ in range(run):
                    out[used] = out[used - distance]
                    used += 1
            else:
                return np.empty(0, dtype=np.uint8)
        if used != expected_size:
            return np.empty(0, dtype=np.uint8)
        return out


def _run_decode(tokens: bytes, expected_size: int) -> bytes:
    if _numba_njit is not None and len(tokens) >= 1024:
        decoded = _run_decode_kernel_numba(np.frombuffer(tokens, dtype=np.uint8),
                                           int(expected_size))
        if decoded.size != expected_size:
            raise ValueError("Invalid PXL run token stream")
        return decoded.tobytes()
    out = bytearray()
    pos = 0
    while pos < len(tokens):
        value = tokens[pos]
        pos += 1
        if value != 0:
            out.append(value)
        else:
            if pos >= len(tokens):
                raise ValueError("Truncated PXL run token")
            op = tokens[pos]
            pos += 1
            if op == 0:
                out.append(0)
            elif op == 1:
                run, pos = _read_uvarint_bytes(tokens, pos)
                if run < 3:
                    raise ValueError("Invalid PXL zero run")
                out.extend(b"\x00" * run)
            elif op == 2:
                if pos >= len(tokens):
                    raise ValueError("Truncated PXL repeat token")
                repeated = tokens[pos]
                pos += 1
                run, pos = _read_uvarint_bytes(tokens, pos)
                if repeated == 0 or run < 5:
                    raise ValueError("Invalid PXL repeat run")
                out.extend(bytes((repeated,)) * run)
            elif op == 3:
                distance, pos = _read_uvarint_bytes(tokens, pos)
                run, pos = _read_uvarint_bytes(tokens, pos)
                if distance < 1 or distance > len(out) or run < 5:
                    raise ValueError("Invalid PXL dictionary match")
                if run > expected_size - len(out):
                    raise ValueError("PXL dictionary match expands beyond expected size")
                for _ in range(run):
                    out.append(out[-distance])
            else:
                raise ValueError("Unknown PXL run token")
        if len(out) > expected_size:
            raise ValueError("PXL run stream expands beyond expected size")
    if len(out) != expected_size:
        raise ValueError("PXL run stream decoded size mismatch")
    return bytes(out)


def _pack_entropy_stream(data: bytes) -> bytes:
    """Frame a lossy coefficient/alpha stream with raw or PXL-native entropy."""
    raw = b"\x00" + struct.pack("<I", len(data)) + data
    rans = b"\x01" + struct.pack("<I", len(data)) + _rans_encode(data)
    context = b"\x03" + struct.pack("<I", len(data)) + _rans_context_encode(data)
    token = _run_encode(data)
    run_rans = (b"\x02" + struct.pack("<II", len(data), len(token)) + _rans_encode(token))
    run_context = (b"\x04" + struct.pack("<II", len(data), len(token)) + _rans_context_encode(token))
    return min((raw, rans, context, run_rans, run_context), key=len)


def _unpack_entropy_stream(frame: bytes) -> bytes:
    if len(frame) < 5:
        raise ValueError("Truncated PXL entropy frame")
    mode = frame[0]
    expected = struct.unpack_from("<I", frame, 1)[0]
    if expected > MAX_TILE_PAYLOAD:
        raise ValueError("PXL entropy frame exceeds safety limit")
    if mode == 0:
        result = frame[5:]
        if len(result) != expected:
            raise ValueError("PXL raw entropy frame size mismatch")
        return result
    if mode == 1:
        return _rans_decode(frame[5:], expected)
    if mode in (2, 4):
        if len(frame) < 9:
            raise ValueError("Truncated PXL run entropy frame")
        token_size = struct.unpack_from("<I", frame, 5)[0]
        if token_size > MAX_TILE_PAYLOAD:
            raise ValueError("PXL token stream exceeds safety limit")
        tokens = (_rans_decode(frame[9:], token_size) if mode == 2
                  else _rans_context_decode(frame[9:], token_size))
        return _run_decode(tokens, expected)
    if mode == 3:
        return _rans_context_decode(frame[5:], expected)
    raise ValueError("Unknown PXL entropy frame mode")


def _encode_lossless_tile(raw: bytes, width: int, height: int, channels: int,
                          compression_level: int = 6, backend: str = "cpu"):
    """Encode one tile, comparing custom PXL entropy paths by actual byte size.

    `compression_level` is the effort value (1 fast, 6 balanced, 9 maximum).
    The search is deliberately staged: broad transform screening first, then
    expensive context/run coding only on the most promising candidates.
    """
    effort = int(compression_level)
    source_array = np.frombuffer(raw, dtype=np.uint8)
    source_entropy = _entropy_cost_array(source_array) / max(1, source_array.size)
    # Truly incompressible byte noise cannot benefit from model setup. This
    # raw-pixel packet is a v3 native mode and avoids spending seconds proving
    # that entropy is already essentially 8 bits/sample.
    if source_entropy > 7.985:
        return CODEC_RAW_PIXELS, TRANSFORM_NONE, raw

    transforms = (TRANSFORM_NONE, TRANSFORM_G, TRANSFORM_YCOCG) if effort <= 1 else (
        TRANSFORM_NONE, TRANSFORM_G, TRANSFORM_R, TRANSFORM_B, TRANSFORM_YCOCG
    )
    native = None
    if backend != "cuda":
        try:
            from pxl_native import get_backend as _get_native_backend
            native = _get_native_backend(False)
        except Exception:
            native = None
    initial = []
    for transform in transforms:
        if native is not None:
            # C++ fuses the reversible color transform with all seven row
            # predictor candidates, avoiding separate NumPy transform buffers.
            filtered = native.filter_legacy(raw, width, height, channels, transform)
        elif backend == "cuda":
            transformed = _forward_color_cuda(raw, channels, transform)
            filtered = _filter_rows_cuda(transformed, width, height, channels)
        else:
            transformed = _forward_color(raw, channels, transform)
            filtered = _filter_rows(transformed, width, height, channels)
        raw_candidate = (len(filtered), CODEC_RAW, transform, filtered, filtered)
        initial.append(raw_candidate)
        entropy_bits = _entropy_cost_array(np.frombuffer(filtered, dtype=np.uint8)) / max(1, len(filtered))
        # At near-uniform entropy, rANS table overhead and symbol work cannot
        # win. For smaller tiles use actual coding regardless of this estimate.
        if len(filtered) < 4096 or entropy_bits < 7.985:
            entropy = _rans_encode(filtered)
            initial.append((len(entropy), CODEC_RANS, transform, entropy, filtered))

    candidates = [(size, codec, transform, payload) for size, codec, transform, payload, _ in initial]
    # More expensive context and run models are tested on only the strongest
    # screened candidates. Maximum effort examines one extra candidate.
    ranked = sorted(initial, key=lambda x: x[0])
    deep_count = 1 if effort <= 1 else (3 if effort >= 9 else 2)
    for _, _, transform, _, filtered in ranked[:deep_count]:
        entropy_bits = _entropy_cost_array(np.frombuffer(filtered, dtype=np.uint8)) / max(1, len(filtered))
        if entropy_bits < 7.9 and len(filtered) >= 256:
            context_entropy = _rans_context_encode(filtered)
            candidates.append((len(context_entropy), CODEC_RANS_CONTEXT, transform, context_entropy))
            spatial_entropy = _rans_spatial_encode(filtered, width, height, channels)
            candidates.append((len(spatial_entropy), CODEC_RANS_SPATIAL, transform, spatial_entropy))
            if effort >= 9:
                context16_entropy = _rans_context16_encode(filtered)
                candidates.append((len(context16_entropy), CODEC_RANS_CONTEXT16, transform, context16_entropy))
        if filtered.count(b"\x00") >= max(8, len(filtered)//100) or effort >= 9:
            tokens = _run_encode(filtered)
            run_entropy = struct.pack("<I", len(tokens)) + _rans_encode(tokens)
            candidates.append((len(run_entropy), CODEC_RANS_ZERO, transform, run_entropy))
            if effort >= 9 and len(tokens) >= 256:
                run_context = struct.pack("<I", len(tokens)) + _rans_context_encode(tokens)
                candidates.append((len(run_context), CODEC_RANS_ZERO_CONTEXT, transform, run_context))
        if effort >= 6 and len(filtered) >= 1024:
            lz_tokens = _lz_run_encode(filtered)
            if len(lz_tokens) < len(filtered):
                lz_rans = struct.pack("<I", len(lz_tokens)) + _rans_encode(lz_tokens)
                candidates.append((len(lz_rans), CODEC_LZ_RANS, transform, lz_rans))
                if effort >= 9:
                    lz_context = struct.pack("<I", len(lz_tokens)) + _rans_context_encode(lz_tokens)
                    candidates.append((len(lz_context), CODEC_LZ_CONTEXT, transform, lz_context))

    # Exact palette coding is a native PXL representation. Both packed and
    # entropy-coded variants compete on final bytes, not heuristics.
    palette = _pack_palette(raw, channels)
    if palette is not None:
        candidates.append((len(palette), CODEC_PALETTE, TRANSFORM_NONE, palette))
        palette_entropy = _pack_entropy_stream(palette)
        candidates.append((len(palette_entropy), CODEC_PALETTE_RANS, TRANSFORM_NONE, palette_entropy))
    _, codec, transform, payload = min(candidates, key=lambda item: item[0])
    return codec, transform, payload


def _zlib_decompress_limited(data: bytes, max_output: int, label: str) -> bytes:
    """Legacy-only DEFLATE decoder for PXL v1/v2 compatibility."""
    try:
        decoder = zlib.decompressobj()
        out = decoder.decompress(data, max_output + 1)
    except zlib.error as e:
        raise ValueError(f"Invalid legacy {label} DEFLATE stream: {e}") from e
    if len(out) > max_output or decoder.unconsumed_tail:
        raise ValueError(f"Legacy {label} stream exceeds its decoded-size limit")
    if not decoder.eof or decoder.unused_data:
        raise ValueError(f"Invalid or trailing data in legacy {label} DEFLATE stream")
    return out


def _decode_lossless_tile(payload: bytes, width: int, height: int, channels: int,
                          codec: int, transform: int, version: int = VERSION) -> bytes:
    expected = height * (width * channels + 1)
    if codec == CODEC_RAW_PIXELS:
        if version < 3 or transform != TRANSFORM_NONE or len(payload) != width*height*channels:
            raise ValueError("Invalid PXL raw-pixel tile")
        return payload
    if codec == CODEC_PALETTE_RANS:
        if version < 3:
            raise ValueError("PXL entropy palette codec is not valid in legacy files")
        return _unpack_palette(_unpack_entropy_stream(payload), width, height, channels)
    if codec == CODEC_PALETTE:
        if version >= 3:
            return _unpack_palette(payload, width, height, channels)
        palette_limit = 2 + 256*channels + (width*height*8+7)//8
        palette_payload = _zlib_decompress_limited(payload, palette_limit, "palette")
        return _unpack_palette(palette_payload, width, height, channels)
    if codec == CODEC_RANS:
        if version < 3:
            raise ValueError("PXL rANS codec is not valid in legacy files")
        filtered = _rans_decode(payload, expected)
    elif codec == CODEC_RANS_CONTEXT:
        if version < 3:
            raise ValueError("PXL context-rANS codec is not valid in legacy files")
        filtered = _rans_context_decode(payload, expected)
    elif codec == CODEC_RANS_CONTEXT16:
        if version < 3:
            raise ValueError("PXL context16-rANS codec is not valid in legacy files")
        filtered = _rans_context16_decode(payload, expected)
    elif codec == CODEC_RANS_SPATIAL:
        if version < 3:
            raise ValueError("PXL spatial-context rANS codec is not valid in legacy files")
        filtered = _rans_spatial_decode(payload, expected, width, height, channels)
    elif codec == CODEC_RANS_ZERO_CONTEXT:
        if version < 3 or len(payload) < 4:
            raise ValueError("Invalid PXL run-context-rANS tile")
        token_size = struct.unpack_from("<I", payload, 0)[0]
        if token_size > MAX_TILE_PAYLOAD:
            raise ValueError("PXL token stream exceeds safety limit")
        filtered = _run_decode(_rans_context_decode(payload[4:], token_size), expected)
    elif codec == CODEC_RANS_ZERO:
        if version < 3 or len(payload) < 4:
            raise ValueError("Invalid PXL run-rANS tile")
        token_size = struct.unpack_from("<I", payload, 0)[0]
        if token_size > MAX_TILE_PAYLOAD:
            raise ValueError("PXL token stream exceeds safety limit")
        filtered = _run_decode(_rans_decode(payload[4:], token_size), expected)
    elif codec in (CODEC_LZ_RANS, CODEC_LZ_CONTEXT):
        if version < 3 or len(payload) < 4:
            raise ValueError("Invalid PXL LZ entropy tile")
        token_size = struct.unpack_from("<I", payload, 0)[0]
        if token_size > MAX_TILE_PAYLOAD:
            raise ValueError("PXL LZ token stream exceeds safety limit")
        token_bytes = (_rans_decode(payload[4:], token_size) if codec == CODEC_LZ_RANS
                       else _rans_context_decode(payload[4:], token_size))
        filtered = _run_decode(token_bytes, expected)
    elif codec == CODEC_DEFLATE:
        if version >= 3:
            raise ValueError("DEFLATE is not used by PXL v3 encoder")
        filtered = _zlib_decompress_limited(payload, expected, "predictive tile")
    elif codec == CODEC_RAW:
        filtered = payload
    else:
        raise ValueError(f"Unsupported lossless tile codec {codec}")
    if version >= 3:
        try:
            from pxl_native import get_backend as _get_native_backend
            native = _get_native_backend(False)
        except Exception:
            native = None
        if native is not None:
            return native.unfilter_legacy(filtered, width, height, channels, transform)
    transformed = _unfilter_rows(filtered, width, height, channels)
    return _inverse_color(transformed, channels, transform)


# JPEG-family quantization tables; PXL uses its own RLE coefficient stream.
_QY = np.array([
    [16,11,10,16,24,40,51,61], [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56], [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77], [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101], [72,92,95,98,112,100,103,99]
], dtype=np.float64)
_QC = np.array([
    [17,18,24,47,99,99,99,99], [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99], [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99], [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99], [99,99,99,99,99,99,99,99]
], dtype=np.float64)
_DCT = np.empty((8, 8), dtype=np.float64)
for _u in range(8):
    _alpha = math.sqrt(1/8) if _u == 0 else math.sqrt(2/8)
    for _x in range(8):
        _DCT[_u, _x] = _alpha * math.cos((2*_x + 1) * _u * math.pi / 16)
_ZIGZAG = np.array([
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,
    13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,
    52,45,38,31,39,46,53,60,61,54,47,55,62,63
], dtype=np.intp)


def _quant_table(quality: int, chroma: bool = False) -> np.ndarray:
    base = _QC if chroma else _QY
    scale = 5000 / quality if quality < 50 else 200 - quality * 2
    q = np.floor((base * scale + 50) / 100)
    return np.clip(q, 1, 255)


def _write_uvarint(out: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("Unsigned varint cannot be negative")
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _write_svarint(out: bytearray, value: int) -> None:
    _write_uvarint(out, (value << 1) ^ (value >> 31))


def _read_uvarint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for i in range(5):
        if pos >= len(data):
            raise ValueError("Truncated coefficient varint")
        b = data[pos]
        pos += 1
        if i == 4 and b > 0x0F:
            raise ValueError("Overflowing coefficient varint")
        value |= (b & 0x7F) << (i * 7)
        if not (b & 0x80):
            return value, pos
    raise ValueError("Oversized coefficient varint")


def _read_svarint(data: bytes, pos: int) -> tuple[int, int]:
    value, pos = _read_uvarint(data, pos)
    return (value >> 1) ^ -(value & 1), pos


def _encode_coefficients(qcoeff: np.ndarray) -> bytes:
    """Zigzag order + DC differential + zero-run coding for quantized 8x8 blocks."""
    blocks = np.ascontiguousarray(qcoeff).reshape(-1, 8, 8)
    out = bytearray()
    previous_dc = 0
    for block in blocks:
        zz = block.reshape(64)[_ZIGZAG]
        dc = int(zz[0])
        _write_svarint(out, dc - previous_dc)
        previous_dc = dc
        run = 0
        for value_np in zz[1:]:
            value = int(value_np)
            if value == 0:
                run += 1
            else:
                out.append(run)
                _write_svarint(out, value)
                run = 0
        out.append(255)  # end-of-block
    return bytes(out)


def _decode_coefficients(stream: bytes, block_count: int) -> np.ndarray:
    blocks = np.zeros((block_count, 64), dtype=np.int32)
    pos = 0
    previous_dc = 0
    for bi in range(block_count):
        diff, pos = _read_svarint(stream, pos)
        dc = previous_dc + diff
        if abs(dc) > 1_000_000:
            raise ValueError("Coefficient DC value exceeds safety limit")
        blocks[bi, 0] = dc
        previous_dc = dc
        idx = 1
        while True:
            if pos >= len(stream):
                raise ValueError("Truncated coefficient block")
            run = stream[pos]
            pos += 1
            if run == 255:
                break
            idx += run
            if idx >= 64:
                raise ValueError("Coefficient run exceeds block")
            value, pos = _read_svarint(stream, pos)
            if abs(value) > 1_000_000:
                raise ValueError("Coefficient value exceeds safety limit")
            blocks[bi, idx] = value
            idx += 1
    if pos != len(stream):
        raise ValueError("Trailing bytes in coefficient stream")
    ordered = np.zeros_like(blocks)
    ordered[:, _ZIGZAG] = blocks
    return ordered.reshape(-1, 8, 8)


def _dct_encode_plane(plane: np.ndarray, qtable: np.ndarray, backend: str = "cpu") -> bytes:
    h, w = plane.shape
    ph, pw = ((h + 7) // 8) * 8, ((w + 7) // 8) * 8
    if backend == "cuda":
        cp = _get_cupy(required=True)
        source = cp.asarray(plane, dtype=cp.float64)
        padded = cp.pad(source, ((0, ph-h), (0, pw-w)), mode="edge") - 128.0
        blocks = padded.reshape(ph//8, 8, pw//8, 8).transpose(0, 2, 1, 3)
        dct = cp.asarray(_DCT)
        coeff = cp.einsum("ui,...ij,vj->...uv", dct, blocks, dct, optimize=True)
        quantized = cp.rint(coeff / cp.asarray(qtable)).astype(cp.int32)
        return _encode_coefficients(cp.asnumpy(quantized))
    padded = np.pad(plane.astype(np.float64), ((0, ph-h), (0, pw-w)), mode="edge") - 128.0
    blocks = padded.reshape(ph//8, 8, pw//8, 8).transpose(0, 2, 1, 3)
    coeff = np.einsum("ui,...ij,vj->...uv", _DCT, blocks, _DCT, optimize=True)
    quantized = np.rint(coeff / qtable).astype(np.int32)
    return _encode_coefficients(quantized)


def _dct_decode_plane(stream: bytes, width: int, height: int, qtable: np.ndarray,
                      backend: str = "cpu") -> np.ndarray:
    bx, by = (width + 7)//8, (height + 7)//8
    if backend == "cuda":
        cp = _get_cupy(required=True)
        blocks = cp.asarray(_decode_coefficients(stream, bx*by), dtype=cp.float64)
        coeff = blocks.reshape(by, bx, 8, 8) * cp.asarray(qtable)
        dct = cp.asarray(_DCT)
        pixels = cp.einsum("ui,...uv,vj->...ij", dct, coeff, dct, optimize=True) + 128.0
        padded = pixels.transpose(0, 2, 1, 3).reshape(by*8, bx*8)
        result = cp.clip(cp.rint(padded[:height, :width]), 0, 255).astype(cp.uint8)
        return cp.asnumpy(result)
    blocks = _decode_coefficients(stream, bx*by).astype(np.float64)
    coeff = blocks.reshape(by, bx, 8, 8) * qtable
    pixels = np.einsum("ui,...uv,vj->...ij", _DCT, coeff, _DCT, optimize=True) + 128.0
    padded = pixels.transpose(0, 2, 1, 3).reshape(by*8, bx*8)
    return np.clip(np.rint(padded[:height, :width]), 0, 255).astype(np.uint8)


def _length_prefixed(blob: bytes) -> bytes:
    if len(blob) > 0xFFFFFFFF:
        raise ValueError("PXL substream too large")
    return struct.pack("<I", len(blob)) + blob


def _read_length_prefixed(data: bytes, pos: int) -> tuple[bytes, int]:
    if pos + 4 > len(data):
        raise ValueError("Truncated PXL substream length")
    n = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    if n > MAX_TILE_PAYLOAD or pos + n > len(data):
        raise ValueError("Invalid PXL substream length")
    return data[pos:pos+n], pos+n


def _encode_alpha(alpha: bytes, width: int, height: int, compression_level: int = 0) -> bytes:
    del compression_level
    filtered = _filter_rows(alpha, width, height, 1)
    return _pack_entropy_stream(filtered)


def _decode_alpha(stream: bytes, width: int, height: int, version: int = VERSION) -> bytes:
    if version >= 3:
        filtered = _unpack_entropy_stream(stream)
    else:
        filtered = _zlib_decompress_limited(stream, height*(width+1), "alpha")
    return _unfilter_rows(filtered, width, height, 1)


def _encode_lossy_tile(raw: bytes, width: int, height: int, channels: int, quality: int, compression_level: int, backend: str = "cpu") -> bytes:
    if channels == 1:
        y_plane = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        streams = [_dct_encode_plane(y_plane, _quant_table(quality, False), backend)]
    else:
        if channels == 4:
            rgba = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
            rgb_bytes = np.ascontiguousarray(rgba[:, :, :3]).tobytes()
        else:
            rgb_bytes = raw
        rgb = Image.frombytes("RGB", (width, height), rgb_bytes)
        ycbcr = rgb.convert("YCbCr")
        y_raw = np.frombuffer(ycbcr.tobytes(), dtype=np.uint8).reshape(height, width, 3)
        y_plane = y_raw[:, :, 0]
        chroma_w, chroma_h = max(1, (width+1)//2), max(1, (height+1)//2)
        cb_img = Image.frombytes("L", (width, height), y_raw[:, :, 1].tobytes()).resize(
            (chroma_w, chroma_h), Image.Resampling.BOX)
        cr_img = Image.frombytes("L", (width, height), y_raw[:, :, 2].tobytes()).resize(
            (chroma_w, chroma_h), Image.Resampling.BOX)
        streams = [
            _dct_encode_plane(y_plane, _quant_table(quality, False), backend),
            _dct_encode_plane(np.asarray(cb_img, dtype=np.uint8), _quant_table(quality, True), backend),
            _dct_encode_plane(np.asarray(cr_img, dtype=np.uint8), _quant_table(quality, True), backend),
        ]
    payload = bytearray()
    del compression_level
    for stream in streams:
        payload.extend(_length_prefixed(_pack_entropy_stream(stream)))
    if channels == 4:
        alpha = raw[3::4]
        payload.extend(_length_prefixed(_encode_alpha(alpha, width, height)))
    return bytes(payload)


def _decode_lossy_tile(payload: bytes, width: int, height: int, channels: int, quality: int, backend: str = "cpu", version: int = VERSION) -> bytes:
    pos = 0
    streams = []
    stream_count = 1 if channels == 1 else 3
    for _ in range(stream_count):
        compressed, pos = _read_length_prefixed(payload, pos)
        block_count = ((width+7)//8) * ((height+7)//8) if _ == 0 else (((width+1)//2+7)//8) * (((height+1)//2+7)//8)
        if version >= 3:
            streams.append(_unpack_entropy_stream(compressed))
        else:
            streams.append(_zlib_decompress_limited(compressed, max(1024, block_count*384), "DCT coefficient"))
    if channels == 1:
        plane = _dct_decode_plane(streams[0], width, height, _quant_table(quality, False), backend)
        pixels = plane.tobytes()
    else:
        y = _dct_decode_plane(streams[0], width, height, _quant_table(quality, False), backend)
        cw, ch = max(1, (width+1)//2), max(1, (height+1)//2)
        cb = _dct_decode_plane(streams[1], cw, ch, _quant_table(quality, True), backend)
        cr = _dct_decode_plane(streams[2], cw, ch, _quant_table(quality, True), backend)
        cb_img = Image.frombytes("L", (cw, ch), cb.tobytes()).resize((width, height), Image.Resampling.BICUBIC)
        cr_img = Image.frombytes("L", (cw, ch), cr.tobytes()).resize((width, height), Image.Resampling.BICUBIC)
        y_img = Image.frombytes("L", (width, height), y.tobytes())
        rgb = Image.merge("YCbCr", (y_img, cb_img, cr_img)).convert("RGB").tobytes()
        if channels == 4:
            alpha_stream, pos = _read_length_prefixed(payload, pos)
            alpha = _decode_alpha(alpha_stream, width, height, version)
            out = bytearray(width*height*4)
            out[0::4], out[1::4], out[2::4], out[3::4] = rgb[0::3], rgb[1::3], rgb[2::3], alpha
            pixels = bytes(out)
        else:
            pixels = rgb
    if pos != len(payload):
        raise ValueError("Trailing bytes in lossy tile")
    return pixels


def _validate_header(version: int, profile: int, channels: int, width: int, height: int,
                     tile_size: int, quality: int) -> None:
    if version not in (LEGACY_V2, VERSION):
        raise ValueError(f"Unsupported PXL version {version}; this build reads versions 1, 2 and {VERSION}")
    if profile not in (PROFILE_LOSSLESS, PROFILE_LOSSY):
        raise ValueError("Unknown PXL compression profile")
    if channels not in (1, 3, 4):
        raise ValueError("Unsupported PXL channel count")
    if not (MIN_DIMENSION <= width <= MAX_DIMENSION and MIN_DIMENSION <= height <= MAX_DIMENSION):
        raise ValueError("Image dimensions outside supported safety limits")
    if width * height > MAX_PIXELS:
        raise ValueError("Image exceeds the 100-megapixel safety limit")
    if not (MIN_TILE <= tile_size <= MAX_TILE and tile_size % 8 == 0):
        raise ValueError("Invalid tile size; use a multiple of 8 between 64 and 1024")
    if profile == PROFILE_LOSSY and not (1 <= quality <= 100):
        raise ValueError("Lossy quality must be between 1 and 100")
    if profile == PROFILE_LOSSLESS and quality != 0:
        raise ValueError("Lossless profile must store quality=0")


def _tile_bytes(raw_image: bytes, full_stride: int, tx: int, ty: int,
                tw: int, th: int, channels: int) -> bytes:
    tile = bytearray(tw*th*channels)
    pos = 0
    for y in range(th):
        start = (ty+y)*full_stride + tx*channels
        n = tw*channels
        tile[pos:pos+n] = raw_image[start:start+n]
        pos += n
    return bytes(tile)


def _encode_tile_job(job):
    """Pickle-safe top-level worker for CPU process-pool tile encoding."""
    raw, tw, th, channels, profile, quality, compression_level = job
    if profile == PROFILE_LOSSLESS:
        codec, transform, payload = _encode_lossless_tile(raw, tw, th, channels, compression_level)
    else:
        lossy_payload = _encode_lossy_tile(raw, tw, th, channels, quality, compression_level)
        lossless_codec, lossless_transform, lossless_payload = _encode_lossless_tile(
            raw, tw, th, channels, compression_level)
        if len(lossless_payload) < len(lossy_payload):
            codec, transform, payload = lossless_codec, lossless_transform, lossless_payload
        else:
            codec, transform, payload = CODEC_DCT, TRANSFORM_NONE, lossy_payload
    if len(payload) > MAX_TILE_PAYLOAD:
        raise ValueError("Encoded tile exceeds PXL safety limit")
    return tw, th, codec, transform, payload, len(raw), binascii.crc32(payload) & 0xFFFFFFFF


def compress(input_path: str | os.PathLike, output_path: str | os.PathLike,
             mode: str = "lossless", quality: int = 88, tile_size: int = 256,
             progress: Progress = None, effort: str = "balanced",
             backend: str = "auto", workers: Optional[int] = None) -> dict:
    src, dst = Path(input_path), Path(output_path)
    if mode not in ("lossless", "lossy"):
        raise ValueError("mode must be 'lossless' or 'lossy'")
    effort_levels = {"fast": 1, "balanced": 6, "maximum": 9}
    if effort not in effort_levels:
        raise ValueError("effort must be fast, balanced, or maximum")
    compression_level = effort_levels[effort]
    if mode == "lossy" and not 1 <= int(quality) <= 100:
        raise ValueError("quality must be between 1 and 100")
    if tile_size < MIN_TILE or tile_size > MAX_TILE or tile_size % 8:
        raise ValueError("tile_size must be a multiple of 8 between 64 and 1024")
    backend, worker_count = _resolve_backend(backend, workers)

    with Image.open(src) as opened:
        img = _normalise_image(opened)
    width, height = img.size
    channels = len(img.getbands())
    if width * height > MAX_PIXELS:
        raise ValueError("Image exceeds the 100-megapixel safety limit")
    raw_image = img.tobytes()
    profile = PROFILE_LOSSLESS if mode == "lossless" else PROFILE_LOSSY
    stored_quality = 0 if mode == "lossless" else int(quality)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tile_count = math.ceil(width/tile_size) * math.ceil(height/tile_size)
    worker_count = min(worker_count, tile_count)
    full_stride = width * channels

    def jobs():
        for ty in range(0, height, tile_size):
            th = min(tile_size, height-ty)
            for tx in range(0, width, tile_size):
                tw = min(tile_size, width-tx)
                tile_raw = _tile_bytes(raw_image, full_stride, tx, ty, tw, th, channels)
                yield (tile_raw, tw, th, channels, profile, stored_quality, compression_level)

    def write_result(stream, result):
        tw, th, codec, transform, payload, decoded_size, crc = result
        stream.write(TILE_HEADER.pack(tw, th, codec, transform, len(payload), decoded_size, crc))
        stream.write(payload)

    completed = 0
    use_pool = backend == "cpu" and worker_count > 1 and tile_count > 1
    with dst.open("wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, profile, channels, 0,
                            width, height, tile_size, stored_quality, 0))
        if use_pool:
            # Bound in-flight jobs to keep memory predictable on large images.
            iterator = iter(jobs())
            pending = deque()
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")) as pool:
                limit = max(2, worker_count * 2)
                for _ in range(limit):
                    try:
                        pending.append(pool.submit(_encode_tile_job, next(iterator)))
                    except StopIteration:
                        break
                while pending:
                    future = pending.popleft()
                    write_result(f, future.result())
                    completed += 1
                    _report(progress, completed*100/tile_count,
                            f"Encoded tile {completed}/{tile_count} · CPU {worker_count} workers")
                    try:
                        pending.append(pool.submit(_encode_tile_job, next(iterator)))
                    except StopIteration:
                        pass
        else:
            for job in jobs():
                raw, tw, th, ch, prof, q, level = job
                if prof == PROFILE_LOSSLESS:
                    codec, transform, payload = _encode_lossless_tile(raw, tw, th, ch, level, backend)
                else:
                    lossy_payload = _encode_lossy_tile(raw, tw, th, ch, q, level, backend)
                    lossless_codec, lossless_transform, lossless_payload = _encode_lossless_tile(
                        raw, tw, th, ch, level, backend)
                    if len(lossless_payload) < len(lossy_payload):
                        codec, transform, payload = lossless_codec, lossless_transform, lossless_payload
                    else:
                        codec, transform, payload = CODEC_DCT, TRANSFORM_NONE, lossy_payload
                if len(payload) > MAX_TILE_PAYLOAD:
                    raise ValueError("Encoded tile exceeds PXL safety limit")
                write_result(f, (tw, th, codec, transform, payload, len(raw),
                                 binascii.crc32(payload) & 0xFFFFFFFF))
                completed += 1
                accelerator = "CUDA" if backend == "cuda" else "CPU 1 worker"
                _report(progress, completed*100/tile_count,
                        f"Encoded tile {completed}/{tile_count} · {accelerator}")
    return {
        "input_bytes": src.stat().st_size,
        "output_bytes": dst.stat().st_size,
        "width": width, "height": height,
        "mode": mode, "quality": stored_quality,
        "tiles": tile_count,
        "backend": backend,
        "workers": worker_count,
        "ratio": dst.stat().st_size / max(1, src.stat().st_size),
    }


def _decompress_v1_stream(f, dst: Path, progress: Progress = None) -> dict:
    f.seek(0)
    head = f.read(HEADER_V1.size)
    if len(head) != HEADER_V1.size:
        raise ValueError("Truncated PXL v1 header")
    magic, version, flags, width, height, tile_size, channels, bit_depth, reserved = HEADER_V1.unpack(head)
    if magic != MAGIC or version != 1:
        raise ValueError("Invalid PXL v1 header")
    if flags != 0 or reserved != 0 or bit_depth != 8 or channels not in (1, 3, 4):
        raise ValueError("Unsupported PXL v1 header fields")
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
        raise ValueError("PXL v1 dimensions outside supported limits")
    if width*height > MAX_PIXELS or not (1 <= tile_size <= MAX_TILE):
        raise ValueError("PXL v1 dimensions or tile size exceed safety limits")
    mode = "L" if channels == 1 else ("RGB" if channels == 3 else "RGBA")
    output = bytearray(width*height*channels)
    tiles_x = math.ceil(width/tile_size)
    tiles_y = math.ceil(height/tile_size)
    total = tiles_x*tiles_y
    full_stride = width*channels
    completed = 0
    for ty in range(0, height, tile_size):
        th = min(tile_size, height-ty)
        for tx in range(0, width, tile_size):
            tile_header = f.read(TILE_HEADER_V1.size)
            if len(tile_header) != TILE_HEADER_V1.size:
                raise ValueError("Unexpected end of PXL v1 tile header")
            raw_size, tw, th2, packed_mode, tile_reserved, comp_size, crc, rows = TILE_HEADER_V1.unpack(tile_header)
            expected_w = min(tile_size, width-tx)
            expected_raw = expected_w*th*channels
            if tw != expected_w or th2 != th or raw_size != expected_raw or rows != th or tile_reserved != 0:
                raise ValueError("Invalid PXL v1 tile geometry")
            if comp_size > MAX_TILE_PAYLOAD:
                raise ValueError("PXL v1 tile payload exceeds safety limit")
            body = f.read(comp_size)
            if len(body) != comp_size:
                raise ValueError("Unexpected end of PXL v1 tile payload")
            transform, codec = packed_mode >> 4, packed_mode & 0x0F
            if transform not in (0, 1) or codec not in (0, 1):
                raise ValueError("Unsupported PXL v1 tile mode")
            expected_filtered = th*(tw*channels+1)
            if codec == 1:
                filtered = _zlib_decompress_limited(body, expected_filtered, "PXL v1 tile")
            else:
                filtered = body
            transformed = _unfilter_rows(filtered, tw, th, channels)
            raw = _inverse_color(transformed, channels, transform)
            if len(raw) != raw_size:
                raise ValueError("PXL v1 decoded tile size mismatch")
            if (binascii.crc32(raw) & 0xFFFFFFFF) != crc:
                raise ValueError("PXL v1 tile checksum failed")
            p = 0
            for y in range(th):
                dst_pos = (ty+y)*full_stride + tx*channels
                n = tw*channels
                output[dst_pos:dst_pos+n] = raw[p:p+n]
                p += n
            completed += 1
            _report(progress, completed*100/total, f"Decoded legacy v1 tile {completed}/{total}")
    if f.read(1):
        raise ValueError("Trailing data after final PXL v1 tile")
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes(mode, (width, height), bytes(output)).save(dst)
    return {"output_bytes": dst.stat().st_size, "width": width, "height": height,
            "mode": "lossless-v1", "quality": None, "tiles": total}


def _decode_tile_job(job):
    payload, tw, th, channels, profile, codec, transform, quality, version = job
    if profile == PROFILE_LOSSLESS:
        if codec == CODEC_DCT:
            raise ValueError("Lossless profile cannot contain DCT tiles")
        raw = _decode_lossless_tile(payload, tw, th, channels, codec, transform, version)
    elif codec == CODEC_DCT:
        if transform != TRANSFORM_NONE:
            raise ValueError("Unsupported lossy DCT transform")
        raw = _decode_lossy_tile(payload, tw, th, channels, quality, version=version)
    else:
        raw = _decode_lossless_tile(payload, tw, th, channels, codec, transform, version)
    return raw


def decompress(input_path: str | os.PathLike, output_path: str | os.PathLike,
               progress: Progress = None, backend: str = "auto",
               workers: Optional[int] = None) -> dict:
    src, dst = Path(input_path), Path(output_path)
    with src.open("rb") as f:
        prefix = f.read(5)
        if len(prefix) != 5 or prefix[:4] != MAGIC:
            raise ValueError("Invalid PXL magic signature")
        if prefix[4] == 1:
            return _decompress_v1_stream(f, dst, progress)
        f.seek(0)
        head = f.read(HEADER.size)
        if len(head) != HEADER.size:
            raise ValueError("File is too small to be a PXL v2/v3 file")
        magic, version, profile, channels, flags, width, height, tile_size, quality, reserved = HEADER.unpack(head)
        if magic != MAGIC:
            raise ValueError("Invalid PXL magic signature")
        _validate_header(version, profile, channels, width, height, tile_size, quality)
        if flags != 0 or reserved != 0:
            raise ValueError("Unsupported PXL flags/reserved values")
        if backend == "auto" and profile == PROFILE_LOSSLESS:
            # No CUDA lossless decoder exists yet; Auto uses CPU processes here.
            backend = "cpu"
        if backend == "cuda" and profile == PROFILE_LOSSLESS:
            raise ValueError("CUDA decode currently accelerates lossy DCT tiles only. Use backend='auto' or 'cpu' for lossless decoding.")
        backend, worker_count = _resolve_backend(backend, workers)
        # CUDA is used for lossy DCT tiles; CPU multiprocess decoding handles
        # lossless profile tiles and any exact fallback tiles.
        mode = "L" if channels == 1 else ("RGB" if channels == 3 else "RGBA")
        output = bytearray(width*height*channels)
        tiles_x = math.ceil(width/tile_size)
        tiles_y = math.ceil(height/tile_size)
        tile_count = tiles_x*tiles_y
        worker_count = min(worker_count, tile_count)
        full_stride = width*channels

        def place_tile(raw: bytes, tx: int, ty: int, tw: int, th: int):
            if len(raw) != tw*th*channels:
                raise ValueError("Decoded PXL tile size mismatch")
            p = 0
            for y in range(th):
                dst_pos = (ty+y)*full_stride + tx*channels
                n = tw*channels
                output[dst_pos:dst_pos+n] = raw[p:p+n]
                p += n

        use_pool = backend == "cpu" and worker_count > 1 and tile_count > 1
        completed = 0
        if use_pool:
            pending = deque()
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")) as pool:
                limit = max(2, worker_count*2)
                for ty in range(0, height, tile_size):
                    th = min(tile_size, height-ty)
                    for tx in range(0, width, tile_size):
                        tw = min(tile_size, width-tx)
                        header = f.read(TILE_HEADER.size)
                        if len(header) != TILE_HEADER.size:
                            raise ValueError("Unexpected end of PXL file while reading tile header")
                        tw2, th2, codec, transform, payload_size, decoded_size, crc = TILE_HEADER.unpack(header)
                        if tw2 != tw or th2 != th or decoded_size != tw*th*channels:
                            raise ValueError("Invalid PXL tile geometry")
                        if payload_size > MAX_TILE_PAYLOAD:
                            raise ValueError("PXL tile payload exceeds safety limit")
                        payload = f.read(payload_size)
                        if len(payload) != payload_size:
                            raise ValueError("Unexpected end of PXL tile payload")
                        if (binascii.crc32(payload) & 0xFFFFFFFF) != crc:
                            raise ValueError("PXL tile payload checksum failed")
                        job = (payload, tw, th, channels, profile, codec, transform, quality, version)
                        pending.append((pool.submit(_decode_tile_job, job), tx, ty, tw, th))
                        if len(pending) >= limit:
                            future, px, py, pw, ph = pending.popleft()
                            place_tile(future.result(), px, py, pw, ph)
                            completed += 1
                            _report(progress, completed*100/tile_count,
                                    f"Decoded tile {completed}/{tile_count} · CPU {worker_count} workers")
                while pending:
                    future, px, py, pw, ph = pending.popleft()
                    place_tile(future.result(), px, py, pw, ph)
                    completed += 1
                    _report(progress, completed*100/tile_count,
                            f"Decoded tile {completed}/{tile_count} · CPU {worker_count} workers")
        else:
            for ty in range(0, height, tile_size):
                th = min(tile_size, height-ty)
                for tx in range(0, width, tile_size):
                    tw = min(tile_size, width-tx)
                    header = f.read(TILE_HEADER.size)
                    if len(header) != TILE_HEADER.size:
                        raise ValueError("Unexpected end of PXL file while reading tile header")
                    tw2, th2, codec, transform, payload_size, decoded_size, crc = TILE_HEADER.unpack(header)
                    if tw2 != tw or th2 != th or decoded_size != tw*th*channels:
                        raise ValueError("Invalid PXL tile geometry")
                    if payload_size > MAX_TILE_PAYLOAD:
                        raise ValueError("PXL tile payload exceeds safety limit")
                    payload = f.read(payload_size)
                    if len(payload) != payload_size:
                        raise ValueError("Unexpected end of PXL tile payload")
                    if (binascii.crc32(payload) & 0xFFFFFFFF) != crc:
                        raise ValueError("PXL tile payload checksum failed")
                    if profile == PROFILE_LOSSLESS:
                        if codec == CODEC_DCT:
                            raise ValueError("Lossless profile cannot contain DCT tiles")
                        raw = _decode_lossless_tile(payload, tw, th, channels, codec, transform, version)
                    elif codec == CODEC_DCT:
                        if transform != TRANSFORM_NONE:
                            raise ValueError("Unsupported lossy DCT transform")
                        raw = _decode_lossy_tile(payload, tw, th, channels, quality, backend, version)
                    else:
                        raw = _decode_lossless_tile(payload, tw, th, channels, codec, transform, version)
                    place_tile(raw, tx, ty, tw, th)
                    completed += 1
                    label = "CUDA DCT" if backend == "cuda" else "CPU 1 worker"
                    _report(progress, completed*100/tile_count,
                            f"Decoded tile {completed}/{tile_count} · {label}")
        if f.read(1):
            raise ValueError("Trailing data after final PXL tile")
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes(mode, (width, height), bytes(output)).save(dst)
    return {"output_bytes": dst.stat().st_size, "width": width, "height": height,
            "mode": "lossless" if profile == PROFILE_LOSSLESS else "lossy",
            "quality": quality, "tiles": tile_count, "backend": backend,
            "workers": worker_count}


def inspect(path: str | os.PathLike) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        prefix = f.read(5)
        if len(prefix) != 5 or prefix[:4] != MAGIC:
            raise ValueError("Invalid PXL magic signature")
        f.seek(0)
        if prefix[4] == 1:
            head = f.read(HEADER_V1.size)
            if len(head) != HEADER_V1.size:
                raise ValueError("Truncated PXL v1 header")
            magic, version, flags, width, height, tile_size, channels, bit_depth, reserved = HEADER_V1.unpack(head)
            if flags != 0 or reserved != 0 or bit_depth != 8 or channels not in (1, 3, 4):
                raise ValueError("Unsupported PXL v1 header fields")
            if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION) or width*height > MAX_PIXELS or not (1 <= tile_size <= MAX_TILE):
                raise ValueError("Invalid PXL v1 dimensions or tile size")
            return {"version": 1, "profile": "lossless (legacy)",
                    "dimensions": f"{width} x {height}", "channels": channels,
                    "tile_size": tile_size, "quality": None,
                    "file_bytes": path.stat().st_size,
                    "tile_count": math.ceil(width/tile_size)*math.ceil(height/tile_size)}
        head = f.read(HEADER.size)
    if len(head) != HEADER.size:
        raise ValueError("Not a valid PXL file")
    magic, version, profile, channels, flags, width, height, tile_size, quality, reserved = HEADER.unpack(head)
    if magic != MAGIC:
        raise ValueError("Invalid PXL magic signature")
    _validate_header(version, profile, channels, width, height, tile_size, quality)
    if flags != 0 or reserved != 0:
        raise ValueError("Unsupported PXL flags/reserved values")
    return {"version": version, "profile": "lossless" if profile == PROFILE_LOSSLESS else "lossy",
            "dimensions": f"{width} x {height}", "channels": channels, "tile_size": tile_size,
            "quality": quality if profile == PROFILE_LOSSY else None,
            "file_bytes": path.stat().st_size,
            "tile_count": math.ceil(width/tile_size)*math.ceil(height/tile_size)}


def benchmark(path: str | os.PathLike, mode: str = "lossless", quality: int = 88,
              effort: str = "balanced", backend: str = "auto",
              workers: Optional[int] = None) -> dict:
    import tempfile
    src = Path(path)
    with tempfile.TemporaryDirectory() as td:
        packed, restored = Path(td)/"benchmark.pxl", Path(td)/"restored.png"
        t0 = time.perf_counter()
        c = compress(src, packed, mode=mode, quality=quality, effort=effort, backend=backend, workers=workers)
        encode_s = time.perf_counter()-t0
        t1 = time.perf_counter()
        d = decompress(packed, restored, backend=backend, workers=workers)
        decode_s = time.perf_counter()-t1
        result = {**c, "encode_seconds": encode_s, "decode_seconds": decode_s,
                  "restored_bytes": d["output_bytes"]}
        if mode == "lossless":
            with Image.open(src) as a, Image.open(restored) as b:
                aa, bb = _normalise_image(a), _normalise_image(b)
                result["pixel_exact"] = aa.tobytes() == bb.tobytes()
        return result



# PXL v8 public API. Readers for all previous container versions remain available.
_compress_v3 = compress
_decompress_legacy = decompress
_inspect_legacy = inspect

def compress(input_path, output_path, mode="lossless", quality=88, tile_size=256,
             progress=None, effort="balanced", backend="auto", workers=None,
             adaptive_tiles=True, checkpoint_interval=8):
    from pxl4_engine import compress as _compress_v8
    return _compress_v8(input_path, output_path, mode=mode, quality=quality,
                        tile_size=tile_size, progress=progress, effort=effort,
                        backend=backend, workers=workers, adaptive_tiles=adaptive_tiles,
                        checkpoint_interval=checkpoint_interval)

def decompress(input_path, output_path, progress=None, backend="auto", workers=None):
    path = Path(input_path)
    with path.open("rb") as f:
        prefix = f.read(5)
    if len(prefix) >= 5 and prefix[:4] == MAGIC and prefix[4] in (7, 8):
        with path.open("rb") as f:
            header = f.read(HEADER.size)
        if prefix[4] == 7 and len(header) == HEADER.size and (HEADER.unpack(header)[4] & 0x03) in (1, 2, 3):
            from pxl7_hdr import decompress as _decompress_v7_hdr
            return _decompress_v7_hdr(input_path, output_path, progress=progress,
                                      backend=backend, workers=workers)
        from pxl4_engine import decompress as _decompress_v7
        return _decompress_v7(input_path, output_path, progress=progress,
                              backend=backend, workers=workers)
    if len(prefix) >= 5 and prefix[:4] == MAGIC and prefix[4] in (4, 5, 6):
        from pxl4_engine import decompress as _decompress_v6
        return _decompress_v6(input_path, output_path, progress=progress,
                              backend=backend, workers=workers)
    return _decompress_legacy(input_path, output_path, progress=progress,
                              backend=backend, workers=workers)

def inspect(path):
    file_path = Path(path)
    with file_path.open("rb") as f:
        prefix = f.read(5)
    if len(prefix) >= 5 and prefix[:4] == MAGIC and prefix[4] in (7, 8):
        with file_path.open("rb") as f:
            header = f.read(HEADER.size)
        if prefix[4] == 7 and len(header) == HEADER.size and (HEADER.unpack(header)[4] & 0x03) in (1, 2, 3):
            from pxl7_hdr import inspect as _inspect_v7_hdr
            return _inspect_v7_hdr(path)
        from pxl4_engine import inspect as _inspect_v7
        return _inspect_v7(path)
    if len(prefix) >= 5 and prefix[:4] == MAGIC and prefix[4] in (4, 5, 6):
        from pxl4_engine import inspect as _inspect_v6
        return _inspect_v6(path)
    return _inspect_legacy(path)


def _cli() -> int:
    parser = argparse.ArgumentParser(prog="pxl", description="PXL v8 C++-accelerated lossless/HDR image compressor")
    sub = parser.add_subparsers(dest="command", required=True)
    pc = sub.add_parser("compress", help="Compress an image into .pxl")
    pc.add_argument("input")
    pc.add_argument("output")
    pc.add_argument("--mode", choices=("lossless", "lossy"), default="lossless")
    pc.add_argument("--quality", type=int, default=88)
    pc.add_argument("--tile", type=int, default=256)
    pc.add_argument("--effort", choices=("fast", "balanced", "maximum"), default="balanced")
    pc.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    pc.add_argument("--workers", type=int, default=0, help="CPU workers; 0 means auto")
    pc.add_argument("--fixed-tiles", action="store_true", help="Disable adaptive tile-boundary segmentation")
    pc.add_argument("--checkpoint-interval", type=int, default=8,
                    help="Tiles per independent decode-history group (lower values increase parallelism and overhead)")
    pd = sub.add_parser("decompress", help="Decode a .pxl file")
    pd.add_argument("input")
    pd.add_argument("output")
    pd.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    pd.add_argument("--workers", type=int, default=0, help="CPU workers; 0 means auto")
    pi = sub.add_parser("info", help="Show PXL metadata")
    pi.add_argument("input")
    pb = sub.add_parser("benchmark", help="Encode/decode timing and size report")
    pb.add_argument("input")
    pb.add_argument("--mode", choices=("lossless", "lossy"), default="lossless")
    pb.add_argument("--quality", type=int, default=88)
    pb.add_argument("--effort", choices=("fast", "balanced", "maximum"), default="balanced")
    pb.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    pb.add_argument("--workers", type=int, default=0, help="CPU workers; 0 means auto")
    args = parser.parse_args()
    try:
        if args.command == "compress":
            result = compress(args.input, args.output, args.mode, args.quality, args.tile,
                              effort=args.effort, backend=args.backend, workers=args.workers,
                              adaptive_tiles=not args.fixed_tiles,
                              checkpoint_interval=args.checkpoint_interval)
        elif args.command == "decompress":
            result = decompress(args.input, args.output, backend=args.backend, workers=args.workers)
        elif args.command == "info":
            result = inspect(args.input)
        else:
            result = benchmark(args.input, args.mode, args.quality, args.effort, args.backend, args.workers)
        for key, value in result.items():
            print(f"{key.replace('_', ' ').title():22}: {value}")
        return 0
    except Exception as exc:
        print(f"PXL error: {exc}", file=sys.stderr)
        return 2


def cli_entry() -> None:
    raise SystemExit(_cli())


if __name__ == "__main__":
    raise SystemExit(_cli())
