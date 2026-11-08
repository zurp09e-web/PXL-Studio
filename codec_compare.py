#!/usr/bin/env python3
"""Disk-backed, reproducible image codec comparison for PXL Studio.

Timings include encoding/decoding and writing the encoded/decoded files to a
local temporary directory for every codec. Codec options are documented in
notes because numeric quality values are not calibrated across formats.
"""
from __future__ import annotations

import csv
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

# Optional JPEG XL plugin; importing registers JXL with Pillow.
try:
    import pillow_jxl  # noqa: F401
    JXL_AVAILABLE = True
except Exception:
    JXL_AVAILABLE = False

from pxl import compress as pxl_compress, decompress as pxl_decompress, _normalise_image, cuda_status

Progress = Optional[Callable[[int, str], None]]


def _progress(cb: Progress, percent: int, message: str) -> None:
    if cb:
        cb(max(0, min(100, int(percent))), message)


def _ssim_8x8(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean channel-wise SSIM over non-overlapping 8x8 windows."""
    a = reference.astype(np.float64)
    b = candidate.astype(np.float64)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    scores: list[float] = []
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    h, w, _ = a.shape
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            aa = a[y:y+8, x:x+8, :]
            bb = b[y:y+8, x:x+8, :]
            if aa.size == 0:
                continue
            axes = (0, 1)
            ma, mb = aa.mean(axis=axes), bb.mean(axis=axes)
            va, vb = aa.var(axis=axes), bb.var(axis=axes)
            cov = ((aa - ma) * (bb - mb)).mean(axis=axes)
            value = ((2*ma*mb+c1) * (2*cov+c2)) / ((ma*ma+mb*mb+c1) * (va+vb+c2))
            scores.extend(value.tolist())
    return float(np.mean(scores)) if scores else 1.0


def _quality_metrics(reference: Image.Image, decoded: Image.Image) -> tuple[bool, float, float]:
    a = np.asarray(reference, dtype=np.uint8)
    b = np.asarray(decoded, dtype=np.uint8)
    exact = a.shape == b.shape and np.array_equal(a, b)
    if a.shape != b.shape:
        raise ValueError(f"Decoded shape mismatch: {b.shape} vs {a.shape}")
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    psnr = math.inf if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    return exact, psnr, _ssim_8x8(a, b)


def _candidates(mode: str, quality: int, effort: str):
    """Yield (name, file extension, save callback, target mode, note)."""
    method = {"fast": 0, "balanced": 4, "maximum": 6}[effort]
    if mode == "lossless":
        yield ("PXL lossless", ".pxl", "pxl", None,
               f"PXL v8.0; adaptive tiles, cross-tile references and spatial entropy; tile=256; effort={effort}")
        yield ("PNG", ".png", {"format": "PNG", "compress_level": {"fast": 1, "balanced": 6, "maximum": 9}[effort]}, None,
               "DEFLATE; exact lossless")
        yield ("WebP lossless", ".webp", {"format": "WEBP", "lossless": True, "method": method}, None,
               "WebP lossless; exact comparison includes every RGBA channel, including hidden RGB under transparent pixels")
        if JXL_AVAILABLE:
            yield ("JPEG XL lossless", ".jxl", {"format": "JXL", "lossless": True, "effort": {"fast": 3, "balanced": 6, "maximum": 9}[effort]}, None,
                   "JPEG XL lossless via pillow-jxl-plugin; exact pixel comparison")
        yield ("TIFF Deflate", ".tif", {"format": "TIFF", "compression": "tiff_adobe_deflate"}, None,
               "TIFF with Adobe Deflate; exact lossless")
        yield ("TIFF LZW", ".tif", {"format": "TIFF", "compression": "tiff_lzw"}, None,
               "TIFF with LZW; exact lossless")
        yield ("QOI", ".qoi", {"format": "QOI"}, "RGB",
               "Quite OK Image; grayscale input is expanded to RGB")
        yield ("BMP", ".bmp", {"format": "BMP"}, "RGB",
               "Uncompressed bitmap baseline; alpha/grayscale normalized to RGB")
    else:
        yield ("PXL lossy", ".pxl", "pxl", None,
               f"PXL v8.0 container with DCT tiles; quality={quality}; tile=256; effort={effort}")
        yield ("JPEG", ".jpg", {"format": "JPEG", "quality": quality, "optimize": True, "progressive": True}, "RGB",
               "JPEG; alpha is discarded; input converted to RGB")
        yield ("WebP lossy", ".webp", {"format": "WEBP", "quality": quality, "method": method}, "RGB",
               "WebP lossy; alpha-capable input is kept when source has alpha")
        if JXL_AVAILABLE:
            yield ("JPEG XL lossy", ".jxl", {"format": "JXL", "quality": quality, "effort": {"fast": 3, "balanced": 6, "maximum": 9}[effort], "lossless_jpeg": False}, "RGB",
                   "JPEG XL lossy via pillow-jxl-plugin; quality scale is codec-specific")
        yield ("AVIF lossy", ".avif", {"format": "AVIF", "quality": quality, "speed": max(0, 8-method)}, "RGB",
               "AVIF; optional Pillow AVIF plugin; alpha-capable input is kept when source has alpha")


def compare_codecs(input_path: str | os.PathLike, mode: str = "lossless", quality: int = 88,
                   effort: str = "balanced", progress: Progress = None,
                   backend: str = "auto", workers: int | None = None) -> list[dict]:
    """Compare installed codecs using the same normalized source and temp disk.

    Unavailable encoders are returned as skipped rows, not silently omitted.
    Lossy results include PSNR and an 8x8-window SSIM estimate. These are useful
    indicators, not substitutes for visual inspection.
    """
    if mode not in ("lossless", "lossy"):
        raise ValueError("mode must be lossless or lossy")
    if not 1 <= int(quality) <= 100:
        raise ValueError("quality must be between 1 and 100")
    if effort not in ("fast", "balanced", "maximum"):
        raise ValueError("effort must be fast, balanced, or maximum")

    with Image.open(input_path) as opened:
        opened.seek(0)
        source = _normalise_image(opened)
    raw_pixels = source.tobytes()
    source_bytes = Path(input_path).stat().st_size
    raw_bytes = len(raw_pixels)
    candidates = list(_candidates(mode, int(quality), effort))
    results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="pxl_codec_bench_") as temp_dir:
        temp = Path(temp_dir)
        # Stage one shared normalized image outside the timed sections. Every
        # encoder then reads that same PNG from disk, including PXL's path API.
        staged_source = temp / "benchmark_source.png"
        source.save(staged_source, format="PNG", compress_level=1)
        for index, (name, extension, options, forced_mode, note) in enumerate(candidates):
            _progress(progress, round(index * 100 / len(candidates)), f"Testing {name} ({index+1}/{len(candidates)})")
            encoded_path = temp / f"encoded_{index}{extension}"
            decoded_path = temp / f"decoded_{index}.png"
            target_mode = forced_mode or source.mode
            # WebP/AVIF encoders accept RGB(A); expand grayscale, preserve RGBA.
            if name in ("WebP lossless", "WebP lossy", "AVIF lossy", "JPEG XL lossy"):
                target_mode = "RGBA" if source.mode == "RGBA" else "RGB"
            reference = source if target_mode == source.mode else source.convert(target_mode)
            row = {
                "codec": name, "status": "OK", "size_bytes": 0, "raw_bytes": raw_bytes,
                "source_bytes": source_bytes, "ratio_raw": 0.0, "encode_ms": 0.0,
                "decode_ms": 0.0, "exact": False, "psnr_db": None, "ssim": None,
                "note": note, "target_mode": target_mode,
                "backend_requested": backend if options == "pxl" else "n/a",
                "backend_used": "n/a",
            }
            try:
                encode_start = time.perf_counter()
                if options == "pxl":
                    # Codec Lab measures the codec, not whether an optional GPU
                    # stack happens to initialize. Try CUDA when requested, but
                    # always retain a CPU route so PXL itself never disappears
                    # from the comparison because of a CuPy/driver problem.
                    pxl_backend = backend
                    cuda_fallback = None
                    if backend in ("auto", "cuda"):
                        gpu_status = cuda_status()
                        if gpu_status.get("available"):
                            pxl_backend = "cuda"
                        else:
                            pxl_backend = "cpu"
                            if backend == "cuda":
                                cuda_fallback = gpu_status.get("reason", "CUDA smoke test failed")
                    try:
                        pxl_compress(staged_source, encoded_path, mode=mode, quality=quality,
                                     tile_size=256, effort=effort, backend=pxl_backend, workers=workers)
                    except Exception as gpu_exc:
                        if pxl_backend != "cuda":
                            raise
                        encoded_path.unlink(missing_ok=True)
                        pxl_backend = "cpu"
                        cuda_fallback = f"{type(gpu_exc).__name__}: {gpu_exc}"
                        pxl_compress(staged_source, encoded_path, mode=mode, quality=quality,
                                     tile_size=256, effort=effort, backend="cpu", workers=workers)
                    row["backend_used"] = pxl_backend
                    if cuda_fallback:
                        row["note"] += f"; CUDA failed/unavailable, benchmark used CPU fallback: {cuda_fallback}"
                else:
                    save_options = dict(options)
                    fmt = save_options.pop("format")
                    with Image.open(staged_source) as staged:
                        prepared = staged.convert(target_mode)
                        prepared.save(encoded_path, format=fmt, **save_options)
                row["encode_ms"] = (time.perf_counter() - encode_start) * 1000
                row["size_bytes"] = encoded_path.stat().st_size
                row["ratio_raw"] = row["size_bytes"] / max(1, raw_bytes)

                decode_start = time.perf_counter()
                if options == "pxl":
                    # PXL v8 reconstruction is CPU-side; requesting CUDA here
                    # would not accelerate this path and can cause a false
                    # Unavailable result on machines with an encoder-only GPU.
                    pxl_decompress(encoded_path, decoded_path, backend="cpu", workers=workers)
                    with Image.open(decoded_path) as out:
                        out.load()
                        decoded = out.convert(target_mode)
                else:
                    with Image.open(encoded_path) as out:
                        out.load()
                        decoded = out.convert(target_mode)
                        decoded.save(decoded_path, format="PNG", compress_level=1)
                row["decode_ms"] = (time.perf_counter() - decode_start) * 1000
                exact, psnr, ssim = _quality_metrics(reference, decoded)
                row["exact"] = exact
                row["psnr_db"] = psnr
                row["ssim"] = ssim
                if mode == "lossless" and not exact:
                    row["status"] = "CHECK"
                    row["note"] += "; decoded pixels differ from normalized input (not byte-exact for this sample)"
                if target_mode != source.mode:
                    row["note"] += f"; compared after {source.mode} → {target_mode} conversion"
            except Exception as exc:
                row["status"] = "Unavailable"
                row["note"] = f"Encoder/decoder unavailable for this configuration: {exc}"
            results.append(row)

    results.sort(key=lambda r: (r["status"] != "OK", r["size_bytes"] if r["size_bytes"] else 10**30))
    _progress(progress, 100, "Codec comparison complete")
    return results


def export_csv(results: list[dict], path: str | os.PathLike) -> None:
    fields = ["codec", "status", "size_bytes", "raw_bytes", "source_bytes", "ratio_raw",
              "encode_ms", "decode_ms", "exact", "psnr_db", "ssim", "target_mode",
              "backend_requested", "backend_used", "note"]
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
