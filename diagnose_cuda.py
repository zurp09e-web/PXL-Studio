#!/usr/bin/env python3
"""Diagnose PXL's optional CUDA predictor path on the current Python install."""
from __future__ import annotations

import os
import platform
from importlib import metadata
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from pxl import cuda_status, compress, decompress
from pxl_native import backend_info


def main() -> int:
    print("PXL CUDA diagnostic")
    print("=" * 64)
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', '<not set>')}")

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                timeout=10, check=False)
        print("\nnvidia-smi:")
        print((result.stdout or result.stderr or "nvidia-smi returned no output").strip()[:5000])
        if result.returncode:
            print(f"nvidia-smi exit code: {result.returncode}")
    except Exception as exc:
        print(f"\nnvidia-smi: unavailable ({type(exc).__name__}: {exc})")

    native = backend_info(refresh=True)
    print("\nNative CPU backend:")
    for key in ("available", "implementation", "library", "simd", "abi", "error"):
        print(f"  {key}: {native.get(key)}")

    # CuPy's CUDA-specific wheels intentionally share the ``cupy`` import
    # namespace. Installing the generic ``cupy`` wheel alongside a CUDA wheel
    # is unsupported and can cause import-time warnings or mixed files.
    cupy_distributions = {}
    for distribution in metadata.distributions():
        name = (distribution.metadata.get("Name") or "").strip().lower().replace("_", "-")
        if name in {"cupy", "cupy-cuda12x", "cupy-cuda13x"}:
            cupy_distributions[name] = distribution.version
    print("\nInstalled CuPy distributions:")
    if cupy_distributions:
        for name, version in sorted(cupy_distributions.items()):
            print(f"  {name}: {version}")
    else:
        print("  none found")
    if len(cupy_distributions) > 1:
        print("  WARNING: Multiple CuPy distributions are installed.")
        print("  Clean them before further testing:")
        print("    py -m pip uninstall -y cupy cupy-cuda12x cupy-cuda13x")
        print("  Then install exactly one matching requirement file.")

    gpu = cuda_status(refresh=True)
    print("\nCuPy / CUDA probe:")
    for key, value in gpu.items():
        print(f"  {key}: {value}")
    if not gpu.get("available"):
        print("\nCUDA test not run. The native build script does not install CUDA; install a matching CuPy wheel and run this diagnostic again.")
        return 2

    # Smooth colour fields force PXL's predictor candidate path instead of the
    # raw/noise fast path, making this an actual encoder-kernel smoke test.
    h, w = 128, 160
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack(((xx * 3 + yy) & 255,
                    (xx + yy * 2) & 255,
                    (xx * 2 + yy * 3) & 255), axis=2).astype(np.uint8)
    with tempfile.TemporaryDirectory(prefix="pxl_cuda_diagnostic_") as td:
        td = Path(td)
        src, packed, restored = td / "source.png", td / "test.pxl", td / "restored.png"
        Image.fromarray(arr, "RGB").save(src)
        print("\nRunning real PXL encode with backend='cuda'...")
        try:
            encoded = compress(src, packed, mode="lossless", tile_size=64,
                               effort="balanced", backend="cuda", workers=1)
            print(f"  encode result: {encoded}")
            print("Running PXL decode on CPU (decoder is CPU-side)...")
            decoded = decompress(packed, restored, backend="cpu", workers=1)
            print(f"  decode result: {decoded}")
            with Image.open(restored) as image:
                exact = image.convert("RGB").tobytes() == Image.open(src).convert("RGB").tobytes()
            print(f"  exact reconstruction: {exact}")
            if not exact:
                print("FAIL: pixels did not round-trip exactly")
                return 3
            if int(encoded.get("cuda_tiles_used", 0)) < 1:
                print("WARNING: encode completed but did not invoke a CUDA predictor tile; inspect image/fast-path selection.")
                return 4
        except Exception as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}")
            return 5

    print("\nPASS: CuPy initialized, the PXL CUDA predictor ran, and pixels reconstructed exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
