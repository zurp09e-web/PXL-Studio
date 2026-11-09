#!/usr/bin/env python3
"""Build the optional PXL C++17 acceleration library using a local compiler."""
from __future__ import annotations
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "native" / "pxl_native.cpp"
SRC_V8 = ROOT / "native" / "pxl_v8.cpp"
SYSTEM = platform.system()

if SYSTEM == "Windows":
    output = ROOT / "pxl_native.dll"
    candidates = ["cl", "g++"]
    compiler = next((shutil.which(x) for x in candidates if shutil.which(x)), None)
    if compiler is None:
        raise SystemExit("No C++ compiler found. Install Visual Studio Build Tools (C++ workload) or MinGW-w64.")
    if Path(compiler).name.lower() == "cl.exe":
        # Keep ISA-specific code in separate translation units. The baseline
        # dispatcher remains safe on older CPUs and calls only supported kernels.
        build_dir = ROOT / "native" / "build_msvc"
        build_dir.mkdir(parents=True, exist_ok=True)
        objects = []
        defines = []
        for isa, source, macro in (
            ("AVX2", ROOT / "native" / "pxl_v8_msvc_avx2.cpp", "PXL_HAS_MSVC_AVX2"),
            ("AVX512", ROOT / "native" / "pxl_v8_msvc_avx512.cpp", "PXL_HAS_MSVC_AVX512"),
        ):
            obj = build_dir / f"pxl_v8_msvc_{isa.lower()}.obj"
            # Never let an object from an older build survive a failed compiler probe.
            obj.unlink(missing_ok=True)
            probe = [compiler, "/nologo", "/O2", "/std:c++17", "/EHsc", "/MD",
                     f"/arch:{isa}", "/c", str(source), f"/Fo:{obj}"]
            result = subprocess.run(probe, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            if result.returncode == 0 and obj.exists():
                objects.append(str(obj))
                defines.append(f"/D{macro}")
                print(f"Enabled MSVC {isa} SIMD kernel")
            else:
                print(f"MSVC {isa} kernel unavailable; using lower ISA/scalar fallback")
                if result.stdout:
                    print(result.stdout.strip())
        cmd = [compiler, "/nologo", "/O2", "/std:c++17", "/EHsc", "/MD", "/LD",
               *defines, str(SRC), str(SRC_V8), *objects, f"/Fe:{output}"]
    else:
        cmd = [compiler, "-O3", "-std=c++17", "-shared", str(SRC), str(SRC_V8), "-o", str(output)]
elif SYSTEM == "Darwin":
    output = ROOT / "libpxl_native.dylib"
    compiler = shutil.which("clang++") or shutil.which("c++")
    if not compiler:
        raise SystemExit("Install Xcode Command Line Tools to build the PXL C++ backend.")
    cmd = [compiler, "-O3", "-std=c++17", "-dynamiclib", str(SRC), str(SRC_V8), "-o", str(output)]
else:
    output = ROOT / "libpxl_native.so"
    compiler = shutil.which("g++") or shutil.which("c++")
    if not compiler:
        raise SystemExit("Install g++ or another C++17 compiler to build the PXL native backend.")
    cmd = [compiler, "-O3", "-std=c++17", "-fPIC", "-shared", "-pthread", str(SRC), str(SRC_V8), "-o", str(output)]

print("Building PXL C++ CPU backend:", " ".join(cmd))
print("Note: this builds the native CPU/SIMD library only. CUDA is supplied separately by CuPy; run diagnose_cuda.py to test it.")
subprocess.run(cmd, cwd=ROOT, check=True)
print("Built:", output)
