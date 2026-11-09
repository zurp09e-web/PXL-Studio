"""PXL's C++17 codec backend, exposed through a stable C ABI and Python ctypes.

The Python application remains the GUI, I/O and orchestration layer. When the
native library is built, hot predictor, dictionary, rANS and inverse-filter
paths are executed in C++. The Python/Numba implementation remains a portable
fallback for systems without a C++ build toolchain.
"""
from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NATIVE_DIR = ROOT / "native"


def _candidate_libraries():
    names = {
        "Windows": ("pxl_native.dll",),
        "Darwin": ("libpxl_native.dylib", "pxl_native.dylib"),
    }.get(platform.system(), ("libpxl_native.so",))
    for name in names:
        yield ROOT / name
        yield NATIVE_DIR / name
        # CMake's multi-configuration generators (Visual Studio) put DLLs in
        # a configuration subdirectory, unlike build_native.py's root output.
        for base in (NATIVE_DIR / "build", ROOT / "build"):
            yield base / name
            yield base / "Release" / name
            yield base / "Debug" / name
            yield base / "RelWithDebInfo" / name


def _void_ptr_from_bytes(data: bytes):
    # c_char_p references the immutable bytes object without copying its data.
    return ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p)


def _ptr(array: np.ndarray):
    return ctypes.c_void_p(int(array.ctypes.data))


class NativeBackend:
    def __init__(self, path: Path):
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.pxl_native_abi_version.restype = ctypes.c_int
        abi = self.lib.pxl_native_abi_version()
        if abi != 5:
            raise RuntimeError(f"Unsupported PXL native ABI {abi}; rebuild the native library")
        self._bind()

    def _bind(self):
        cvoid = ctypes.c_void_p
        u32 = ctypes.c_uint32
        i32 = ctypes.c_int
        self.lib.pxl_rans_encode_ids.argtypes = [cvoid, cvoid, u32, cvoid, cvoid, cvoid, u32,
                                                  ctypes.POINTER(u32), ctypes.POINTER(u32)]
        self.lib.pxl_rans_encode_ids.restype = i32
        self.lib.pxl_rans_encode_context16.argtypes = [cvoid, u32, cvoid, cvoid, cvoid, u32,
                                                        ctypes.POINTER(u32), ctypes.POINTER(u32)]
        self.lib.pxl_rans_encode_context16.restype = i32
        self.lib.pxl_rans_decode_context16.argtypes = [cvoid, u32, u32, u32, cvoid, cvoid, cvoid,
                                                        cvoid, ctypes.POINTER(u32)]
        self.lib.pxl_rans_decode_context16.restype = i32
        self.lib.pxl_rans_decode_context2d.argtypes = [cvoid, u32, u32, u32, u32, u32, u32,
                                                       cvoid, cvoid, cvoid, cvoid,
                                                       ctypes.POINTER(u32)]
        self.lib.pxl_rans_decode_context2d.restype = i32
        self.lib.pxl_rans_decode_context2d_signed.argtypes = [cvoid, u32, u32, u32, u32, u32, u32,
                                                              cvoid, cvoid, cvoid, cvoid,
                                                              ctypes.POINTER(u32)]
        self.lib.pxl_rans_decode_context2d_signed.restype = i32
        self.lib.pxl_rans_decode_context2d_triple.argtypes = [cvoid, u32, u32, u32, u32, u32, u32,
                                                              cvoid, cvoid, cvoid, cvoid,
                                                              ctypes.POINTER(u32)]
        self.lib.pxl_rans_decode_context2d_triple.restype = i32
        self.lib.pxl_block_state_create.argtypes = [u32, u32]
        self.lib.pxl_block_state_create.restype = cvoid
        self.lib.pxl_block_state_clone.argtypes = [cvoid]
        self.lib.pxl_block_state_clone.restype = cvoid
        self.lib.pxl_block_state_destroy.argtypes = [cvoid]
        self.lib.pxl_block_state_destroy.restype = None
        self.lib.pxl_block_state_encode.argtypes = [cvoid, cvoid, u32, u32, cvoid, u32, ctypes.POINTER(u32)]
        self.lib.pxl_block_state_encode.restype = i32
        self.lib.pxl_block_state_update.argtypes = [cvoid, cvoid, u32, u32]
        self.lib.pxl_block_state_update.restype = i32
        self.lib.pxl_block_state_decode.argtypes = [cvoid, cvoid, u32, u32, u32, cvoid, u32]
        self.lib.pxl_block_state_decode.restype = i32
        self.lib.pxl_pixel_lz_state_create.argtypes = [u32, u32]
        self.lib.pxl_pixel_lz_state_create.restype = cvoid
        self.lib.pxl_pixel_lz_state_destroy.argtypes = [cvoid]
        self.lib.pxl_pixel_lz_state_destroy.restype = None
        self.lib.pxl_pixel_lz_state_seed.argtypes = [cvoid, cvoid, u32, ctypes.c_uint64]
        self.lib.pxl_pixel_lz_state_seed.restype = i32
        self.lib.pxl_pixel_lz_state_update.argtypes = [cvoid, cvoid, u32]
        self.lib.pxl_pixel_lz_state_update.restype = i32
        self.lib.pxl_pixel_lz_state_encode.argtypes = [cvoid, cvoid, u32, cvoid, u32, ctypes.POINTER(u32)]
        self.lib.pxl_pixel_lz_state_encode.restype = i32
        self.lib.pxl_pixel_lz_encode.argtypes = [cvoid, u32, cvoid, u32, u32, u32, cvoid, u32,
                                                ctypes.POINTER(u32)]
        self.lib.pxl_pixel_lz_encode.restype = i32
        self.lib.pxl_pixel_lz_decode.argtypes = [cvoid, u32, cvoid, u32, u32, u32, cvoid, u32]
        self.lib.pxl_pixel_lz_decode.restype = i32
        self.lib.pxl_filter_color_v3.argtypes = [cvoid, u32, u32, u32, u32, cvoid, u32, cvoid]
        self.lib.pxl_filter_color_v3.restype = i32
        self.lib.pxl_unfilter_color_v3.argtypes = [cvoid, u32, u32, u32, u32, cvoid, u32, u32]
        self.lib.pxl_unfilter_color_v3.restype = i32
        self.lib.pxl_filter_legacy.argtypes = [cvoid, u32, u32, u32, u32, cvoid, u32]
        self.lib.pxl_filter_legacy.restype = i32
        self.lib.pxl_unfilter_legacy.argtypes = [cvoid, u32, u32, u32, u32, cvoid, u32, u32]
        self.lib.pxl_unfilter_legacy.restype = i32
        self.lib.pxl_rans_encode_spatial.argtypes = [cvoid, u32, u32, u32, u32, cvoid, u32, ctypes.POINTER(u32)]
        self.lib.pxl_rans_encode_spatial.restype = i32
        self.lib.pxl_rans_decode_spatial.argtypes = [cvoid, u32, u32, u32, u32, u32, cvoid]
        self.lib.pxl_rans_decode_spatial.restype = i32
        self.lib.pxl_v8_simd_level.restype = i32
        self.lib.pxl_v8_equal_count.argtypes = [cvoid, cvoid, u32]
        self.lib.pxl_v8_equal_count.restype = ctypes.c_uint64
        self.lib.pxl_v8_mixed_encode.argtypes = [cvoid, u32, u32, u32, u32, cvoid, u32, ctypes.POINTER(u32)]
        self.lib.pxl_v8_mixed_encode.restype = i32
        self.lib.pxl_v8_mixed_decode.argtypes = [cvoid, u32, u32, u32, u32, u32, cvoid]
        self.lib.pxl_v8_mixed_decode.restype = i32
        self.lib.pxl_v8_var_create.argtypes = [u32, u32]
        self.lib.pxl_v8_var_create.restype = cvoid
        self.lib.pxl_v8_var_clone.argtypes = [cvoid]
        self.lib.pxl_v8_var_clone.restype = cvoid
        self.lib.pxl_v8_var_destroy.argtypes = [cvoid]
        self.lib.pxl_v8_var_destroy.restype = None
        self.lib.pxl_v8_var_encode.argtypes = [cvoid, cvoid, u32, u32, cvoid, u32, ctypes.POINTER(u32)]
        self.lib.pxl_v8_var_encode.restype = i32
        self.lib.pxl_v8_var_update.argtypes = [cvoid, cvoid, u32, u32]
        self.lib.pxl_v8_var_update.restype = i32
        self.lib.pxl_v8_var_has_large.argtypes = [cvoid, cvoid, u32, u32]
        self.lib.pxl_v8_var_has_large.restype = i32
        self.lib.pxl_v8_var_decode.argtypes = [cvoid, cvoid, u32, u32, u32, cvoid, u32]
        self.lib.pxl_v8_var_decode.restype = i32

    @staticmethod
    def _tables(freq, cumul):
        f = np.ascontiguousarray(freq, dtype=np.uint16).reshape(-1)
        c = np.ascontiguousarray(cumul, dtype=np.uint16).reshape(-1)
        if f.size != c.size:
            raise ValueError("PXL native rANS table shape mismatch")
        return f, c

    def rans_encode_context16(self, data: bytes, frequencies, cumulative):
        n = len(data)
        f, c = self._tables(frequencies, cumulative)
        cap = max(64, n * 2 + 32)
        out = ctypes.create_string_buffer(cap)
        out_n, state = ctypes.c_uint32(), ctypes.c_uint32()
        rc = self.lib.pxl_rans_encode_context16(
            _void_ptr_from_bytes(data), n, _ptr(f), _ptr(c), ctypes.cast(out, ctypes.c_void_p), cap,
            ctypes.byref(out_n), ctypes.byref(state))
        if rc != 0:
            raise RuntimeError(f"PXL C++ context16 rANS encoder failed ({rc})")
        return int(state.value), out.raw[:out_n.value]

    def rans_encode_ids(self, data: bytes, ids: np.ndarray, contexts: int, frequencies, cumulative):
        n = len(data)
        ids = np.ascontiguousarray(ids, dtype=np.uint8).reshape(-1)
        f, c = self._tables(frequencies, cumulative)
        if ids.size != n or f.size != contexts * 256:
            raise ValueError("PXL native rANS input geometry mismatch")
        cap = max(64, n * 2 + 32)
        out = ctypes.create_string_buffer(cap)
        out_n, state = ctypes.c_uint32(), ctypes.c_uint32()
        rc = self.lib.pxl_rans_encode_ids(
            _void_ptr_from_bytes(data), _ptr(ids), n, _ptr(f), _ptr(c), ctypes.cast(out, ctypes.c_void_p), cap,
            ctypes.byref(out_n), ctypes.byref(state))
        if rc != 0:
            raise RuntimeError(f"PXL C++ rANS encoder failed ({rc})")
        return int(state.value), out.raw[:out_n.value]

    def rans_decode_context16(self, blob: bytes, start: int, output_size: int,
                              frequencies, cumulative, decode_table: np.ndarray):
        f, c = self._tables(frequencies, cumulative)
        d = np.ascontiguousarray(decode_table, dtype=np.uint8).reshape(-1)
        out = ctypes.create_string_buffer(output_size)
        end = ctypes.c_uint32()
        rc = self.lib.pxl_rans_decode_context16(
            _void_ptr_from_bytes(blob), len(blob), start, output_size, _ptr(f), _ptr(c), _ptr(d),
            ctypes.cast(out, ctypes.c_void_p), ctypes.byref(end))
        if rc == -6:
            raise ValueError("Truncated PXL context16-rANS renormalisation data")
        if rc != 0:
            raise ValueError(f"Invalid PXL context16-rANS stream (native error {rc})")
        return out.raw[:output_size], int(end.value)

    def rans_decode_context2d(self, blob: bytes, start: int, output_size: int,
                              width: int, height: int, channels: int,
                              frequencies, cumulative, decode_table: np.ndarray):
        f, c = self._tables(frequencies, cumulative)
        d = np.ascontiguousarray(decode_table, dtype=np.uint8).reshape(-1)
        out = ctypes.create_string_buffer(output_size)
        end = ctypes.c_uint32()
        rc = self.lib.pxl_rans_decode_context2d(
            _void_ptr_from_bytes(blob), len(blob), start, output_size, width, height, channels,
            _ptr(f), _ptr(c), _ptr(d), ctypes.cast(out, ctypes.c_void_p), ctypes.byref(end))
        if rc == -6:
            raise ValueError("Truncated PXL 2D rANS renormalisation data")
        if rc != 0:
            raise ValueError(f"Invalid PXL 2D rANS stream (native error {rc})")
        return out.raw[:output_size], int(end.value)

    def rans_decode_context2d_signed(self, blob: bytes, start: int, output_size: int,
                                     width: int, height: int, channels: int,
                                     frequencies, cumulative, decode_table: np.ndarray):
        f, c = self._tables(frequencies, cumulative)
        d = np.ascontiguousarray(decode_table, dtype=np.uint8).reshape(-1)
        out = ctypes.create_string_buffer(output_size)
        end = ctypes.c_uint32()
        rc = self.lib.pxl_rans_decode_context2d_signed(
            _void_ptr_from_bytes(blob), len(blob), start, output_size, width, height, channels,
            _ptr(f), _ptr(c), _ptr(d), ctypes.cast(out, ctypes.c_void_p), ctypes.byref(end))
        if rc == -6:
            raise ValueError("Truncated PXL signed-context rANS renormalisation data")
        if rc != 0:
            raise ValueError(f"Invalid PXL signed-context rANS stream (native error {rc})")
        return out.raw[:output_size], int(end.value)

    def rans_decode_context2d_triple(self, blob: bytes, start: int, output_size: int,
                                     width: int, height: int, channels: int,
                                     frequencies, cumulative, decode_table: np.ndarray):
        f, c = self._tables(frequencies, cumulative)
        d = np.ascontiguousarray(decode_table, dtype=np.uint8).reshape(-1)
        out = ctypes.create_string_buffer(output_size)
        end = ctypes.c_uint32()
        rc = self.lib.pxl_rans_decode_context2d_triple(
            _void_ptr_from_bytes(blob), len(blob), start, output_size, width, height, channels,
            _ptr(f), _ptr(c), _ptr(d), ctypes.cast(out, ctypes.c_void_p), ctypes.byref(end))
        if rc == -6:
            raise ValueError("Truncated PXL triple-context rANS renormalisation data")
        if rc != 0:
            raise ValueError(f"Invalid PXL triple-context rANS stream (native error {rc})")
        return out.raw[:output_size], int(end.value)

    @property
    def simd_name(self):
        return {0: "scalar", 1: "NEON", 2: "AVX2", 3: "AVX-512"}.get(
            int(self.lib.pxl_v8_simd_level()), "unknown")

    def mixed_rans_encode(self, data: bytes, width: int, height: int, channels: int):
        cap = len(data) * 2 + channels * 17 * 520 + 64
        out = ctypes.create_string_buffer(cap)
        out_n = ctypes.c_uint32()
        rc = self.lib.pxl_v8_mixed_encode(_void_ptr_from_bytes(data), len(data), width, height,
                                          channels, ctypes.cast(out, ctypes.c_void_p), cap,
                                          ctypes.byref(out_n))
        if rc != 0:
            raise RuntimeError(f"PXL native mixed-context encoder failed ({rc})")
        return out.raw[:out_n.value]

    def mixed_rans_decode(self, blob: bytes, output_size: int, width: int, height: int, channels: int):
        out = ctypes.create_string_buffer(output_size)
        rc = self.lib.pxl_v8_mixed_decode(_void_ptr_from_bytes(blob), len(blob), output_size,
                                          width, height, channels, ctypes.cast(out, ctypes.c_void_p))
        if rc != 0:
            raise ValueError(f"Invalid PXL native mixed-context stream ({rc})")
        return out.raw[:output_size]

    def new_variable_block_state(self, channels: int, history_limit: int = 32768):
        return NativeVariableBlockState(self, channels, history_limit)

    def new_pixel_lz_state(self, channels: int, window_pixels: int):
        return NativePixelLZState(self, channels, window_pixels)

    def new_block_state(self, channels: int, history_limit: int = 32768):
        return NativeBlockState(self, channels, history_limit)

    def pixel_lz_encode(self, history: bytes, raw: bytes, channels: int, window_pixels: int):
        if len(history) % channels or len(raw) % channels:
            raise ValueError("PXL native PixelLZ input is not pixel-aligned")
        pixels = len(raw) // channels
        cap = len(raw) + pixels * 8 + 64
        out = ctypes.create_string_buffer(cap)
        out_n = ctypes.c_uint32()
        rc = self.lib.pxl_pixel_lz_encode(
            _void_ptr_from_bytes(history), len(history), _void_ptr_from_bytes(raw), len(raw),
            channels, window_pixels, ctypes.cast(out, ctypes.c_void_p), cap, ctypes.byref(out_n))
        if rc != 0:
            raise RuntimeError(f"PXL C++ PixelLZ encoder failed ({rc})")
        return out.raw[:out_n.value]

    def pixel_lz_decode(self, tokens: bytes, history: bytes, expected_pixels: int, channels: int):
        out_n = expected_pixels * channels
        out = ctypes.create_string_buffer(out_n)
        rc = self.lib.pxl_pixel_lz_decode(
            _void_ptr_from_bytes(tokens), len(tokens), _void_ptr_from_bytes(history), len(history),
            expected_pixels, channels, ctypes.cast(out, ctypes.c_void_p), out_n)
        if rc != 0:
            raise ValueError(f"Invalid PXL PixelLZ stream (native error {rc})")
        return out.raw[:out_n]

    def rans_decode_spatial(self, blob: bytes, output_size: int, width: int, height: int, channels: int):
        out = ctypes.create_string_buffer(output_size)
        rc = self.lib.pxl_rans_decode_spatial(_void_ptr_from_bytes(blob), len(blob), output_size,
                                              width, height, channels, ctypes.cast(out, ctypes.c_void_p))
        if rc != 0:
            raise ValueError(f"Invalid PXL C++ spatial rANS stream ({rc})")
        return out.raw[:output_size]

    def filter_legacy(self, raw: bytes, width: int, height: int, channels: int, transform: int):
        output_size = height * (width * channels + 1)
        out = ctypes.create_string_buffer(output_size)
        rc = self.lib.pxl_filter_legacy(_void_ptr_from_bytes(raw), width, height, channels,
                                        transform, ctypes.cast(out, ctypes.c_void_p), output_size)
        if rc != 0:
            raise RuntimeError(f"PXL C++ legacy row filter failed ({rc})")
        return out.raw[:output_size]

    def unfilter_legacy(self, filtered: bytes, width: int, height: int, channels: int, transform: int):
        output_size = width * height * channels
        out = ctypes.create_string_buffer(output_size)
        rc = self.lib.pxl_unfilter_legacy(_void_ptr_from_bytes(filtered), width, height, channels,
                                          transform, ctypes.cast(out, ctypes.c_void_p), len(filtered), output_size)
        if rc != 0:
            raise ValueError(f"Invalid PXL C++ legacy predictor stream ({rc})")
        return out.raw[:output_size]

    def rans_encode_spatial(self, data: bytes, width: int, height: int, channels: int):
        cap = len(data) * 2 + 8192
        out = ctypes.create_string_buffer(cap)
        out_n = ctypes.c_uint32()
        rc = self.lib.pxl_rans_encode_spatial(_void_ptr_from_bytes(data), len(data), width, height,
                                              channels, ctypes.cast(out, ctypes.c_void_p), cap,
                                              ctypes.byref(out_n))
        if rc != 0:
            raise RuntimeError(f"PXL C++ spatial rANS encoder failed ({rc})")
        return out.raw[:out_n.value]

    def filter_color_v3(self, raw: bytes, width: int, height: int, channels: int,
                        transform: int, with_contexts: bool = False):
        expected = height * (width * channels + channels)
        out = ctypes.create_string_buffer(expected)
        ids = np.empty(expected, dtype=np.uint8) if with_contexts else None
        ids_ptr = _ptr(ids) if ids is not None else None
        rc = self.lib.pxl_filter_color_v3(_void_ptr_from_bytes(raw), width, height, channels,
                                          transform, ctypes.cast(out, ctypes.c_void_p), expected, ids_ptr)
        if rc != 0:
            raise RuntimeError(f"PXL C++ fused predictor failed ({rc})")
        filtered = out.raw[:expected]
        return (filtered, ids) if with_contexts else filtered

    def unfilter_color_v3(self, filtered: bytes, width: int, height: int,
                          channels: int, transform: int):
        output_size = width * height * channels
        out = ctypes.create_string_buffer(output_size)
        rc = self.lib.pxl_unfilter_color_v3(
            _void_ptr_from_bytes(filtered), width, height, channels, transform,
            ctypes.cast(out, ctypes.c_void_p), len(filtered), output_size)
        if rc != 0:
            raise ValueError(f"Invalid PXL predictor stream (native error {rc})")
        return out.raw[:output_size]


class NativeBlockState:
    """Native cross-tile 8x8 block dictionary."""
    def __init__(self, backend: NativeBackend, channels: int, history_limit: int = 32768, handle=None):
        self.backend = backend
        self.channels = int(channels)
        self.history_limit = int(history_limit)
        self.handle = handle if handle is not None else backend.lib.pxl_block_state_create(
            self.channels, self.history_limit)
        if not self.handle:
            raise MemoryError("Could not allocate native PXL block dictionary")

    def clone(self):
        handle = self.backend.lib.pxl_block_state_clone(self.handle)
        if not handle:
            raise MemoryError("Could not clone native PXL block dictionary")
        return NativeBlockState(self.backend, self.channels, self.history_limit, handle)

    def encode(self, raw: bytes, width: int, height: int):
        blocks = ((width + 7)//8) * ((height + 7)//8)
        cap = len(raw) + blocks * 16 + 64
        out = ctypes.create_string_buffer(cap)
        out_n = ctypes.c_uint32()
        rc = self.backend.lib.pxl_block_state_encode(self.handle, _void_ptr_from_bytes(raw),
            width, height, ctypes.cast(out, ctypes.c_void_p), cap, ctypes.byref(out_n))
        if rc != 0:
            raise RuntimeError(f"PXL C++ block encoder failed ({rc})")
        return out.raw[:out_n.value]

    def update(self, raw: bytes, width: int, height: int):
        rc = self.backend.lib.pxl_block_state_update(self.handle, _void_ptr_from_bytes(raw), width, height)
        if rc != 0:
            raise RuntimeError(f"PXL C++ block history update failed ({rc})")

    def decode(self, tokens: bytes, width: int, height: int):
        size = width * height * self.channels
        out = ctypes.create_string_buffer(size)
        rc = self.backend.lib.pxl_block_state_decode(self.handle, _void_ptr_from_bytes(tokens),
            len(tokens), width, height, ctypes.cast(out, ctypes.c_void_p), size)
        if rc != 0:
            raise ValueError(f"Invalid PXL C++ block stream ({rc})")
        return out.raw[:size]

    def close(self):
        if getattr(self, "handle", None):
            self.backend.lib.pxl_block_state_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class NativeVariableBlockState:
    """Native variable-size block dictionary with reversible delta references."""
    def __init__(self, backend: NativeBackend, channels: int, history_limit: int = 32768, handle=None):
        self.backend = backend
        self.channels = int(channels)
        self.history_limit = int(history_limit)
        self.handle = handle if handle is not None else backend.lib.pxl_v8_var_create(
            self.channels, self.history_limit)
        if not self.handle:
            raise MemoryError("Could not allocate native PXL variable-block dictionary")

    def clone(self):
        handle = self.backend.lib.pxl_v8_var_clone(self.handle)
        if not handle:
            raise MemoryError("Could not clone native PXL variable-block dictionary")
        return NativeVariableBlockState(self.backend, self.channels, self.history_limit, handle)

    def encode(self, raw: bytes, width: int, height: int):
        cap = len(raw) + max(64, (width + 7) * (height + 7) // 16)
        out = ctypes.create_string_buffer(cap)
        out_n = ctypes.c_uint32()
        rc = self.backend.lib.pxl_v8_var_encode(self.handle, _void_ptr_from_bytes(raw), width, height,
                                                ctypes.cast(out, ctypes.c_void_p), cap,
                                                ctypes.byref(out_n))
        if rc != 0:
            raise RuntimeError(f"PXL native variable-block encoder failed ({rc})")
        return out.raw[:out_n.value]

    def update(self, raw: bytes, width: int, height: int):
        rc = self.backend.lib.pxl_v8_var_update(self.handle, _void_ptr_from_bytes(raw), width, height)
        if rc != 0:
            raise RuntimeError(f"PXL native variable-block history update failed ({rc})")

    def has_large_matches(self, raw: bytes, width: int, height: int):
        return bool(self.backend.lib.pxl_v8_var_has_large(self.handle, _void_ptr_from_bytes(raw), width, height))

    def decode(self, tokens: bytes, width: int, height: int):
        size = width * height * self.channels
        out = ctypes.create_string_buffer(size)
        rc = self.backend.lib.pxl_v8_var_decode(self.handle, _void_ptr_from_bytes(tokens), len(tokens),
                                                width, height, ctypes.cast(out, ctypes.c_void_p), size)
        if rc != 0:
            raise ValueError(f"Invalid PXL native variable-block stream ({rc})")
        return out.raw[:size]

    def close(self):
        if getattr(self, "handle", None):
            self.backend.lib.pxl_v8_var_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class NativePixelLZState:
    """Persistent native PixelLZ index; the C++ dictionary survives tile calls."""
    def __init__(self, backend: NativeBackend, channels: int, window_pixels: int):
        self.backend = backend
        self.channels = int(channels)
        self.window_pixels = int(window_pixels)
        self.handle = backend.lib.pxl_pixel_lz_state_create(self.channels, self.window_pixels)
        if not self.handle:
            raise MemoryError("Could not allocate native PXL PixelLZ state")

    def seed(self, history: bytes, total_pixels: int):
        if len(history) % self.channels:
            raise ValueError("PXL native PixelLZ seed is not pixel aligned")
        rc = self.backend.lib.pxl_pixel_lz_state_seed(
            self.handle, _void_ptr_from_bytes(history), len(history), int(total_pixels))
        if rc != 0:
            raise RuntimeError(f"PXL native PixelLZ seed failed ({rc})")

    def update(self, raw: bytes):
        if len(raw) % self.channels:
            raise ValueError("PXL native PixelLZ update is not pixel aligned")
        rc = self.backend.lib.pxl_pixel_lz_state_update(
            self.handle, _void_ptr_from_bytes(raw), len(raw))
        if rc != 0:
            raise RuntimeError(f"PXL native PixelLZ update failed ({rc})")

    def encode(self, raw: bytes):
        if len(raw) % self.channels:
            raise ValueError("PXL native PixelLZ input is not pixel aligned")
        pixels = len(raw) // self.channels
        cap = len(raw) + pixels * 8 + 64
        out = ctypes.create_string_buffer(cap)
        out_n = ctypes.c_uint32()
        rc = self.backend.lib.pxl_pixel_lz_state_encode(
            self.handle, _void_ptr_from_bytes(raw), len(raw),
            ctypes.cast(out, ctypes.c_void_p), cap, ctypes.byref(out_n))
        if rc != 0:
            raise RuntimeError(f"PXL C++ persistent PixelLZ encoder failed ({rc})")
        return out.raw[:out_n.value]

    def close(self):
        if getattr(self, "handle", None):
            self.backend.lib.pxl_pixel_lz_state_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_backend = None
_checked = False
_load_error = None


def get_backend(required: bool = False, refresh: bool = False):
    global _backend, _checked, _load_error
    if refresh:
        _backend = None
        _checked = False
        _load_error = None
    if not _checked:
        _checked = True
        for path in _candidate_libraries():
            if path.is_file():
                try:
                    _backend = NativeBackend(path)
                    _load_error = None
                    break
                except Exception as exc:  # keep pure-Python fallback usable
                    _load_error = f"{path}: {type(exc).__name__}: {exc}"
    if required and _backend is None:
        extra = f" Last load error: {_load_error}" if _load_error else ""
        raise RuntimeError("PXL C++ backend is unavailable. Run python build_native.py from the project folder." + extra)
    return _backend


def backend_info(refresh: bool = False):
    backend = get_backend(False, refresh=refresh)
    candidates = [str(p) for p in _candidate_libraries() if p.is_file()]
    return {"available": backend is not None,
            "implementation": "C++17 native" if backend else "Python/Numba fallback",
            "library": str(backend.path) if backend else None,
            "simd": backend.simd_name if backend else "scalar/fallback",
            "abi": 5 if backend else None,
            "candidates": candidates,
            "error": _load_error}
