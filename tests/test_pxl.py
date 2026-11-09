import os
import random
import tempfile
import unittest

import pxl as pxl_core
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pxl import (compress, decompress, inspect, HEADER, TILE_HEADER, CODEC_DCT, CODEC_RANS_SPATIAL, CODEC_RANS_CONTEXT16, CODEC_LZ_RANS, CODEC_LZ_CONTEXT,
                 HEADER_V1, TILE_HEADER_V1, _forward_color, _filter_rows,
                 TRANSFORM_G, TRANSFORM_YCOCG, _rans_encode, _rans_decode,
                 _rans_context_encode, _rans_context_decode, _rans_context16_encode, _rans_context16_decode, _rans_spatial_encode, _rans_spatial_decode, _encode_lossless_tile, _decode_lossless_tile, _run_encode, _run_decode, _lz_run_encode)
import binascii
import struct
import zlib


class PXLTests(unittest.TestCase):
    def _roundtrip(self, image, mode="lossless", quality=88):
        with tempfile.TemporaryDirectory() as td:
            src, packed, restored = Path(td)/"in.png", Path(td)/"out.pxl", Path(td)/"decoded.png"
            image.save(src)
            compress(src, packed, mode=mode, quality=quality, tile_size=64, backend="cpu", workers=1)
            meta = inspect(packed)
            self.assertEqual(meta["profile"], mode)
            decompress(packed, restored, backend="cpu", workers=1)
            return Image.open(restored).copy(), src, packed, restored

    def test_lossless_modes_exact(self):
        rng = np.random.default_rng(14)
        for mode, shape in (("L", (83, 119)), ("RGB", (83, 119, 3)), ("RGBA", (83, 119, 4))):
            arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
            img = Image.fromarray(arr, mode=mode)
            restored, *_ = self._roundtrip(img)
            self.assertEqual(restored.mode, mode)
            self.assertEqual(img.tobytes(), restored.tobytes())

    def test_palette_and_flat_tiles(self):
        img = Image.new("RGB", (129, 129), (20, 40, 60))
        for x in range(64, 129):
            for y in range(129):
                img.putpixel((x, y), ((x//8*40) % 256, (y//8*30) % 256, 180))
        restored, *_ = self._roundtrip(img)
        self.assertEqual(img.tobytes(), restored.tobytes())

    def test_lossy_dimensions_and_alpha(self):
        rng = np.random.default_rng(5)
        arr = rng.integers(0, 256, size=(77, 91, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        restored, *_ = self._roundtrip(img, mode="lossy", quality=90)
        self.assertEqual(restored.size, img.size)
        self.assertEqual(restored.mode, "RGBA")
        self.assertEqual(restored.getchannel("A").tobytes(), img.getchannel("A").tobytes())

    def test_lossy_grayscale(self):
        arr = np.tile(np.arange(101, dtype=np.uint8), (73, 1))
        img = Image.fromarray(arr, "L")
        restored, *_ = self._roundtrip(img, mode="lossy", quality=85)
        self.assertEqual(restored.size, img.size)
        self.assertEqual(restored.mode, "L")


    def test_lossy_flat_tiles_can_fall_back_to_exact(self):
        img = Image.new("RGB", (128, 128), (20, 40, 60))
        for x in range(32, 96):
            for y in range(128):
                img.putpixel((x, y), (220, 80, 30))
        with tempfile.TemporaryDirectory() as td:
            src, packed, restored = Path(td)/"in.png", Path(td)/"out.pxl", Path(td)/"decoded.png"
            img.save(src)
            compress(src, packed, mode="lossy", quality=72, tile_size=64, backend="cpu", workers=1)
            with packed.open("rb") as f:
                f.seek(HEADER.size)
                tile = TILE_HEADER.unpack(f.read(TILE_HEADER.size))
            self.assertNotEqual(tile[2], CODEC_DCT)
            decompress(packed, restored, backend="cpu", workers=1)
            self.assertEqual(img.tobytes(), Image.open(restored).tobytes())

    def test_tile_corruption_is_detected(self):
        img = Image.new("RGB", (80, 80), (12, 45, 67))
        with tempfile.TemporaryDirectory() as td:
            src, packed, restored = Path(td)/"in.png", Path(td)/"out.pxl", Path(td)/"decoded.png"
            img.save(src)
            compress(src, packed, tile_size=64, backend="cpu", workers=1)
            data = bytearray(packed.read_bytes())
            data[-1] ^= 0x80
            packed.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                decompress(packed, restored)


    def test_legacy_v1_files_still_decode(self):
        rng = np.random.default_rng(21)
        arr = rng.integers(0, 256, size=(71, 75, 3), dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        with tempfile.TemporaryDirectory() as td:
            legacy, restored = Path(td)/"legacy.pxl", Path(td)/"restored.png"
            width, height, channels, tile_size = 75, 71, 3, 64
            with legacy.open("wb") as f:
                f.write(HEADER_V1.pack(b"PXL\0", 1, 0, width, height, tile_size, channels, 8, 0))
                for ty in range(0, height, tile_size):
                    th = min(tile_size, height-ty)
                    for tx in range(0, width, tile_size):
                        tw = min(tile_size, width-tx)
                        tile = bytearray(tw*th*channels)
                        p = 0
                        for y in range(th):
                            start = ((ty+y)*width+tx)*channels
                            n = tw*channels
                            tile[p:p+n] = image.tobytes()[start:start+n]
                            p += n
                        transformed = _forward_color(bytes(tile), channels, TRANSFORM_G)
                        filtered = _filter_rows(transformed, tw, th, channels)
                        body = zlib.compress(filtered, 9)
                        mode = (TRANSFORM_G << 4) | 1
                        crc = binascii.crc32(tile) & 0xffffffff
                        f.write(TILE_HEADER_V1.pack(len(tile), tw, th, mode, 0, len(body), crc, th))
                        f.write(body)
            self.assertEqual(inspect(legacy)["version"], 1)
            decompress(legacy, restored)
            self.assertEqual(image.tobytes(), Image.open(restored).tobytes())


    def test_legacy_v2_files_still_decode(self):
        image = Image.new("RGB", (75, 71), (31, 89, 144))
        width, height, channels, tile_size = 75, 71, 3, 64
        with tempfile.TemporaryDirectory() as td:
            legacy, restored = Path(td)/"legacy-v2.pxl", Path(td)/"restored.png"
            with legacy.open("wb") as f:
                f.write(HEADER.pack(b"PXL\0", 2, 0, channels, 0, width, height, tile_size, 0, 0))
                raw_image = image.tobytes()
                for ty in range(0, height, tile_size):
                    th = min(tile_size, height-ty)
                    for tx in range(0, width, tile_size):
                        tw = min(tile_size, width-tx)
                        tile = bytearray(tw*th*channels)
                        pos = 0
                        for y in range(th):
                            start = ((ty+y)*width+tx)*channels
                            n = tw*channels
                            tile[pos:pos+n] = raw_image[start:start+n]
                            pos += n
                        filtered = bytearray()
                        for y in range(th):
                            filtered.append(0)
                            filtered.extend(tile[y*tw*channels:(y+1)*tw*channels])
                        body = zlib.compress(bytes(filtered), 6)
                        crc = binascii.crc32(body) & 0xffffffff
                        f.write(TILE_HEADER.pack(tw, th, 1, 0, len(body), len(tile), crc))
                        f.write(body)
            self.assertEqual(inspect(legacy)["version"], 2)
            decompress(legacy, restored, workers=1)
            self.assertEqual(image.tobytes(), Image.open(restored).tobytes())

    def test_cpu_multiprocess_roundtrip(self):
        rng = np.random.default_rng(1234)
        arr = rng.integers(0, 256, size=(150, 173, 3), dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        with tempfile.TemporaryDirectory() as td:
            src = Path(td)/"in.png"
            packed = Path(td)/"out.pxl"
            restored = Path(td)/"restored.png"
            image.save(src)
            result = compress(src, packed, tile_size=64, backend="cpu", workers=2)
            self.assertEqual(result["backend"], "cpu")
            self.assertGreaterEqual(result["workers"], 2)
            decoded = decompress(packed, restored, backend="cpu", workers=2)
            self.assertEqual(decoded["backend"], "cpu")
            self.assertEqual(image.tobytes(), Image.open(restored).tobytes())
            # The same pool path must also handle lossy DCT jobs and preserve alpha.
            rgba = Image.fromarray(rng.integers(0, 256, size=(150, 173, 4), dtype=np.uint8), mode="RGBA")
            rgba.save(src)
            compress(src, packed, mode="lossy", quality=84, tile_size=64, backend="cpu", workers=2)
            decompress(packed, restored, backend="cpu", workers=1)
            self.assertEqual(Image.open(restored).getchannel("A").tobytes(), rgba.getchannel("A").tobytes())

    def test_cuda_is_optional_and_reports_missing_runtime(self):
        from pxl import cuda_status
        status = cuda_status()
        self.assertIn("available", status)
        if not status["available"]:
            with tempfile.TemporaryDirectory() as td:
                src, packed = Path(td)/"in.png", Path(td)/"out.pxl"
                Image.new("RGB", (64, 64), (10, 20, 30)).save(src)
                with self.assertRaisesRegex(RuntimeError, "CUDA backend requested"):
                    compress(src, packed, backend="cuda")

    def test_cuda_roundtrips_when_device_is_available(self):
        from pxl import cuda_status
        if not cuda_status().get("available"):
            self.skipTest("CuPy CUDA device not installed")
        rng = np.random.default_rng(99)
        arr = rng.integers(0, 256, size=(83, 91, 3), dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        with tempfile.TemporaryDirectory() as td:
            src, packed, restored = Path(td)/"in.png", Path(td)/"out.pxl", Path(td)/"restored.png"
            image.save(src)
            compress(src, packed, mode="lossless", tile_size=64, backend="cuda")
            decompress(packed, restored, backend="auto")
            self.assertEqual(image.tobytes(), Image.open(restored).tobytes())
            compress(src, packed, mode="lossy", quality=87, tile_size=64, backend="cuda")
            decompress(packed, restored, backend="cuda")
            self.assertEqual(Image.open(restored).size, image.size)

    def test_native_entropy_coders_roundtrip(self):
        rng = np.random.default_rng(2026)
        samples = [
            bytes([0]) * 4096,
            bytes([7, 7, 7, 7, 9, 9, 2, 2, 2, 2, 2]) * 400,
            rng.integers(0, 256, 8192, dtype=np.uint8).tobytes(),
            bytes((int(i * 17) & 255 for i in range(10000))),
        ]
        for original in samples:
            self.assertEqual(_rans_decode(_rans_encode(original), len(original)), original)
            self.assertEqual(_rans_context_decode(_rans_context_encode(original), len(original)), original)
            self.assertEqual(_rans_context16_decode(_rans_context16_encode(original), len(original)), original)
            token = _run_encode(original)
            self.assertEqual(_run_decode(token, len(original)), original)

    def test_custom_lz_tokens_roundtrip_and_compress_repeated_data(self):
        rng = np.random.default_rng(8)
        pattern = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8).tobytes()
        original = pattern * 80 + (b"\x01\x02\x03\x04\x05" * 100) + (b"\x00" * 3000)
        tokens = _lz_run_encode(original)
        self.assertEqual(_run_decode(tokens, len(original)), original)
        self.assertLess(len(tokens), len(original) // 4)

    def test_lz_codec_is_selected_for_repeating_image(self):
        pattern = np.random.default_rng(8).integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        pixels = np.tile(pattern, (8, 8, 1))
        raw = pixels.tobytes()
        codec, transform, payload = _encode_lossless_tile(raw, 128, 128, 3, 6, "cpu")
        self.assertEqual(codec, CODEC_LZ_RANS)
        self.assertEqual(_decode_lossless_tile(payload, 128, 128, 3, codec, transform, 3), raw)

    def test_spatial_codec_is_selected_and_decodes(self):
        width = height = 96
        y, x = np.mgrid[0:height, 0:width]
        pixels = np.stack(((x*2) % 256, (y*3) % 256, ((x+y)*2) % 256), axis=2).astype(np.uint8)
        raw = pixels.tobytes()
        from pxl import _forward_color, _filter_rows, TRANSFORM_YCOCG, _rans_spatial_encode
        transformed = _forward_color(raw, 3, TRANSFORM_YCOCG)
        filtered = _filter_rows(transformed, width, height, 3)
        payload = _rans_spatial_encode(filtered, width, height, 3)
        restored = _decode_lossless_tile(payload, width, height, 3, CODEC_RANS_SPATIAL, TRANSFORM_YCOCG, 3)
        self.assertEqual(restored, raw)

    def test_context16_and_lz_context_tile_decoders(self):
        from pxl import (_forward_color, _filter_rows, TRANSFORM_YCOCG,
                         _rans_context16_encode, _lz_run_encode, _rans_context_encode)
        width = height = 96
        pattern = np.random.default_rng(8).integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        pixels = np.tile(pattern, (6, 6, 1))[:height, :width]
        raw = pixels.tobytes()
        transformed = _forward_color(raw, 3, TRANSFORM_YCOCG)
        filtered = _filter_rows(transformed, width, height, 3)
        context16 = _rans_context16_encode(filtered)
        self.assertEqual(_decode_lossless_tile(context16, width, height, 3,
                         CODEC_RANS_CONTEXT16, TRANSFORM_YCOCG, 3), raw)
        tokens = _lz_run_encode(filtered)
        lz_context = struct.pack("<I", len(tokens)) + _rans_context_encode(tokens)
        self.assertEqual(_decode_lossless_tile(lz_context, width, height, 3,
                         CODEC_LZ_CONTEXT, TRANSFORM_YCOCG, 3), raw)

    def test_spatial_context_rans_roundtrip(self):
        rng = np.random.default_rng(71)
        for channels in (1, 3, 4):
            width, height = 71, 13
            stride = width * channels
            # Build a filtered stream with predictor-ID bytes before every row.
            rows = []
            for y in range(height):
                rows.append(bytes([y % 7]) + rng.integers(0, 32, stride, dtype=np.uint8).tobytes())
            original = b"".join(rows)
            encoded = _rans_spatial_encode(original, width, height, channels)
            decoded = _rans_spatial_decode(encoded, len(original), width, height, channels)
            self.assertEqual(decoded, original)

    def test_v8_lossless_tiles_do_not_use_deflate(self):
        image = Image.new("RGB", (192, 128), (28, 72, 140))
        with tempfile.TemporaryDirectory() as td:
            src, packed = Path(td)/"in.png", Path(td)/"out.pxl"
            image.save(src)
            from unittest.mock import patch
            # A new v8 encode must remain fully independent of DEFLATE.
            with patch("pxl.zlib.compress", side_effect=AssertionError("zlib compressor called")):
                compress(src, packed, tile_size=64, workers=1, backend="cpu")
            self.assertEqual(inspect(packed)["version"], 8)
            with packed.open("rb") as f:
                f.seek(HEADER.size)
                codecs = []
                from pxl4_engine import V8_TILE_HEADER
                for _ in range(6):
                    tile_header = f.read(V8_TILE_HEADER.size)
                    tx, ty, tw, th, codec, transform, n, decoded, crc = V8_TILE_HEADER.unpack(tile_header)
                    codecs.append(codec)
                    f.seek(n, 1)
            self.assertTrue(all(codec not in (1,) for codec in codecs))

    def test_uint32_varint_readers_reject_overflow(self):
        maximum = bytes((0xFF,)) * 4 + bytes((0x0F,))
        self.assertEqual(pxl_core._read_uvarint_bytes(maximum, 0), ((1 << 32) - 1, 5))
        self.assertEqual(pxl_core._read_uvarint(maximum, 0), ((1 << 32) - 1, 5))

        malformed = bytes((0xFF,)) * 4 + bytes((0x10,))
        with self.assertRaisesRegex(ValueError, "Overflowing PXL entropy table varint"):
            pxl_core._read_uvarint_bytes(malformed, 0)
        with self.assertRaisesRegex(ValueError, "Overflowing coefficient varint"):
            pxl_core._read_uvarint(malformed, 0)

        # Exercise the optional Numba token parser too when it is installed.
        if pxl_core._numba_njit is not None:
            tokens = np.frombuffer(bytes((0x80,)) * 9 + bytes((0x02,)), dtype=np.uint8)
            _, _, error = pxl_core._run_read_varint_numba(tokens, 0)
            self.assertEqual(error, -2)


    def test_transform_roundtrip_non_image_bytes(self):
        from pxl import _forward_color, _inverse_color
        data = bytes(random.Random(0).randrange(256) for _ in range(4*100))
        for transform in range(5):
            self.assertEqual(_inverse_color(_forward_color(data, 4, transform), 4, transform), data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
