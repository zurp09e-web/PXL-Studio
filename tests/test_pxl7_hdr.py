import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

import pxl
import pxl4_engine as engine
import pxl7_hdr
from pxl7_hdr import SAMPLE_FLOAT16, SAMPLE_FLOAT32, SAMPLE_UINT16, probe_high_precision


def pxl7_read_png(path):
    import png as pypng
    width, height, rows, info = pypng.Reader(filename=str(path)).read()
    planes = int(info.get("planes", 1))
    raw = np.fromiter((sample for row in rows for sample in row), dtype=np.uint16,
                      count=width * height * planes)
    shaped = raw.reshape(height, width, planes)
    return shaped[..., 0] if planes == 1 else shaped


class PXL7HighPrecisionTests(unittest.TestCase):
    def test_legacy_v7_high_precision_fixture_still_decodes(self):
        fixture_dir = Path(__file__).parent
        packed = fixture_dir / "fixture_v7_hdr.pxl"
        expected = fixture_dir / "fixture_v7_hdr_expected.tif"
        with tempfile.TemporaryDirectory() as td:
            restored = Path(td) / "restored.tif"
            info = pxl.inspect(packed)
            self.assertEqual(info["version"], 7)
            self.assertEqual(info["sample_format"], "uint16")
            result = pxl.decompress(packed, restored, backend="cpu", workers=1)
            self.assertEqual(result["version"], 7)
            self.assertTrue(np.array_equal(tifffile.imread(expected), tifffile.imread(restored)))

    def _roundtrip_tiff(self, array, channels, sample_format):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, packed, restored = root / "source.tif", root / "image.pxl", root / "restored.tif"
            tifffile.imwrite(source, array, photometric="rgb" if channels in (3, 4) else "minisblack")
            probe = probe_high_precision(source)
            self.assertEqual((probe["width"], probe["height"], probe["channels"]),
                             (array.shape[1], array.shape[0], channels))
            result = pxl.compress(source, packed, tile_size=64, effort="fast", backend="cpu", workers=1)
            self.assertEqual(result["version"], 7)
            self.assertEqual(result["sample_format"], sample_format)
            self.assertEqual(pxl.inspect(packed)["sample_format"], sample_format)
            # The high-precision envelope stays v7, but every newly written
            # embedded 8-bit byte-plane stream should use the v8 tile format.
            from pxl7_hdr import PLANE_ENTRY
            with packed.open("rb") as stream:
                stream.seek(engine.HEADER.size)
                entries = [PLANE_ENTRY.unpack(stream.read(PLANE_ENTRY.size))
                           for _ in range(array.dtype.itemsize)]
                for length, _crc in entries:
                    plane = stream.read(length)
                    self.assertGreaterEqual(len(plane), 5)
                    self.assertEqual(plane[4], 8)
            pxl.decompress(packed, restored, backend="cpu", workers=1)
            decoded = tifffile.imread(restored)
            self.assertEqual(decoded.shape, array.shape)
            if array.dtype.kind == "f":
                uint = np.uint16 if array.dtype.itemsize == 2 else np.uint32
                self.assertTrue(np.array_equal(array.view(uint), decoded.view(uint)))
            else:
                self.assertTrue(np.array_equal(array, decoded))

    def test_uint16_rgb_tiff_exact(self):
        y, x = np.mgrid[:80, :96]
        array = np.stack(((x * 701 + y * 13) & 65535,
                          (y * 929 + x * 3) & 65535,
                          (x * 17 + y * 41) & 65535), axis=-1).astype(np.uint16)
        self._roundtrip_tiff(array, 3, "uint16")

    @unittest.skipUnless(importlib.util.find_spec("png"), "PyPNG optional 16-bit PNG support is not installed")
    def test_uint16_rgb_png_exact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(128)
            array = rng.integers(0, 65536, size=(80, 96, 3), dtype=np.uint16)
            source, packed, restored = root / "source.png", root / "image.pxl", root / "restored.png"
            import png as pypng
            writer = pypng.Writer(width=96, height=80, bitdepth=16)
            with source.open("wb") as stream:
                writer.write(stream, (array[y].reshape(-1).tolist() for y in range(80)))
            pxl.compress(source, packed, tile_size=64, effort="fast", backend="cpu", workers=1)
            pxl.decompress(packed, restored, backend="cpu", workers=1)
            decoded = pxl7_read_png(restored)
            self.assertTrue(np.array_equal(array, decoded))

    @unittest.skipUnless(importlib.util.find_spec("OpenEXR"), "OpenEXR optional support is not installed")
    def test_openexr_float32_exact_when_installed(self):
        import OpenEXR
        from pxl7_hdr import _read_exr
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bits = np.arange(64 * 80 * 3, dtype=np.uint32).reshape(64, 80, 3) * 7919
            bits[0, 0, 0] = 0x80000000
            bits[0, 0, 1] = 0x7FC01234
            array = bits.view(np.float32)
            source, packed, restored = root / "source.exr", root / "image.pxl", root / "restored.exr"
            with OpenEXR.File({"type": OpenEXR.scanlineimage}, {"RGB": array}) as output:
                output.write(str(source))
            pxl.compress(source, packed, tile_size=64, effort="fast", backend="cpu", workers=1)
            pxl.decompress(packed, restored, backend="cpu", workers=1)
            decoded = _read_exr(restored)
            self.assertTrue(np.array_equal(array.view(np.uint32), decoded.view(np.uint32)))

    def test_uint16_grayscale_png_exact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            array = (np.arange(80 * 96, dtype=np.uint32).reshape(80, 96) * 131 % 65536).astype(np.uint16)
            source, packed, restored = root / "source.png", root / "image.pxl", root / "restored.png"
            Image.fromarray(array).save(source)
            probe = probe_high_precision(source)
            self.assertEqual(probe["sample_format"], "uint16")
            pxl.compress(source, packed, tile_size=64, effort="fast", backend="cpu", workers=1)
            pxl.decompress(packed, restored, backend="cpu", workers=1)
            decoded = np.asarray(Image.open(restored), dtype=np.uint16)
            self.assertTrue(np.array_equal(array, decoded))

    def test_float16_rgba_exact(self):
        bits = np.arange(80 * 96 * 4, dtype=np.uint32).reshape(80, 96, 4)
        # Include diverse finite values, signed zero and selected NaN payloads.
        raw16 = (bits * 37 + 0x3800).astype(np.uint16)
        raw16.flat[0] = 0x8000
        raw16.flat[1] = 0x7E55
        array = raw16.view(np.float16)
        self._roundtrip_tiff(array, 4, "float16")

    def test_float32_hdr_exact(self):
        bits = np.arange(80 * 96 * 3, dtype=np.uint32).reshape(80, 96, 3) * 7919
        bits[..., 0] = 0x3F800000  # 1.0
        bits[0, 0, 1] = 0x80000000  # negative zero
        bits[0, 0, 2] = 0x7FC01234  # quiet NaN with payload
        array = bits.view(np.float32)
        self._roundtrip_tiff(array, 3, "float32")

    def test_all_reversible_uint16_color_transforms(self):
        rng = np.random.default_rng(903)
        array = rng.integers(0, 65536, size=(29, 37, 4), dtype=np.uint16)
        from pxl7_hdr import _forward_uint16_color, _inverse_uint16_color
        for transform in range(5):
            encoded = _forward_uint16_color(array, 4, transform)
            decoded = _inverse_uint16_color(encoded, 4, transform)
            self.assertTrue(np.array_equal(array, decoded), f"transform {transform}")

    def test_lossy_high_precision_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.tif"
            tifffile.imwrite(source, np.ones((64, 64), dtype=np.uint16))
            with self.assertRaisesRegex(ValueError, "lossless mode only"):
                pxl.compress(source, root / "lossy.pxl", mode="lossy", tile_size=64, workers=1)


class PXL7ModelTests(unittest.TestCase):
    def test_context_mixing_entropy_roundtrip(self):
        y, x = np.mgrid[:73, :81]
        pixels = np.stack(((x + y) & 255, (x * 2 + y) & 255, (x + y * 2) & 255), axis=-1).astype(np.uint8)
        filtered = engine._filter_rows_v3_fast(pixels.tobytes(), 81, 73, 3)
        encoded = engine._rans_context2d_mixed_encode(filtered, 81, 73, 3)
        decoded = engine._rans_context2d_mixed_decode(encoded, len(filtered), 81, 73, 3)
        self.assertEqual(decoded, filtered)

    def test_variable_block_dictionary_32_16_8_roundtrip(self):
        rng = np.random.default_rng(718)
        tile = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8).tobytes()
        encoder = engine.VariableBlockReferenceState(3)
        decoder = engine.VariableBlockReferenceState(3)
        encoder.update(tile, 64, 64)
        decoder.update(tile, 64, 64)
        tokens = encoder.encode(tile, 64, 64)
        self.assertLess(len(tokens), len(tile))
        self.assertEqual(decoder.decode(tokens, 64, 64), tile)


    def test_v7_inspect_rejects_aggregate_embedded_payload_amplification(self):
        width = height = 64
        channels = 1
        bytes_per_sample = 2
        plane_count = 2
        decoded_sample_bytes = width * height * channels * bytes_per_sample
        declared_payload_bytes = (
            decoded_sample_bytes + pxl7_hdr.MAX_EMBEDDED_ARCHIVE_OVERHEAD + 1
        )
        first_length = declared_payload_bytes - 1
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "oversized-planes.pxl"
            header = pxl.HEADER.pack(
                pxl.MAGIC, pxl7_hdr.VERSION, pxl.PROFILE_LOSSLESS,
                channels, SAMPLE_UINT16, width, height, 64,
                bytes_per_sample, plane_count,
            )
            with archive.open("wb") as stream:
                stream.write(header)
                stream.write(pxl7_hdr.PLANE_ENTRY.pack(first_length, 0))
                stream.write(pxl7_hdr.PLANE_ENTRY.pack(1, 0))
                stream.seek(len(header) + plane_count * pxl7_hdr.PLANE_ENTRY.size
                            + declared_payload_bytes - 1)
                stream.write(bytes((0,)))

            with self.assertRaisesRegex(ValueError, "aggregate embedded planes"):
                pxl7_hdr.inspect(archive)


    def test_v7_decoder_rejects_embedded_plane_dimension_mismatch_before_decode(self):
        width = height = 64
        channels = 1
        bytes_per_sample = 2
        plane_count = 2
        # A tiny nested PXL header advertises a huge decoded image. The outer
        # dimensions must be checked before invoking the nested decoder.
        oversized_plane = pxl.HEADER.pack(
            pxl.MAGIC, engine.VERSION, pxl.PROFILE_LOSSLESS,
            channels, 0, 8192, 8192, 64, 0, 0,
        )
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "mismatched-plane.pxl"
            output = Path(td) / "restored.tif"
            header = pxl.HEADER.pack(
                pxl.MAGIC, pxl7_hdr.VERSION, pxl.PROFILE_LOSSLESS,
                channels, SAMPLE_UINT16, width, height, 64,
                bytes_per_sample, plane_count,
            )
            with archive.open("wb") as stream:
                stream.write(header)
                for _ in range(plane_count):
                    stream.write(pxl7_hdr.PLANE_ENTRY.pack(len(oversized_plane), 0))
                stream.write(oversized_plane)
                stream.write(oversized_plane)

            with self.assertRaisesRegex(ValueError, "geometry/profile mismatch"):
                pxl7_hdr.decompress(archive, output, backend="cpu", workers=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
