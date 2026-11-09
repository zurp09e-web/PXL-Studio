import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codec_compare import compare_codecs, export_csv


class CodecComparisonTests(unittest.TestCase):
    def _sample(self, root):
        y, x = np.mgrid[0:83, 0:97]
        pixels = np.stack(((x*3+y) % 256, (y*5) % 256, (x+y*2) % 256), axis=2).astype(np.uint8)
        path = Path(root) / "compare-source.png"
        Image.fromarray(pixels, "RGB").save(path)
        return path, pixels

    def test_lossless_comparison_contains_exact_pxl_and_png(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self._sample(td)
            results = compare_codecs(path, mode="lossless", effort="fast")
            by_name = {row["codec"]: row for row in results}
            self.assertEqual(by_name["PXL lossless"]["status"], "OK")
            self.assertTrue(by_name["PXL lossless"]["exact"])
            self.assertEqual(by_name["PNG"]["status"], "OK")
            self.assertTrue(by_name["PNG"]["exact"])


    def test_pxl_remains_available_when_requested_cuda_is_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self._sample(td)
            with patch("codec_compare.cuda_status", return_value={
                "available": False, "reason": "simulated driver initialization failure"
            }):
                results = compare_codecs(path, mode="lossless", effort="fast", backend="cuda")
            pxl = next(row for row in results if row["codec"] == "PXL lossless")
            self.assertEqual(pxl["status"], "OK")
            self.assertTrue(pxl["exact"])
            self.assertEqual(pxl["backend_requested"], "cuda")
            self.assertEqual(pxl["backend_used"], "cpu")
            self.assertIn("simulated driver initialization failure", pxl["note"])

    def test_external_comparison_codecs_are_not_part_of_codec_lab(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self._sample(td)
            results = compare_codecs(path, mode="lossless", effort="fast")
            names = {row["codec"] for row in results}
            self.assertFalse(any("EMMA" in name or "BMF" in name for name in names))

    def test_lossy_comparison_returns_quality_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self._sample(td)
            results = compare_codecs(path, mode="lossy", quality=82, effort="fast")
            pxl = next(row for row in results if row["codec"] == "PXL lossy")
            self.assertEqual(pxl["status"], "OK")
            self.assertIsNotNone(pxl["psnr_db"])
            self.assertIsNotNone(pxl["ssim"])
            self.assertGreater(pxl["ssim"], 0)
            self.assertLessEqual(pxl["ssim"], 1.000001)

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = self._sample(td)
            rows = compare_codecs(path, mode="lossless", effort="fast")
            out = Path(td) / "results.csv"
            export_csv(rows, out)
            text = out.read_text(encoding="utf-8-sig")
            self.assertIn("PXL lossless", text)
            self.assertIn("size_bytes", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
