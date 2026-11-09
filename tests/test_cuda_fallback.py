import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pxl4_engine
from pxl import compress, decompress


def test_auto_backend_falls_back_after_cuda_predictor_failure():
    height, width = 128, 160
    y, x = np.mgrid[0:height, 0:width]
    pixels = np.stack(((x * 3 + y) & 255,
                       (x + y * 2) & 255,
                       (x * 2 + y * 3) & 255), axis=2).astype(np.uint8)
    original_predictor = pxl4_engine._predictor_candidate
    calls = []

    def failing_cuda_predictor(*args, **kwargs):
        selected_backend = kwargs.get("backend", args[5] if len(args) > 5 else "cpu")
        calls.append(selected_backend)
        if selected_backend == "cuda":
            raise RuntimeError("simulated CUDA kernel launch failure")
        return original_predictor(*args, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, archive, restored = td / "in.png", td / "out.pxl", td / "restored.png"
        Image.fromarray(pixels, "RGB").save(src)
        with patch.object(pxl4_engine.core, "_get_cupy", return_value=object()), \
             patch.object(pxl4_engine, "_predictor_candidate", side_effect=failing_cuda_predictor):
            result = compress(src, archive, tile_size=64, effort="fast", backend="auto", workers=1)
        assert result["backend"] == "cpu"
        assert "simulated CUDA kernel launch failure" in result["cuda_fallback"]
        assert "cuda" in calls and "cpu" in calls
        decompress(archive, restored, backend="cpu", workers=1)
        with Image.open(restored) as decoded:
            assert decoded.convert("RGB").tobytes() == pixels.tobytes()


def test_cuda_predictor_uses_filter_implementation_from_core_module():
    """Regression test for the v8 CUDA NameError reported by Windows users."""
    height, width, channels = 64, 80, 3
    y, x = np.mgrid[0:height, 0:width]
    pixels = np.stack(((x * 3 + y) & 255,
                       (x + y * 2) & 255,
                       (x * 2 + y * 3) & 255), axis=2).astype(np.uint8)
    raw = pixels.tobytes()
    filtered_calls = []

    def fake_cuda_forward(data, channel_count, transform):
        return pxl4_engine.core._forward_color(data, channel_count, transform)

    def fake_cuda_filter(data, w, h, channel_count):
        filtered_calls.append((w, h, channel_count))
        return pxl4_engine._filter_rows_v3_fast(data, w, h, channel_count)

    with patch.object(pxl4_engine.core, "_forward_color_cuda", side_effect=fake_cuda_forward), \
         patch.object(pxl4_engine.core, "_filter_rows_cuda_v3", side_effect=fake_cuda_filter):
        candidate = pxl4_engine._predictor_candidate(
            raw, width, height, channels, effort=6, backend="cuda",
            parallel_transforms=False)

    assert candidate is not None
    assert len(candidate) == 3
    assert len(filtered_calls) == 3  # NONE, G and YCoCg candidate paths
    assert all(call == (width, height, channels) for call in filtered_calls)


def test_full_explicit_cuda_encode_roundtrips_after_core_filter_routing():
    """Exercise the public PXL encode route without requiring a physical GPU."""
    height, width = 128, 128
    y, x = np.mgrid[0:height, 0:width]
    pixels = np.stack(((x * 3 + y) & 255,
                       (x + y * 2) & 255,
                       (x * 2 + y * 3) & 255), axis=2).astype(np.uint8)
    calls = []

    def fake_cuda_forward(data, channel_count, transform):
        return pxl4_engine.core._forward_color(data, channel_count, transform)

    def fake_cuda_filter(data, w, h, channel_count):
        calls.append((w, h, channel_count))
        return pxl4_engine._filter_rows_v3_fast(data, w, h, channel_count)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, archive, restored = td / "source.png", td / "result.pxl", td / "restored.png"
        Image.fromarray(pixels, "RGB").save(src)
        with patch.object(pxl4_engine.core, "_get_cupy", return_value=object()), \
             patch.object(pxl4_engine.core, "_forward_color_cuda", side_effect=fake_cuda_forward), \
             patch.object(pxl4_engine.core, "_filter_rows_cuda_v3", side_effect=fake_cuda_filter):
            result = compress(src, archive, mode="lossless", tile_size=64,
                              effort="fast", backend="cuda", workers=1,
                              adaptive_tiles=False)
        assert result["cuda_tiles_used"] > 0
        assert calls
        pxl4_engine.decompress(archive, restored, backend="cpu", workers=1)
        with Image.open(restored) as decoded:
            assert decoded.convert("RGB").tobytes() == pixels.tobytes()
