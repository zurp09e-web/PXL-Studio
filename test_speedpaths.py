"""Tests for PXL 5.0's compiled hot paths and exact compatibility."""
import numpy as np
import pxl
import pxl4_engine as v4


def test_v42_predictor_jit_matches_reference_bytes():
    rng = np.random.default_rng(4202)
    for channels in (1, 3, 4):
        width, height = 137, 91
        raw = rng.integers(0, 256, width * height * channels, dtype=np.uint8).tobytes()
        expected = pxl._filter_rows_v3(raw, width, height, channels)
        actual = v4._filter_rows_v3_fast(raw, width, height, channels)
        assert actual == expected
        assert v4._unfilter_rows_v3_fast(actual, width, height, channels) == raw


def test_v42_pixel_lz_jit_cross_tile_exact_roundtrip():
    rng = np.random.default_rng(4203)
    for channels in (1, 3, 4):
        encoder = v4.PixelLZState(channels, tile_size=64)
        decoder = v4.PixelLZState(channels, tile_size=64)
        first = rng.integers(0, 256, 64 * 64 * channels, dtype=np.uint8).tobytes()
        # A full repeated tile exercises long-distance cross-tile references.
        for tile in (first, first, first[:channels] * 4096):
            tokens = encoder.encode(tile)
            restored = decoder.decode(tokens, len(tile) // channels)
            assert restored == tile


def test_v42_decode_state_does_not_allocate_encoder_index():
    state = v4.PixelLZState(3)
    assert not state._hash_arrays_ready
    # The decoder only needs byte history to resolve references.
    rng = np.random.default_rng(4204)
    tile = rng.integers(0, 256, 64 * 64 * 3, dtype=np.uint8).tobytes()
    tokens = v4.PixelLZState(3).encode(tile)
    assert state.decode(tokens, 64 * 64) == tile
    assert not state._hash_arrays_ready


def test_v42_run_decode_jit_and_reference():
    rng = np.random.default_rng(4205)
    cases = [
        b"\x00" * 5000,
        b"A" * 6000,
        (b"abcde" * 1500),
        rng.integers(0, 256, 8000, dtype=np.uint8).tobytes(),
    ]
    for raw in cases:
        tokens = pxl._run_encode(raw)
        assert pxl._run_decode(tokens, len(raw)) == raw


def test_v42_palette_fast_path_is_lossless():
    rng = np.random.default_rng(4206)
    for channels in (1, 3, 4):
        palette = rng.integers(0, 256, (37, channels), dtype=np.uint8)
        indices = rng.integers(0, len(palette), 4096)
        pixels = palette[indices].tobytes()
        payload = pxl._pack_palette(pixels, channels)
        assert payload is not None
        assert pxl._unpack_palette(payload, 64, 64, channels) == pixels
