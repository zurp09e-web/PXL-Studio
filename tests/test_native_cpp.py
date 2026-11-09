import numpy as np
import pytest
from pathlib import Path
from PIL import Image

import pxl
import pxl4_engine as engine
import pxl_native


def native_backend():
    backend = pxl_native.get_backend(False)
    if backend is None:
        pytest.skip("PXL C++ acceleration library has not been built")
    return backend


def test_cpp_native_pixel_lz_rejects_uint64_varint_overflow():
    backend = native_backend()
    malformed = bytes((0,)) + bytes((0x80,)) * 9 + bytes((0x02,))
    with pytest.raises(ValueError, match="Invalid PXL PixelLZ stream"):
        backend.pixel_lz_decode(malformed, b"", expected_pixels=1, channels=1)


def test_cpp_native_variable_blocks_reject_uint64_varint_overflow():
    native_backend()
    decoder = engine.VariableBlockReferenceState(1)
    decoder.update(bytes((17,)) * 64, 8, 8)
    malformed = bytes((1, 0)) + bytes((0x80,)) * 9 + bytes((0x02,))
    with pytest.raises(ValueError, match="Invalid PXL native variable-block stream"):
        decoder.decode(malformed, 8, 8)


def test_cpp_fused_legacy_filter_is_byte_exact():
    backend = native_backend()
    rng = np.random.default_rng(501)
    for channels in (1, 3, 4):
        width, height = 73, 51
        raw = rng.integers(0, 256, width * height * channels, dtype=np.uint8).tobytes()
        for transform in range(5):
            filtered = backend.filter_legacy(raw, width, height, channels, transform)
            reference = pxl._filter_rows(pxl._forward_color(raw, channels, transform),
                                         width, height, channels)
            assert filtered == reference
            assert backend.unfilter_legacy(filtered, width, height, channels, transform) == raw


def test_cpp_spatial_rans_matches_reference_bytes():
    backend = native_backend()
    rng = np.random.default_rng(502)
    from unittest.mock import patch
    for channels in (1, 3, 4):
        width, height = 69, 37
        size = height * (width * channels + 1)
        data = rng.integers(0, 32, size, dtype=np.uint8).tobytes()
        native = backend.rans_encode_spatial(data, width, height, channels)
        with patch("pxl_native.get_backend", return_value=None):
            reference = pxl._rans_spatial_encode(data, width, height, channels)
        assert native == reference
        assert backend.rans_decode_spatial(native, size, width, height, channels) == data


def test_cpp_pixel_lz_state_cross_tile_roundtrip():
    native_backend()
    rng = np.random.default_rng(503)
    encoder = engine.PixelLZState(3, 64)
    decoder = engine.PixelLZState(3, 64)
    source = rng.integers(0, 256, (64 * 64, 3), dtype=np.uint8).tobytes()
    for tile in (source, source, source[:3] * (64 * 64)):
        tokens = encoder.encode(tile)
        assert decoder.decode(tokens, len(tile) // 3) == tile


def test_cpp_block_dictionary_cross_tile_roundtrip():
    native_backend()
    rng = np.random.default_rng(504)
    source = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).tobytes()
    encoder = engine.BlockReferenceState(3)
    decoder = engine.BlockReferenceState(3)
    for tile in (source, source, source):
        tokens = encoder.encode(tile, 64, 64)
        assert decoder.decode(tokens, 64, 64) == tile


def test_native_legacy_v4_file_reader():
    from pathlib import Path
    fixture = Path(__file__).with_name("fixture_v4.pxl")
    expected = Path(__file__).with_name("fixture_v4_expected.png")
    if not fixture.exists() or not expected.exists():
        pytest.skip("legacy PXL v4 fixture not included")
    assert pxl.inspect(fixture)["version"] == 4
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        restored = Path(td) / "legacy.png"
        result = pxl.decompress(fixture, restored, backend="cpu", workers=1)
        assert result["version"] == 4
        with Image.open(expected) as a, Image.open(restored) as b:
            assert a.mode == b.mode
            assert a.size == b.size
            assert a.tobytes() == b.tobytes()


def test_v6_parallel_entropy_chunks_roundtrip():
    native_backend()
    # Each independently coded region has a different low-entropy symbol.
    # A single global model is less efficient, so PXMS should win by final size.
    raw = b"".join(bytes((symbol,)) * 65536 for symbol in range(8))
    payload = engine._encode_rans_tokens(raw, effort="balanced")
    assert payload[4:8] == b"PXMS"
    assert engine._decode_rans_tokens(payload, allow_parallel=True) == raw


def test_all_effort_profiles_write_exact_pxl_v8(tmp_path):
    source = tmp_path / "effort.png"
    pixels = np.zeros((128, 128, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(128, dtype=np.uint8)[None, :]
    pixels[:, :, 1] = np.arange(128, dtype=np.uint8)[:, None]
    pixels[:, :, 2] = (pixels[:, :, 0] + pixels[:, :, 1]) & 255
    Image.fromarray(pixels, "RGB").save(source)
    for effort in ("fast", "balanced", "maximum"):
        packed = tmp_path / f"{effort}.pxl"
        restored = tmp_path / f"{effort}.png"
        result = pxl.compress(source, packed, mode="lossless", tile_size=64,
                              effort=effort, backend="cpu", workers=1)
        assert result["version"] == 8
        pxl.decompress(packed, restored, backend="cpu", workers=1)
        with Image.open(restored) as decoded:
            assert decoded.convert("RGB").tobytes() == pixels.tobytes()


def test_cpp_signed_context_rans_roundtrip():
    backend = native_backend()
    # A smooth field produces many exact 0/+1/-1 prediction residuals.
    y, x = np.mgrid[0:97, 0:113]
    image = np.stack(((x + y) & 255, (x * 2 + y) & 255, (x + y * 2) & 255), axis=-1).astype(np.uint8)
    filtered, _ = backend.filter_color_v3(image.tobytes(), 113, 97, 3, 1, with_contexts=True)
    ids = engine._context2d_signed_ids(filtered, 113, 97, 3)
    packed = engine._rans_context2d_encode(filtered, 113, 97, 3, ids)
    decoded = engine._rans_context2d_decode(packed, len(filtered), 113, 97, 3,
                                             signed_context=True)
    assert decoded == filtered


def test_v5_fixture_remains_decodable_after_v6_upgrade(tmp_path):
    fixture = Path(__file__).with_name("fixture_v5.pxl")
    assert pxl.inspect(fixture)["version"] == 5
    restored = tmp_path / "legacy-v5.png"
    result = pxl.decompress(fixture, restored, backend="cpu", workers=1)
    assert result["version"] == 5
    y, x = np.mgrid[:96, :112]
    expected = np.stack(((x * 3 + y) & 255, (x + y * 5) & 255,
                         (x * 7 + y * 2) & 255), axis=-1).astype(np.uint8)
    with Image.open(restored) as decoded:
        assert np.array_equal(np.asarray(decoded.convert("RGB")), expected)


def test_cpp_triple_context_rans_roundtrip_and_python_fallback():
    backend = native_backend()
    y, x = np.mgrid[0:81, 0:93]
    image = np.stack(((x + y) & 255, (x * 2 + y) & 255,
                      (x + y * 2) & 255), axis=-1).astype(np.uint8)
    filtered, _ = backend.filter_color_v3(image.tobytes(), 93, 81, 3, 0, with_contexts=True)
    ids = engine._context2d_triple_ids(filtered, 93, 81, 3)
    packed = engine._rans_context2d_encode(filtered, 93, 81, 3, ids, context_count=65)
    native_decoded = engine._rans_context2d_decode(
        packed, len(filtered), 93, 81, 3, triple_context=True)
    assert native_decoded == filtered
    from unittest.mock import patch
    with patch("pxl_native.get_backend", return_value=None):
        fallback_packed = engine._rans_context2d_encode(
            filtered, 93, 81, 3, ids, context_count=65)
        fallback_decoded = engine._rans_context2d_decode(
            fallback_packed, len(filtered), 93, 81, 3, triple_context=True)
    assert fallback_decoded == filtered


def test_maximum_effort_can_select_context_codec(tmp_path):
    source = Path(__file__).parents[1] / "benchmarks_v50" / "images" / "photo_like.png"
    packed, restored = tmp_path / "maximum.pxl", tmp_path / "restored.png"
    result = pxl.compress(source, packed, tile_size=256, effort="maximum",
                          backend="cpu", workers=1)
    assert result["version"] == 8
    codecs = []
    with packed.open("rb") as stream:
        stream.seek(engine.HEADER.size)
        for _ in range(result["tiles"]):
            tx, ty, tw, th, codec, transform, payload_size, decoded_size, crc = engine.V8_TILE_HEADER.unpack(
                stream.read(engine.V8_TILE_HEADER.size))
            codecs.append(codec)
            stream.seek(payload_size, 1)
    assert engine.CODEC_PRED2D_TRIPLE in codecs
    pxl.decompress(packed, restored, backend="cpu", workers=1)
    with Image.open(source) as a, Image.open(restored) as b:
        assert pxl._normalise_image(a).tobytes() == pxl._normalise_image(b).tobytes()


def test_cpp_mixed_context_rans_matches_python_fallback():
    backend = native_backend()
    rng = np.random.default_rng(8801)
    for channels in (1, 3, 4):
        width, height = 79, 53
        row = width * channels + channels
        values = rng.integers(0, 16, size=(height, row), dtype=np.uint8)
        # Preserve the dedicated row-selector byte and create strong local patterns.
        values[:, 0] = np.arange(height, dtype=np.uint8) % 5
        raw = values.tobytes()
        native = backend.mixed_rans_encode(raw, width, height, channels)
        from unittest.mock import patch
        with patch("pxl_native.get_backend", return_value=None):
            fallback = engine._rans_context2d_mixed_encode(raw, width, height, channels)
            decoded = engine._rans_context2d_mixed_decode(fallback, len(raw), width, height, channels)
        assert native == fallback
        assert backend.mixed_rans_decode(native, len(raw), width, height, channels) == raw
        assert decoded == raw


def test_cpp_variable_blocks_emit_real_delta_references():
    native_backend()
    rng = np.random.default_rng(8802)
    reference = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    current = reference.copy()
    # A few scattered changes leave most bytes equal, favouring zero-run deltas.
    current.reshape(-1)[::97] ^= 0x5A
    ref_bytes, cur_bytes = reference.tobytes(), current.tobytes()
    encoder = engine.VariableBlockReferenceState(3)
    decoder = engine.VariableBlockReferenceState(3)
    encoder.update(ref_bytes, 64, 64)
    decoder.update(ref_bytes, 64, 64)
    assert encoder.has_large_matches(cur_bytes, 64, 64)
    tokens = encoder.encode(cur_bytes, 64, 64)
    assert tokens[0] == 2 and tokens[1] in (1, 2), \
        f"expected a delta-reference token, got {tokens[:4]!r}"
    assert decoder.decode(tokens, 64, 64) == cur_bytes


def test_native_simd_dispatch_has_scalar_correctness():
    backend = native_backend()
    a = np.random.default_rng(8803).integers(0, 256, size=10003, dtype=np.uint8)
    b = a.copy()
    b[::11] ^= 0xFF
    expected = int(np.count_nonzero(a == b))
    got = int(backend.lib.pxl_v8_equal_count(
        a.ctypes.data_as(__import__('ctypes').c_void_p),
        b.ctypes.data_as(__import__('ctypes').c_void_p), len(a)))
    assert got == expected
    assert backend.simd_name in {"scalar", "NEON", "AVX2", "AVX-512"}


def test_python_fallback_variable_blocks_emit_delta_references():
    from unittest.mock import patch
    rng = np.random.default_rng(8804)
    reference = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    current = reference.copy()
    current.reshape(-1)[::97] ^= 0x5A
    ref_bytes, cur_bytes = reference.tobytes(), current.tobytes()
    with patch("pxl_native.get_backend", return_value=None):
        encoder = engine.VariableBlockReferenceState(3)
        decoder = engine.VariableBlockReferenceState(3)
        encoder.update(ref_bytes, 64, 64)
        decoder.update(ref_bytes, 64, 64)
        tokens = encoder.encode(cur_bytes, 64, 64)
        assert tokens[0] == 2 and tokens[1] in (1, 2)
        assert decoder.decode(tokens, 64, 64) == cur_bytes


def _mixed_rans_blob_with_first_context(table: bytes) -> bytes:
    # The one-channel model has 17 contexts. Context zero is deliberately
    # malformed; the remaining contexts are absent, as allowed by the format.
    contexts = 17
    blob = bytearray((contexts,))
    for context in range(contexts):
        entry = table if context == 0 else b""
        blob.extend(len(entry).to_bytes(2, "little"))
        blob.extend(entry)
    blob.extend((1 << 23).to_bytes(4, "little"))
    return bytes(blob)


@pytest.mark.parametrize("table", [
    b"\x00" + (2049).to_bytes(2, "little") + (2048).to_bytes(2, "little")
    + bytes(254 * 2),
    bytes((1, 2, 0, 0x81, 0x10, 1, 0x80, 0x10)),
], ids=["dense-frequency-overflow", "sparse-frequency-overflow"])
def test_cpp_mixed_context_decoder_rejects_frequency_sum_overflow(table):
    backend = native_backend()
    # width=height=channels=1 produces two filtered bytes. The corrupted
    # model sums to 4097, one past the fixed 4096-entry rANS decode table.
    blob = _mixed_rans_blob_with_first_context(table)
    with pytest.raises(ValueError, match="mixed-context stream"):
        backend.mixed_rans_decode(blob, output_size=2, width=1, height=1, channels=1)
