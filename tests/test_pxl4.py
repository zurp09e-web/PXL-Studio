import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import pxl
import pxl4_engine as v4


def run_roundtrip(array, mode, tile=64, workers=1):
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'source.png'
        packed = Path(td) / 'image.pxl'
        restored = Path(td) / 'restored.png'
        Image.fromarray(array, mode=mode).save(src)
        result = pxl.compress(src, packed, mode='lossless', tile_size=tile,
                              effort='balanced', backend='cpu', workers=workers)
        assert result['version'] == 8
        info = pxl.inspect(packed)
        assert info['version'] == 8
        decoded = pxl.decompress(packed, restored, backend='cpu', workers=1)
        assert decoded['version'] == 8
        with Image.open(src) as a, Image.open(restored) as b:
            aa = pxl._normalise_image(a)
            bb = pxl._normalise_image(b)
            assert aa.mode == bb.mode
            assert aa.size == bb.size
            assert aa.tobytes() == bb.tobytes()
        return packed.stat().st_size


def test_modes_and_edge_tiles_v8():
    rng = np.random.default_rng(12345)
    for mode, shape in [('L', (131, 145)), ('RGB', (131, 145, 3)), ('RGBA', (131, 145, 4))]:
        if mode == 'L':
            a = rng.integers(0, 256, size=shape, dtype=np.uint8)
        else:
            a = rng.integers(0, 256, size=shape, dtype=np.uint8)
        run_roundtrip(a, mode)


def test_cross_tile_repeated_pixels():
    # Repeating pixel data across tile boundaries should remain bit-exact.
    rng = np.random.default_rng(8)
    patch = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    image = np.tile(patch, (4, 4, 1))
    run_roundtrip(image, 'RGB', tile=64)


def test_repeated_2d_blocks():
    rng = np.random.default_rng(9)
    block = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    image = np.tile(block, (16, 16, 1))
    run_roundtrip(image, 'RGB', tile=64)


def test_smooth_prediction():
    y, x = np.mgrid[0:129, 0:141]
    image = np.stack(((x * 2 + y) % 256, (x + y * 3) % 256, (x * 3 + y * 2) % 256), axis=-1).astype(np.uint8)
    run_roundtrip(image, 'RGB')


def test_multicore_matches_single_core():
    y, x = np.mgrid[0:128, 0:128]
    image = np.stack(((x * 3) % 256, (y * 5) % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'src.png'; a = Path(td) / 'a.pxl'; b = Path(td) / 'b.pxl'
        Image.fromarray(image, mode='RGB').save(src)
        pxl.compress(src, a, tile_size=64, workers=1, backend='cpu')
        pxl.compress(src, b, tile_size=64, workers=2, backend='cpu')
        # Predictor work is deterministic and dictionary state is committed in
        # tile order, so worker scheduling must not alter the stream.
        assert a.read_bytes() == b.read_bytes()


def test_lossy_roundtrip_and_alpha_exact():
    rng = np.random.default_rng(77)
    image = rng.integers(0, 256, size=(96, 96, 4), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'src.png'; packed = Path(td) / 'lossy.pxl'; out = Path(td) / 'out.png'
        Image.fromarray(image, mode='RGBA').save(src)
        pxl.compress(src, packed, mode='lossy', quality=90, tile_size=64, workers=1, backend='cpu')
        pxl.decompress(packed, out)
        decoded = np.asarray(Image.open(out).convert('RGBA'))
        assert np.array_equal(decoded[..., 3], image[..., 3])


def test_pixel_lz_state_roundtrip_direct():
    ch = 3
    state_enc = v4.PixelLZState(ch)
    state_dec = v4.PixelLZState(ch)
    rng = np.random.default_rng(4)
    first = rng.integers(0, 256, size=(100, 3), dtype=np.uint8).tobytes()
    second = first[:60] + first[60:]
    t1 = state_enc.encode(first)
    assert state_dec.decode(t1, len(first)//ch) == first
    t2 = state_enc.encode(second)
    assert state_dec.decode(t2, len(second)//ch) == second


def test_block_state_roundtrip_direct():
    rng = np.random.default_rng(10)
    tile1 = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8).tobytes()
    tile2 = tile1
    enc = v4.BlockReferenceState(3)
    dec = v4.BlockReferenceState(3)
    t1 = enc.encode(tile1, 64, 64)
    assert dec.decode(t1, 64, 64) == tile1
    t2 = enc.encode(tile2, 64, 64)
    assert dec.decode(t2, 64, 64) == tile2


def test_corrupt_tile_rejected():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'src.png'; packed = Path(td) / 'bad.pxl'; out = Path(td) / 'out.png'
        Image.fromarray(image, mode='RGB').save(src)
        pxl.compress(src, packed, tile_size=64, workers=1, backend='cpu')
        blob = bytearray(packed.read_bytes())
        blob[-1] ^= 0x80
        packed.write_bytes(blob)
        try:
            pxl.decompress(packed, out)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupted tile must be rejected')


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))

def test_legacy_v3_files_still_decode():
    image = np.arange(96 * 80 * 3, dtype=np.uint8).reshape(80, 96, 3)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'src.png'; packed = Path(td) / 'legacy-v3.pxl'; out = Path(td) / 'out.png'
        Image.fromarray(image, mode='RGB').save(src)
        pxl._compress_v3(src, packed, tile_size=64, backend='cpu', workers=1)
        with packed.open('rb') as f:
            assert f.read(5)[4] == 3
        pxl.decompress(packed, out, backend='cpu', workers=1)
        assert np.array_equal(np.asarray(Image.open(out).convert('RGB')), image)

def test_skipped_high_entropy_history_can_be_indexed_by_later_tile():
    rng = np.random.default_rng(4242)
    tile = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    image = np.concatenate([tile, tile], axis=1)
    run_roundtrip(image, 'RGB', tile=64)


def test_signed_context_candidate_selected_for_smooth_gradient():
    y, x = np.mgrid[0:128, 0:128]
    image = np.stack(((x + y) & 255, (x * 2 + y) & 255,
                      (x + y * 2) & 255), axis=-1).astype(np.uint8)
    codec, transform, payload = v4._predictor_candidate(
        image.tobytes(), 128, 128, 3, 6, "cpu", parallel_transforms=False)
    assert codec == v4.CODEC_PRED2D_SIGNED
    assert len(payload) > 4
    filtered_size = int.from_bytes(payload[:4], "little")
    assert filtered_size == 128 * (128 * 3 + 3)


def test_adaptive_tile_partition_is_valid_and_splits_heterogeneous_regions():
    rng = np.random.default_rng(8810)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, 128:, :] = rng.integers(0, 256, size=(256, 128, 3), dtype=np.uint8)
    regions = v4._adaptive_tile_regions(image.tobytes(), 256, 256, 3, 256, enabled=True)
    assert len(regions) > 1
    v4._validate_v8_partition(regions, 256, 256)
    fixed = v4._adaptive_tile_regions(image.tobytes(), 256, 256, 3, 256, enabled=False)
    assert fixed == [(0, 0, 256, 256)]


def test_v8_parallel_reference_groups_roundtrip():
    rng = np.random.default_rng(8811)
    image = rng.integers(0, 256, size=(192, 192, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src, packed, restored = root / 'source.png', root / 'image.pxl', root / 'restored.png'
        Image.fromarray(image, mode='RGB').save(src)
        encoded = pxl.compress(src, packed, tile_size=64, effort='balanced', backend='cpu',
                               workers=1, adaptive_tiles=False, checkpoint_interval=2)
        assert encoded['version'] == 8
        assert encoded['tiles'] == 9
        decoded = pxl.decompress(packed, restored, backend='cpu', workers=3)
        assert decoded['version'] == 8
        assert decoded['parallel_groups'] >= 4
        assert np.array_equal(np.asarray(Image.open(restored).convert('RGB')), image)


def test_v8_adaptive_layout_roundtrip_and_coordinates():
    rng = np.random.default_rng(8812)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, 128:, :] = rng.integers(0, 256, size=(256, 128, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src, packed, restored = root / 'source.png', root / 'image.pxl', root / 'restored.png'
        Image.fromarray(image, mode='RGB').save(src)
        encoded = pxl.compress(src, packed, tile_size=256, backend='cpu', workers=1,
                               adaptive_tiles=True, checkpoint_interval=2)
        assert encoded['tiles'] > 1
        pxl.decompress(packed, restored, backend='cpu', workers=3)
        assert np.array_equal(np.asarray(Image.open(restored).convert('RGB')), image)


def test_v8_python_fallback_roundtrip_keeps_pixel_history_in_sync():
    from unittest.mock import patch
    rng = np.random.default_rng(8813)
    patch_image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    image = np.concatenate([patch_image, patch_image, patch_image], axis=1)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src, packed, restored = root / 'source.png', root / 'image.pxl', root / 'restored.png'
        Image.fromarray(image, mode='RGB').save(src)
        with patch('pxl_native.get_backend', return_value=None):
            pxl.compress(src, packed, tile_size=64, backend='cpu', workers=1,
                         adaptive_tiles=False, checkpoint_interval=8)
            pxl.decompress(packed, restored, backend='cpu', workers=1)
        assert np.array_equal(np.asarray(Image.open(restored).convert('RGB')), image)


def test_legacy_v7_fixture_still_decodes():
    fixture_dir = Path(__file__).parent
    packed = fixture_dir / "fixture_v7.pxl"
    expected = fixture_dir / "fixture_v7_expected.png"
    with tempfile.TemporaryDirectory() as td:
        restored = Path(td) / "restored.png"
        info = pxl.inspect(packed)
        assert info["version"] == 7
        result = pxl.decompress(packed, restored, backend="cpu", workers=2)
        assert result["version"] == 7
        with Image.open(expected) as a, Image.open(restored) as b:
            assert np.array_equal(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB")))


def test_v8_decoder_rejects_tile_payload_larger_than_decoded_pixels(tmp_path):
    source = tmp_path / "source.png"
    packed = tmp_path / "image.pxl"
    corrupted = tmp_path / "oversized-tile.pxl"
    restored = tmp_path / "restored.png"
    Image.new("RGB", (64, 64), (12, 34, 56)).save(source)
    pxl.compress(source, packed, tile_size=64, backend="cpu", workers=1,
                 adaptive_tiles=False)

    data = bytearray(packed.read_bytes())
    offset = v4.HEADER.size
    fields = list(v4.V8_TILE_HEADER.unpack_from(data, offset))
    fields[6] = fields[7] + 1  # payload_size = decoded_size + 1
    data[offset:offset + v4.V8_TILE_HEADER.size] = v4.V8_TILE_HEADER.pack(*fields)
    corrupted.write_bytes(data)

    with pytest.raises(ValueError, match="payload exceeds decoded tile size"):
        pxl.decompress(corrupted, restored, backend="cpu", workers=1)


def test_v8_decoder_rejects_reference_groups_longer_than_encoder_limit(tmp_path):
    source = tmp_path / "source.png"
    packed = tmp_path / "image.pxl"
    corrupted = tmp_path / "oversized-group.pxl"
    restored = tmp_path / "restored.png"
    # 17 x 17 = 289 fixed 64-pixel tiles, so a valid encoder writes a new
    # independent group after the first 256 tiles.
    Image.new("RGB", (1088, 1088), (18, 27, 36)).save(source)
    result = pxl.compress(source, packed, tile_size=64, backend="cpu", workers=1,
                          adaptive_tiles=False, checkpoint_interval=256)
    assert result["tiles"] == 289

    data = bytearray(packed.read_bytes())
    offset = v4.HEADER.size
    for index in range(result["tiles"]):
        fields = list(v4.V8_TILE_HEADER.unpack_from(data, offset))
        payload_size = fields[6]
        if index == 256:
            fields[5] &= ~0x40  # remove the second group's checkpoint marker
            data[offset:offset + v4.V8_TILE_HEADER.size] = v4.V8_TILE_HEADER.pack(*fields)
            break
        offset += v4.V8_TILE_HEADER.size + payload_size
    else:
        raise AssertionError("Did not find the second reference group")

    corrupted.write_bytes(data)
    with pytest.raises(ValueError, match="256-tile safety limit"):
        pxl.decompress(corrupted, restored, backend="cpu", workers=1)
