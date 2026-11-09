from pathlib import Path
from queue import SimpleQueue
import tempfile

from PIL import Image

from pxl_batch import execute_queued_job


def test_queued_compress_decode_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "source.png"
        packed = root / "source.pxl"
        restored = root / "restored.png"

        image = Image.new("RGB", (91, 73))
        pixels = image.load()
        for y in range(image.height):
            for x in range(image.width):
                pixels[x, y] = ((x * 9) & 255, (y * 13) & 255, ((x + y) * 7) & 255)
        image.save(src)

        messages = SimpleQueue()
        encoded = execute_queued_job({
            "id": "test-encode", "kind": "compress", "src": str(src),
            "dst": str(packed), "mode": "lossless", "quality": 88,
            "tile_size": 256, "effort": "fast", "backend": "cpu",
            "overwrite": False,
        }, messages)
        assert packed.is_file()
        assert encoded["output_bytes"] == packed.stat().st_size

        decoded = execute_queued_job({
            "id": "test-decode", "kind": "decompress", "src": str(packed),
            "dst": str(restored), "backend": "cpu", "overwrite": False,
        }, messages)
        assert restored.is_file()
        assert decoded["output_bytes"] == restored.stat().st_size
        with Image.open(src) as a, Image.open(restored) as b:
            assert a.convert("RGB").tobytes() == b.convert("RGB").tobytes()


def test_existing_output_is_not_overwritten_by_default():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "source.png"
        dst = root / "source.pxl"
        Image.new("RGB", (8, 8), (12, 34, 56)).save(src)
        dst.write_bytes(b"keep-me")
        try:
            execute_queued_job({
                "id": "test-overwrite", "kind": "compress", "src": str(src),
                "dst": str(dst), "backend": "cpu", "overwrite": False,
            })
        except FileExistsError:
            pass
        else:
            raise AssertionError("Expected FileExistsError")
        assert dst.read_bytes() == b"keep-me"
