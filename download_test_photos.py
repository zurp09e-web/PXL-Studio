#!/usr/bin/env python3
"""Download the original-resolution, attributed Wikimedia Commons test photos."""
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent / "test_photos"
FILES = {
    "rocky_mountains_rind.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/5/5a/Landscape_in_the_rocky_mountains_of_Rind_06.jpg",
    "new_york_night.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/1/10/New_York_City%2C_night.jpg",
    "tatra_gubalowka.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/7/78/Tatra_mountains_view_from_Gubalowka_mountain.jpg",
}

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = ROOT / name
        if target.exists() and target.stat().st_size > 100_000:
            print(f"Already present: {name} ({target.stat().st_size:,} bytes)")
            continue
        print(f"Downloading {name} ...")
        request = Request(url, headers={"User-Agent": "PXL-Studio-test-corpus/2.3 (benchmark data)"})
        with urlopen(request, timeout=90) as response, target.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if target.stat().st_size < 100_000:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Download for {name} appears incomplete")
        print(f"Saved {name}: {target.stat().st_size:,} bytes")
    print("Photo corpus ready. Run: python benchmark_photos.py --mode lossless")

if __name__ == "__main__":
    main()
