#!/usr/bin/env python3
"""Generate the five 512x512 RGB synthetic images used by the PXL 4.2 report."""
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)
H = W = 512
y, x = np.mgrid[:H, :W]

# Smooth modular gradient.
gradient = np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8)
Image.fromarray(gradient, "RGB").save(OUT / "gradient.png")

# Small random patch repeated over the complete image.
rng = np.random.default_rng(4)
patch = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
texture = np.tile(patch, (16, 16, 1))
Image.fromarray(texture, "RGB").save(OUT / "texture.png")

# Deterministic smooth colour field with additive Gaussian noise.
rng = np.random.default_rng(5)
base = np.stack((x / 2 + y / 5, y / 2 + x / 8, x / 3 + y / 3), axis=2)
photo_like = np.clip(base + rng.normal(0, 15, (H, W, 3)), 0, 255).astype(np.uint8)
Image.fromarray(photo_like, "RGB").save(OUT / "photo_like.png")

# Incompressible uniformly distributed RGB.
noise = np.random.default_rng(1).integers(0, 256, (H, W, 3), dtype=np.uint8)
Image.fromarray(noise, "RGB").save(OUT / "random_noise.png")

# Large solid-colour blocks.
blocks = np.zeros((H, W, 3), dtype=np.uint8)
for by in range(0, H, 32):
    for bx in range(0, W, 32):
        blocks[by:by+32, bx:bx+32] = [(bx * 3) % 256, (by * 5) % 256, ((bx + by) * 2) % 256]
Image.fromarray(blocks, "RGB").save(OUT / "block_mosaic.png")
print(f"Wrote synthetic test images to {OUT}")
