# PXL Studio 4.2 performance and Codec Lab results

## A. v4.1 → v4.2 implementation-preserving speed test

Five deterministic 512×512 RGB synthetic images were encoded and decoded
with tile size 256, balanced effort and one CPU worker. v4.2 timings were
measured after JIT warm-up. Both versions ran in the same Linux container
(3 logical CPUs exposed). Times below are one recorded run, not a multi-run
median; use the scripts for measurements on your hardware.

| Image | v4.1 size | v4.2 size | v4.1 encode | v4.2 encode | Encode speedup | v4.1 decode | v4.2 decode | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient | 16,220 B | 16,220 B | 2,877 ms | 325 ms | 8.9× | 674 ms | 44 ms | 15.4× |
| Repeated texture | 4,955 B | 4,955 B | 3,239 ms | 691 ms | 4.7× | 186 ms | 35 ms | 5.4× |
| Photo-like synthetic | 559,937 B | 559,937 B | 4,469 ms | 1,086 ms | 4.1× | 860 ms | 133 ms | 6.5× |
| Random noise | 786,524 B | 786,524 B | 81 ms | 124 ms | 0.65× | 221 ms | 178 ms | 1.25× |
| Block mosaic | 460 B | 460 B | 3,179 ms | 661 ms | 4.8× | 155 ms | 86 ms | 1.8× |

The five PXL v4.1 and v4.2 output files were **byte-for-byte identical**.
Therefore, on these tests v4.2 preserves the exact compression ratio and
representation choices rather than trading compression quality for speed.
Random noise bypasses most expensive work already, so JIT overhead and system
timing variance can make that case slower; the encoder does not claim a speedup
for every input.

## B. Codec Lab: v4.2 vs available lossless codecs

The following figures are from the included Codec Lab run, with a shared
normalized PNG source, tile=256, balanced effort, one CPU worker. Encoder and
decoder times include the benchmark's local file workflow. Every row returned
exact pixels. JPEG XL was not available in this environment because the
optional Pillow JXL plugin was not installed.

| Image | Codec | Size | Encode | Decode |
|---|---|---:|---:|---:|
| Gradient | PXL v4.2 | 16,220 B | 959 ms | 134 ms |
| Gradient | WebP lossless | 156 B | 143 ms | 10 ms |
| Gradient | PNG | 1,813 B | 14 ms | 16 ms |
| Repeated texture | PXL v4.2 | 4,955 B | 786 ms | 52 ms |
| Repeated texture | WebP lossless | 12,122 B | 346 ms | 11 ms |
| Repeated texture | PNG | 55,167 B | 15 ms | 11 ms |
| Photo-like | PXL v4.2 | 559,937 B | 1,141 ms | 146 ms |
| Photo-like | WebP lossless | 540,692 B | 565 ms | 59 ms |
| Photo-like | PNG | 583,686 B | 72 ms | 61 ms |
| Random noise | PXL v4.2 | 786,524 B | 41 ms | 93 ms |
| Random noise | WebP lossless | 786,512 B | 65 ms | 50 ms |
| Random noise | PNG | 788,088 B | 54 ms | 62 ms |
| Block mosaic | PXL v4.2 | 460 B | 662 ms | 77 ms |
| Block mosaic | WebP lossless | 242 B | 76 ms | 6 ms |
| Block mosaic | PNG | 2,034 B | 8 ms | 6 ms |

**Interpretation:** the v4.2 hot-path work makes PXL much faster than its own
v4.1 implementation without changing the bitstream. It does not make PXL faster
than mature optimized codecs in general. WebP/PNG still have a major speed
advantage on many of these files, and WebP also compresses several patterns
better. PXL's custom architecture remains experimental.

## Reproduction

From the project root:

```bash
python benchmarks_v42/generate_synthetic.py
python benchmarks_v42/run_codec_lab.py
```

To reproduce the v4.1/v4.2 comparison, use a PXL 4.1 project checkout and this
PXL 4.2 project, run both on the generated PNGs with `effort="balanced"`,
`tile_size=256`, and `workers=1`. Warm PXL 4.2 once before recording timing.
