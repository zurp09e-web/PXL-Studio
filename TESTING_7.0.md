# PXL Studio 7.0 test notes

Run all tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Coverage includes exact 8-bit L/RGB/RGBA round trips; v4/v5 fixture decoding;
multicore encode determinism; lossy alpha preservation; context-mixing rANS
round trips; variable 8/16/32 block dictionary references; uint16 PNG/TIFF;
float16 RGBA TIFF; float32 HDR TIFF; exact float bit patterns; and rejection of
lossy high-precision requests.

Optional OpenEXR tests require `python -m pip install -r requirements-hdr.txt`
and an OpenEXR 3.3+ package. The current environment did not have OpenEXR
installed, so actual EXR file I/O must be verified on a machine with that
optional dependency. Windows/MSVC and CUDA builds also require validation on
the target machine.


## Known unvalidated paths and omissions

- OpenEXR import/export tests are skipped unless OpenEXR 3.3+ is installed.
- 16-bit RGB/RGBA PNG import/export tests are skipped unless PyPNG is installed.
  The base requirements include PyPNG; the current build environment did not
  have the module installed.
- No Windows/MSVC or CUDA hardware build was available for this release check.
- Codec Lab comparisons remain limited to normalized 8-bit sources.
- Adaptive tile partitioning, approximate/delta block references, parallel
  cross-tile reconstruction and new SIMD kernels are not implemented in 7.0.
