# PXL Studio 6.0 validation report

## Automated tests

- 51 passed, 1 skipped.
- The skipped test is the optional CUDA hardware test; no supported CUDA
  device was available in this environment.
- Native C++17 backend rebuilt successfully with ABI 4 on Linux.
- Native and pure-Python fallback decode paths tested for the new 65-context
  stream.
- Existing PXL v4 and v5 fixtures decoded successfully.
- All five benchmark images reconstructed byte-exactly in balanced and maximum
  profiles.

## Scope and limitations

The benchmark corpus contains five 512x512 synthetic RGB images. The results
are useful for regression testing but do not establish competitiveness on a
representative set of camera photographs. Windows MSVC and macOS builds were
not available for validation in this run. Windows users upgrading from 5.0
must rebuild the native library because the C ABI changed to version 4.
