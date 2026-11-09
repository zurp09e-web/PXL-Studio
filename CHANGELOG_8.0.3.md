# PXL Studio 8.0.3 — decoder security hardening

## Security fixes

- Fixed an out-of-bounds write in the native PXL v8 blended-context rANS decoder. Dense and sparse frequency tables are now fully validated against the 4096-symbol total before the fixed-size decode lookup table is populated.
- Bound PXL v8 tile payloads to their decoded tile size and bound the sum of all tile payloads to the decoded image byte count.
- Reject PXL v8 reference-history groups longer than the encoder-supported maximum of 256 tiles.
- Bound the aggregate size of PXL v7 high-precision embedded byte-plane archives. Plane archives are copied to temporary files in 1 MiB chunks rather than retained together in memory.
- Revalidate the high-precision header and aggregate embedded lengths when the decoder reads the file.
- Verify each embedded byte-plane archive's dimensions, channels and lossless profile against the outer image before invoking its decoder.
- Reject overflowing 64-bit and 32-bit varint encodings before shifts can overflow in native and Python token readers.

## Regression coverage

- Native dense and sparse mixed-rANS tables whose frequencies total 4097 are rejected.
- PXL v8 archives with oversized tile payloads or more than 256 tiles in a reference group are rejected.
- PXL v7 archives whose combined byte-plane payloads exceed the decoded-image budget are rejected.
- Embedded PXL byte planes with mismatched geometry and overflowing native/Python token varints are rejected.

## Compatibility

Valid PXL v1–v8 archives remain supported. PXL v8 files produced by the encoder use at most 256 tiles per reference group and select each tile payload against the verbatim-pixel candidate, so the new decoder bounds match the writer's existing format invariants.

## Validation status

- The native C++ backend compiled successfully in the Linux CI environment.
- The targeted native malformed-input tests passed under AddressSanitizer and UndefinedBehaviorSanitizer (**4 passed**).
- The first complete-suite run reached **85 passed, 2 skipped, 1 failed**. The failure was in the pre-existing uint16 RGB PNG test fixture, which had left PyPNG in its default grayscale mode while supplying RGB rows. The fixture was corrected to set `greyscale=False`.
- A clean full-suite result after that fixture correction is still required before this patch can be considered fully verified.
