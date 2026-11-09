# PXL Studio 4.1 verification

Verification run for this package:

- `python -m pytest -q tests`: **34 passed, 1 skipped**.
- `python -m py_compile pxl.py pxl4_engine.py pxl_gui.py pxl_batch.py codec_compare.py`: passed.
- Tkinter GUI initialization under a virtual X server: passed.
- End-to-end batch smoke test under Tkinter + spawned process workers:
  - two image-to-PXL jobs completed with concurrency set to two;
  - one PXL-to-PNG decode job completed;
  - output files were created in the selected output directory.

The skipped test is the optional CUDA hardware check; no usable CUDA device was
available in the verification environment. This does not affect CPU batch mode.

## Batch processing notes

The GUI runs queued files in separate `ProcessPoolExecutor` workers using the
`spawn` start method. Each task uses one inner PXL CPU worker to avoid nested
process pools. Independent files can therefore use different CPU cores. CUDA
queue work is limited to one active file at a time. Progress is reported from
PXL's tile callbacks through a multiprocessing manager queue and displayed in
per-file bars plus a pixel-weighted aggregate bar.
