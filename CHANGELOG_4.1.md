# PXL Studio 4.1

- Added a dedicated Batch Queue tab for mixed image-to-PXL and PXL-to-image jobs.
- Added per-file determinate progress bars, stage messages, elapsed time and ETA.
- Added weighted aggregate batch progress and queued/running/complete/failed/skipped counts.
- Added process-isolated parallel file jobs with configurable concurrency.
- Added pause/resume, remove queued task, clear finished tasks, common output folder,
  duplicate destination protection and opt-in overwrite.
- Queue writes to a same-directory temporary output then atomically replaces the
  destination after a successful encode/decode.
- CUDA batch work is serialized to one GPU worker to reduce VRAM contention.
- Existing PXL format and command-line codec APIs are unchanged.
