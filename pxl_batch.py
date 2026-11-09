"""Process-isolated batch operations used by the PXL Studio GUI.

Each queued file is executed in its own process. The PXL encoder is configured
with one internal CPU worker in this mode so multiple files can use separate
cores without creating nested process pools and oversubscribing the machine.
"""
from __future__ import annotations

import os
from pathlib import Path

from pxl import compress, decompress


def execute_queued_job(job: dict, progress_queue=None) -> dict:
    """Run one queued encode/decode job and atomically publish its output.

    `progress_queue` is normally a multiprocessing.Manager queue proxy. It is
    deliberately optional so the function can also be unit-tested in-process.
    """
    job_id = str(job["id"])
    src = Path(job["src"])
    dst = Path(job["dst"])
    if not src.is_file():
        raise FileNotFoundError(f"Input file no longer exists: {src}")
    if dst.exists() and not bool(job.get("overwrite", False)):
        raise FileExistsError(f"Output already exists (overwrite is disabled): {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep the destination suffix on the temporary file: Pillow selects its
    # output image format from the suffix during PXL decoding.
    temp = dst.with_name(f".{dst.stem}.{job_id}.partial{dst.suffix}")
    if temp.exists():
        temp.unlink()

    def report(percent: int, message: str) -> None:
        if progress_queue is not None:
            try:
                progress_queue.put((job_id, int(percent), str(message)))
            except (BrokenPipeError, EOFError, OSError):
                # The GUI may have been closed while a job was finishing.
                pass

    backend = str(job.get("backend", "cpu"))
    try:
        if job["kind"] == "compress":
            result = compress(
                src,
                temp,
                mode=job.get("mode", "lossless"),
                quality=int(job.get("quality", 88)),
                tile_size=int(job.get("tile_size", 256)),
                progress=report,
                effort=job.get("effort", "balanced"),
                backend=backend,
                # One inner worker per process. Parallelism across files is
                # managed by the GUI's ProcessPoolExecutor.
                workers=1 if backend == "cpu" else None,
                adaptive_tiles=bool(job.get("adaptive_tiles", True)),
                checkpoint_interval=int(job.get("checkpoint_interval", 8)),
            )
        elif job["kind"] == "decompress":
            result = decompress(
                src,
                temp,
                progress=report,
                backend=backend,
                workers=1 if backend == "cpu" else None,
            )
        else:
            raise ValueError(f"Unknown queued operation: {job['kind']}")

        os.replace(temp, dst)
        result = dict(result)
        result["output_path"] = str(dst)
        result["output_bytes"] = dst.stat().st_size
        return result
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
