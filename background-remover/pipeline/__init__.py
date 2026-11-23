"""Background removal pipeline: decode → rembg → edge refine → encode."""

from .process import process_file, process_batch

__all__ = ["process_file", "process_batch"]
