"""Encode RGBA sequences to PNG / animated WebP / WebM VP9+alpha."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image


def encode_png(frame: np.ndarray, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, mode="RGBA").save(dest, format="PNG", optimize=True)
    return dest


def encode_animated_webp(
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    dest: Path,
    quality: int = 90,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to encode")

    pil_frames = [Image.fromarray(f, mode="RGBA") for f in frames]
    durations = [max(20, int(d)) for d in durations_ms]
    # Pillow expects duration per frame; loop=0 means infinite.
    pil_frames[0].save(
        dest,
        format="WEBP",
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        lossless=False,
        quality=quality,
        method=4,
    )
    return dest


def encode_webm_vp9_alpha(
    frames: Sequence[np.ndarray],
    fps: float,
    dest: Path,
) -> Path:
    """Encode RGBA frames to WebM with VP9 + alpha via ffmpeg."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH (required for WebM VP9+alpha)")

    fps = max(float(fps), 1.0)
    with tempfile.TemporaryDirectory(prefix="bgrem-webm-") as tmp:
        tmp_path = Path(tmp)
        for i, frame in enumerate(frames):
            Image.fromarray(frame, mode="RGBA").save(tmp_path / f"frame_{i:06d}.png")

        pattern = str(tmp_path / "frame_%06d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            pattern,
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-auto-alt-ref",
            "0",
            "-b:v",
            "0",
            "-crf",
            "30",
            str(dest),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")
    return dest


def choose_output_path(source: Path, kind: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    if kind == "still":
        return output_dir / f"{stem}_cutout.png"
    if kind == "gif":
        return output_dir / f"{stem}_cutout.webp"
    if kind == "video":
        return output_dir / f"{stem}_cutout.webm"
    raise ValueError(f"Unknown kind: {kind}")
