"""Decode stills, GIF, and MP4 into RGBA frame sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import imageio.v3 as iio
import numpy as np
from PIL import Image

STILL_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
GIF_EXTS = {".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class DecodedMedia:
    frames: List[np.ndarray]  # RGBA uint8
    durations_ms: List[int]
    kind: str  # "still" | "gif" | "video"
    fps: float
    source_path: Path


def _to_rgba(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
        alpha = np.full(arr.shape, 255, dtype=np.uint8)
        return np.concatenate([rgb, alpha[..., None]], axis=-1)
    if arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate([arr.astype(np.uint8), alpha], axis=-1)
    if arr.shape[-1] == 4:
        return arr.astype(np.uint8)
    raise ValueError(f"Unsupported channel count: {arr.shape}")


def _pil_gif_frames(path: Path) -> Tuple[List[np.ndarray], List[int]]:
    frames: List[np.ndarray] = []
    durations: List[int] = []
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        for i in range(n):
            im.seek(i)
            frame = im.convert("RGBA")
            frames.append(np.array(frame, dtype=np.uint8))
            durations.append(int(im.info.get("duration", 100) or 100))
    return frames, durations


def decode_media(path: Path) -> DecodedMedia:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in GIF_EXTS:
        frames, durations = _pil_gif_frames(path)
        fps = 1000.0 / max(1.0, float(np.median(durations)))
        return DecodedMedia(frames, durations, "gif", fps, path)

    if ext in VIDEO_EXTS:
        # Prefer ffmpeg plugin (imageio-ffmpeg); fall back to pyav if present.
        plugin = "ffmpeg"
        try:
            meta = iio.immeta(path, plugin=plugin) or {}
        except Exception:
            plugin = "pyav"
            meta = iio.immeta(path, plugin=plugin) or {}
        fps = float(meta.get("fps") or 24.0)
        raw = list(iio.imiter(path, plugin=plugin))
        frames = [_to_rgba(f) for f in raw]
        duration = int(round(1000.0 / max(fps, 1e-3)))
        return DecodedMedia(frames, [duration] * len(frames), "video", fps, path)

    if ext in STILL_EXTS:
        # Prefer Pillow so animated WebP stills land as first-frame stills when desired;
        # for animated WebP treat as gif-like sequence.
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1)
            if n > 1 and ext == ".webp":
                frames = []
                durations = []
                for i in range(n):
                    im.seek(i)
                    frames.append(np.array(im.convert("RGBA"), dtype=np.uint8))
                    durations.append(int(im.info.get("duration", 100) or 100))
                fps = 1000.0 / max(1.0, float(np.median(durations)))
                return DecodedMedia(frames, durations, "gif", fps, path)
            frame = np.array(im.convert("RGBA"), dtype=np.uint8)
        return DecodedMedia([frame], [100], "still", 1.0, path)

    raise ValueError(f"Unsupported format: {ext}")
