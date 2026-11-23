"""Decode stills, GIF, and MP4 into RGBA frame sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
    """Load GIF frames as RGBA.

    Transparent palette pixels often keep a chroma-key RGB (magenta/green).
    Zero that RGB under alpha=0 so rembg/encoders never paint purple fringes.
    """
    frames: List[np.ndarray] = []
    durations: List[int] = []
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        for i in range(n):
            im.seek(i)
            arr = np.array(im.convert("RGBA"), dtype=np.uint8)
            transparent = arr[..., 3] < 8
            arr[transparent, 0:3] = 0
            frames.append(arr)
            durations.append(int(im.info.get("duration", 100) or 100))
    return frames, durations


def _decode_video_imageio(path: Path) -> Tuple[List[np.ndarray], float]:
    """imageio v3: plugin id is 'FFMPEG' (not 'ffmpeg')."""
    import imageio.v3 as iio

    errors: list[str] = []
    for plugin in (None, "FFMPEG"):
        try:
            kwargs = {} if plugin is None else {"plugin": plugin}
            meta = iio.immeta(path, **kwargs) or {}
            fps = float(meta.get("fps") or 24.0)
            raw = list(iio.imiter(path, **kwargs))
            if not raw:
                raise RuntimeError("0 frames")
            return [_to_rgba(f) for f in raw], fps
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{plugin or 'auto'}: {exc}")
    raise RuntimeError("; ".join(errors))


def _decode_video_cv2(path: Path) -> Tuple[List[np.ndarray], float]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(_to_rgba(rgb))
    cap.release()
    if not frames:
        raise RuntimeError("OpenCV read 0 frames")
    return frames, fps


def _decode_video_ffmpeg_cli(path: Path) -> Tuple[List[np.ndarray], float]:
    """Last-resort: extract PNG frames with system/imageio-ffmpeg binary."""
    import shutil
    import subprocess
    import tempfile

    import imageio_ffmpeg

    ff = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    # Probe fps
    probe = subprocess.run(
        [
            ff,
            "-i",
            str(path),
            "-hide_banner",
        ],
        capture_output=True,
        text=True,
    )
    fps = 24.0
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", probe.stderr or "")
    if m:
        fps = float(m.group(1))

    with tempfile.TemporaryDirectory(prefix="bgrem-vid-") as tmp:
        pattern = str(Path(tmp) / "f_%06d.png")
        proc = subprocess.run(
            [
                ff,
                "-y",
                "-i",
                str(path),
                "-vsync",
                "0",
                pattern,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-1500:])
        paths = sorted(Path(tmp).glob("f_*.png"))
        frames = [_to_rgba(np.array(Image.open(p).convert("RGB"))) for p in paths]
        if not frames:
            raise RuntimeError("ffmpeg extracted 0 frames")
        return frames, fps


def decode_video(path: Path) -> DecodedMedia:
    errors: list[str] = []
    frames: Optional[List[np.ndarray]] = None
    fps = 24.0
    for loader in (_decode_video_imageio, _decode_video_cv2, _decode_video_ffmpeg_cli):
        try:
            frames, fps = loader(path)
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader.__name__}: {exc}")
    if frames is None:
        raise RuntimeError(
            "Impossible de lire la vidéo. Détails: " + " | ".join(errors)
        )
    duration = int(round(1000.0 / max(fps, 1e-3)))
    return DecodedMedia(frames, [duration] * len(frames), "video", fps, path)


def decode_media(path: Path) -> DecodedMedia:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in GIF_EXTS:
        frames, durations = _pil_gif_frames(path)
        fps = 1000.0 / max(1.0, float(np.median(durations)))
        return DecodedMedia(frames, durations, "gif", fps, path)

    if ext in VIDEO_EXTS:
        return decode_video(path)

    if ext in STILL_EXTS:
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1)
            if n > 1 and ext == ".webp":
                frames = []
                durations = []
                for i in range(n):
                    im.seek(i)
                    arr = np.array(im.convert("RGBA"), dtype=np.uint8)
                    transparent = arr[..., 3] < 8
                    arr[transparent, 0:3] = 0
                    frames.append(arr)
                    durations.append(int(im.info.get("duration", 100) or 100))
                fps = 1000.0 / max(1.0, float(np.median(durations)))
                return DecodedMedia(frames, durations, "gif", fps, path)
            frame = np.array(im.convert("RGBA"), dtype=np.uint8)
        return DecodedMedia([frame], [100], "still", 1.0, path)

    raise ValueError(f"Unsupported format: {ext}")
