"""Encode RGBA sequences to PNG / animated GIF / animated WebP / WebM VP9+alpha."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def encode_png(frame: np.ndarray, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, mode="RGBA").save(dest, format="PNG", optimize=True)
    return dest


def _gif_duration_ms(ms: int) -> int:
    """Browsers treat GIF delays <20ms as ~100ms — clamp to avoid choppy playback."""
    d = int(ms) if ms and ms > 0 else 100
    return max(20, d)


def encode_animated_gif(
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    dest: Path,
) -> Path:
    """Animated GIF with real transparency via ffmpeg palettegen.

    Falls back to Pillow if ffmpeg is missing.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to encode")

    durations = [_gif_duration_ms(d) for d in durations_ms]
    if len(durations) < len(frames):
        durations += [durations[-1]] * (len(frames) - len(durations))

    if shutil.which("ffmpeg"):
        return _encode_gif_ffmpeg(frames, durations, dest)
    return _encode_gif_pillow(frames, durations, dest)


def _encode_gif_ffmpeg(
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    dest: Path,
) -> Path:
    """Constant-FPS GIF via palettegen (smooth; avoids concat extra-frame quirks)."""
    med = float(np.median(np.asarray(durations_ms, dtype=np.float64)))
    fps = 1000.0 / max(med, 20.0)

    with tempfile.TemporaryDirectory(prefix="bgrem-gif-") as tmp:
        tmp_path = Path(tmp)
        for i, frame in enumerate(frames):
            rgba = np.array(frame, copy=True)
            a = rgba[..., 3:4].astype(np.float32) / 255.0
            rgba[..., :3] = (rgba[..., :3].astype(np.float32) * a).astype(np.uint8)
            Image.fromarray(rgba, mode="RGBA").save(tmp_path / f"frame_{i:06d}.png")

        pattern = str(tmp_path / "frame_%06d.png")
        palette = tmp_path / "palette.png"
        gen = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            pattern,
            "-vf",
            "palettegen=reserve_transparent=1:stats_mode=full",
            str(palette),
        ]
        proc = subprocess.run(gen, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg palettegen failed:\n{proc.stderr[-2000:]}")

        use = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            pattern,
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse=alpha_threshold=64:dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(dest),
        ]
        proc = subprocess.run(use, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg paletteuse failed:\n{proc.stderr[-2000:]}")
    return dest


def _encode_gif_pillow(
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    dest: Path,
) -> Path:
    """Fallback: Pillow RGBA→GIF (binary alpha). Prefer ffmpeg path."""
    pil_frames = []
    for rgba in frames:
        im = Image.fromarray(rgba, mode="RGBA")
        # Composite on transparent; quantize with alpha
        alpha = im.getchannel("A")
        # Threshold → mask for transparency
        mask = alpha.point(lambda a: 255 if a < 64 else 0)
        rgb = Image.new("RGB", im.size, (0, 0, 0))
        rgb.paste(im, mask=alpha)
        pal = rgb.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        # Index 255 unused — remap transparent pixels to a free slot via paste
        # Use disposal + transparency via saving RGBA directly (Pillow ≥10)
        rgba_im = im.copy()
        rgba_im.putalpha(alpha.point(lambda a: 0 if a < 64 else 255))
        pil_frames.append(rgba_im)

    pil_frames[0].save(
        dest,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=list(durations_ms),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return dest


def encode_animated_webp(
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    dest: Path,
    quality: int = 92,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to encode")

    durations = [_gif_duration_ms(d) for d in durations_ms]
    if len(durations) < len(frames):
        durations += [durations[-1]] * (len(frames) - len(durations))

    # imageio tends to preserve per-frame timing more reliably than Pillow alone
    try:
        import imageio.v3 as iio

        # imageio wants duration in seconds for webp
        iio.imwrite(
            dest,
            [np.asarray(f) for f in frames],
            extension=".webp",
            plugin="pillow",
            duration=[d / 1000.0 for d in durations],
            loop=0,
            quality=quality,
            lossless=False,
        )
        return dest
    except Exception:  # noqa: BLE001
        pass

    pil_frames = [Image.fromarray(f, mode="RGBA") for f in frames]
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
    """Prefer keeping the source container when it can hold transparency."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    ext = source.suffix.lower()
    if kind == "still":
        return output_dir / f"{stem}_cutout.png"
    if kind == "gif":
        if ext == ".gif":
            return output_dir / f"{stem}_cutout.gif"
        return output_dir / f"{stem}_cutout.webp"
    if kind == "video":
        return output_dir / f"{stem}_cutout.webm"
    raise ValueError(f"Unknown kind: {kind}")
