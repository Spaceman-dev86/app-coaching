"""Orchestrate decode → rembg → edge refine → encode for one or many files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from .decode import DecodedMedia, decode_media
from .edge_refine import RefineSettings, apply_alpha, refine_mask
from .encode import (
    choose_output_path,
    encode_animated_webp,
    encode_png,
    encode_webm_vp9_alpha,
)
from .rembg_mask import Masker

ProgressCb = Callable[[float, str], None]


def _temporal_smooth(
    masks: Sequence[np.ndarray],
    blend: float = 0.35,
) -> List[np.ndarray]:
    """Reduce GIF flicker by blending each mask with neighbors."""
    if len(masks) <= 1:
        return list(masks)
    out: List[np.ndarray] = []
    n = len(masks)
    for i in range(n):
        cur = masks[i].astype(np.float32)
        prev = masks[i - 1].astype(np.float32) if i > 0 else cur
        nxt = masks[i + 1].astype(np.float32) if i < n - 1 else cur
        smoothed = (1.0 - blend) * cur + 0.5 * blend * prev + 0.5 * blend * nxt
        out.append(np.clip(smoothed, 0.0, 1.0))
    return out


def process_file(
    path: Path | str,
    output_dir: Path | str = "./output",
    edge_sensitivity: float = 0.55,
    keep_floor_shadow: bool = True,
    masker: Optional[Masker] = None,
    progress: Optional[ProgressCb] = None,
) -> Path:
    path = Path(path)
    output_dir = Path(output_dir)
    if progress:
        progress(0.02, f"Decoding {path.name}")

    media: DecodedMedia = decode_media(path)
    masker = masker or Masker()
    settings = RefineSettings(
        edge_sensitivity=edge_sensitivity,
        keep_floor_shadow=keep_floor_shadow,
    )

    raw_masks: List[np.ndarray] = []
    n = len(media.frames)
    for i, frame in enumerate(media.frames):
        if progress:
            # rembg is the heavy part — map to ~5–80%
            progress(0.05 + 0.75 * (i / max(n, 1)), f"Masking frame {i + 1}/{n}")
        raw_masks.append(masker.alpha_mask(frame))

    if media.kind in ("gif", "video") and n > 1:
        raw_masks = _temporal_smooth(raw_masks, blend=0.30)

    out_frames: List[np.ndarray] = []
    for i, (frame, mask) in enumerate(zip(media.frames, raw_masks)):
        if progress:
            progress(0.80 + 0.12 * (i / max(n, 1)), f"Refining frame {i + 1}/{n}")
        alpha = refine_mask(frame, mask, settings)
        out_frames.append(apply_alpha(frame, alpha))

    dest = choose_output_path(path, media.kind, output_dir)
    if progress:
        progress(0.93, f"Encoding {dest.name}")

    if media.kind == "still":
        encode_png(out_frames[0], dest)
    elif media.kind == "gif":
        encode_animated_webp(out_frames, media.durations_ms, dest)
    else:
        encode_webm_vp9_alpha(out_frames, media.fps, dest)

    if progress:
        progress(1.0, f"Done: {dest.name}")
    return dest


def process_batch(
    paths: Sequence[Path | str],
    output_dir: Path | str = "./output",
    edge_sensitivity: float = 0.55,
    keep_floor_shadow: bool = True,
    make_zip: bool = True,
    progress: Optional[ProgressCb] = None,
) -> tuple[List[Path], Optional[Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masker = Masker()
    results: List[Path] = []
    total = max(len(paths), 1)

    for idx, p in enumerate(paths):
        p = Path(p)

        def file_progress(frac: float, msg: str, _idx=idx) -> None:
            if progress:
                overall = (_idx + frac) / total
                progress(overall, f"[{_idx + 1}/{total}] {msg}")

        results.append(
            process_file(
                p,
                output_dir=output_dir,
                edge_sensitivity=edge_sensitivity,
                keep_floor_shadow=keep_floor_shadow,
                masker=masker,
                progress=file_progress,
            )
        )

    zip_path: Optional[Path] = None
    if make_zip and results:
        zip_path = output_dir / "cutouts.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                zf.write(r, arcname=r.name)
        if progress:
            progress(1.0, f"Zip ready: {zip_path.name}")

    return results, zip_path
