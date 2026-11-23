"""Orchestrate decode → rembg → edge refine → encode for one or many files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .decode import DecodedMedia, decode_media
from .edge_refine import RefineSettings, apply_alpha, refine_mask
from .encode import (
    choose_output_path,
    encode_animated_gif,
    encode_animated_webp,
    encode_png,
    encode_webm_vp9_alpha,
)
from .flat_bg import combine_with_rembg, detect_flat_background, flood_flat_mask
from .rembg_mask import Masker

ProgressCb = Callable[[float, str], None]


def _temporal_smooth(
    masks: Sequence[np.ndarray],
    blend: float = 0.25,
) -> List[np.ndarray]:
    """Light neighbor blend — keep low to avoid mushy / choppy motion."""
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
    also_webp_pro: bool = True,
) -> Tuple[Path, List[Path]]:
    """Returns (primary_output, sidecar_outputs).

    For GIF inputs, primary is `.gif` and a `_pro.webp` (true soft alpha) is
    written alongside when also_webp_pro=True — that's the pro-quality file.
    """
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

    flat_info = detect_flat_background(media.frames[0])

    raw_masks: List[np.ndarray] = []
    n = len(media.frames)
    for i, frame in enumerate(media.frames):
        if progress:
            progress(0.05 + 0.75 * (i / max(n, 1)), f"Masking frame {i + 1}/{n}")
        raw_masks.append(masker.alpha_mask(frame))

    if media.kind in ("gif", "video") and n > 1:
        raw_masks = _temporal_smooth(raw_masks, blend=0.22)

    out_frames: List[np.ndarray] = []
    for i, (frame, mask) in enumerate(zip(media.frames, raw_masks)):
        if progress:
            progress(0.80 + 0.12 * (i / max(n, 1)), f"Refining frame {i + 1}/{n}")

        if flat_info is not None:
            mode, seed = flat_info
            tol = 24 if mode == "black" else 30
            flat = flood_flat_mask(frame, mode, seed, tol=tol)
            alpha = combine_with_rembg(mask, flat)
            # White studio: still run soft fringe refine. Black: skip (wrong target).
            if mode == "white":
                alpha = refine_mask(frame, alpha, settings)
        else:
            alpha = refine_mask(frame, mask, settings)

        out_frames.append(apply_alpha(frame, alpha))

    dest = choose_output_path(path, media.kind, output_dir)
    sidecars: List[Path] = []
    if progress:
        progress(0.93, f"Encoding {dest.name}")

    if media.kind == "still":
        encode_png(out_frames[0], dest)
    elif media.kind == "gif":
        if path.suffix.lower() == ".gif":
            encode_animated_gif(out_frames, media.durations_ms, dest)
            if also_webp_pro:
                pro = dest.with_name(f"{dest.stem}_pro.webp")
                encode_animated_webp(out_frames, media.durations_ms, pro)
                sidecars.append(pro)
        else:
            encode_animated_webp(out_frames, media.durations_ms, dest)
    else:
        encode_webm_vp9_alpha(out_frames, media.fps, dest)

    if progress:
        progress(1.0, f"Done: {dest.name}")
    return dest, sidecars


def process_batch(
    paths: Sequence[Path | str],
    output_dir: Path | str = "./output",
    edge_sensitivity: float = 0.55,
    keep_floor_shadow: bool = True,
    make_zip: bool = True,
    progress: Optional[ProgressCb] = None,
    also_webp_pro: bool = True,
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

        primary, sidecars = process_file(
            p,
            output_dir=output_dir,
            edge_sensitivity=edge_sensitivity,
            keep_floor_shadow=keep_floor_shadow,
            masker=masker,
            progress=file_progress,
            also_webp_pro=also_webp_pro,
        )
        results.append(primary)
        results.extend(sidecars)

    zip_path: Optional[Path] = None
    if make_zip and results:
        zip_path = output_dir / "cutouts.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                zf.write(r, arcname=r.name)
        if progress:
            progress(1.0, f"Zip ready: {zip_path.name}")

    return results, zip_path
