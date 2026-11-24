"""Orchestrate decode → rembg → edge refine → encode for one or many files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .decode import DecodedMedia, decode_media
from .edge_refine import (
    RefineSettings,
    apply_alpha,
    decontaminate_edges,
    refine_mask,
)
from .encode import (
    choose_output_path,
    encode_mp4_h264,
    encode_png,
)
from .flat_bg import (
    combine_studio_alpha,
    detect_flat_background,
    flood_flat_mask,
    soft_color_key,
)
from .rembg_mask import Masker

ProgressCb = Callable[[float, str], None]


def _temporal_smooth(
    masks: Sequence[np.ndarray],
    blend: float = 0.18,
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
    also_gif_compat: bool = False,  # kept for API compat; unused (GIF→MP4)
) -> Tuple[Path, List[Path]]:
    """Returns (primary_output, sidecar_outputs).

    Animated inputs (GIF / video) → MP4 H.264 for Supabase. Stills → PNG.
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
        raw_masks = _temporal_smooth(raw_masks, blend=0.18)

    out_frames: List[np.ndarray] = []
    for i, (frame, mask) in enumerate(zip(media.frames, raw_masks)):
        if progress:
            progress(0.80 + 0.12 * (i / max(n, 1)), f"Refining frame {i + 1}/{n}")

        sens = float(np.clip(edge_sensitivity, 0.0, 1.0))
        if flat_info is not None:
            mode, seed = flat_info
            # Slider = aggressivité du key studio (plus haut = coupe plus net, moins de halo)
            if mode == "black":
                hard = 4.0 + sens * 12.0  # ~4..16
                soft = hard + 6.0 + (1.0 - sens) * 10.0
                flood_tol = int(10 + sens * 14)
                bg_rgb = (0.0, 0.0, 0.0)
            else:
                hard = 12.0 + sens * 20.0
                soft = hard + 8.0 + (1.0 - sens) * 14.0
                flood_tol = int(20 + sens * 16)
                bg_rgb = (255.0, 255.0, 255.0)
            key_a = soft_color_key(frame, mode, seed, hard_tol=hard, soft_tol=soft)
            flood = flood_flat_mask(frame, mode, seed, tol=flood_tol)
            alpha = combine_studio_alpha(mask, key_a, flood)
            if mode == "white":
                alpha = refine_mask(frame, alpha, settings)
            # harden_lo up with sens → less dark fringe on white BG
            harden_lo = 0.04 + sens * 0.12
            harden_hi = 0.96 - sens * 0.06
            out_frames.append(
                decontaminate_edges(
                    frame,
                    alpha,
                    bg_rgb=bg_rgb,
                    harden_lo=harden_lo,
                    harden_hi=harden_hi,
                )
            )
        else:
            alpha = refine_mask(frame, mask, settings)
            # Generic: light decontam without known BG (still helps dual preview)
            out_frames.append(decontaminate_edges(frame, alpha, bg_rgb=None))

    sidecars: List[Path] = []

    # --- encode ---
    if media.kind == "still":
        dest = choose_output_path(path, media.kind, output_dir)
        if progress:
            progress(0.93, f"Encoding {dest.name}")
        encode_png(out_frames[0], dest)
        if progress:
            progress(1.0, f"Done: {dest.name}")
        return dest, sidecars

    if media.kind in ("gif", "video"):
        # GIF + vidéo → MP4 H.264 (animé, Supabase exercise-media OK)
        dest = output_dir / f"{path.stem}_cutout.mp4"
        if dest.suffix.lower() != ".mp4":
            raise RuntimeError(f"Animated cutout must be .mp4, got {dest}")
        if progress:
            progress(0.93, f"Encoding {dest.name} (MP4 H.264)")
        fps = float(media.fps) if media.fps and media.fps > 0 else 0.0
        if fps <= 0 and media.durations_ms:
            med = float(np.median(np.asarray(media.durations_ms, dtype=np.float64)))
            fps = 1000.0 / max(med, 20.0)
        fps = max(fps, 1.0)
        # MP4 has no alpha — composite on black (transparent → black)
        encode_mp4_h264(out_frames, fps, dest, bg_rgb=(0, 0, 0))
        if progress:
            progress(1.0, f"Done: {dest.name}")
        return dest, sidecars

    raise RuntimeError(f"Unhandled media kind: {media.kind}")


def process_batch(
    paths: Sequence[Path | str],
    output_dir: Path | str = "./output",
    edge_sensitivity: float = 0.55,
    keep_floor_shadow: bool = True,
    make_zip: bool = True,
    progress: Optional[ProgressCb] = None,
    also_gif_compat: bool = False,
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
            also_gif_compat=also_gif_compat,
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
