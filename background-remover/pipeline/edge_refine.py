"""Edge-local soft refine for near-white backgrounds.

Soft-thresholds only near the rembg mask boundary so light-gray body
highlights stay opaque, while thin cables/fingers are preserved via
morphological dilation of the AI mask before refine.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RefineSettings:
    edge_sensitivity: float = 0.55  # 0..1 — higher = more aggressive near-white cut
    keep_floor_shadow: bool = True
    edge_band_px: int = 10
    white_luma: float = 245.0
    morph_preserve_px: int = 2


def _luma(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _chroma(rgb: np.ndarray) -> np.ndarray:
    """Rough chroma: max-min channel distance; low on white/gray."""
    mx = rgb.max(axis=-1).astype(np.float32)
    mn = rgb.min(axis=-1).astype(np.float32)
    return mx - mn


def refine_mask(
    rgba: np.ndarray,
    mask: np.ndarray,
    settings: RefineSettings,
) -> np.ndarray:
    """Refine rembg mask; returns float alpha in [0, 1]."""
    h, w = mask.shape[:2]
    rgb = rgba[..., :3]
    luma = _luma(rgb)
    chroma = _chroma(rgb)

    # Protect thin structures: lightly dilate AI mask before edge work.
    k = max(1, settings.morph_preserve_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    preserved = cv2.dilate(mask_u8, kernel, iterations=1).astype(np.float32) / 255.0
    working = np.maximum(mask, preserved * 0.35)

    # Edge band around mask boundary.
    band = max(2, int(settings.edge_band_px))
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    solid = (working > 0.35).astype(np.uint8) * 255
    dil = cv2.dilate(solid, band_k, iterations=1)
    ero = cv2.erode(solid, band_k, iterations=1)
    edge_band = ((dil > 0) & (ero == 0)).astype(np.float32)

    # Interior core: never soft-cut (protect light-gray highlights).
    core_r = max(3, band // 2)
    core_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_r + 1, 2 * core_r + 1))
    core = cv2.erode(solid, core_k, iterations=1).astype(np.float32) / 255.0

    # Soft near-white probability (high only for bright + low chroma).
    sens = float(np.clip(settings.edge_sensitivity, 0.0, 1.0))
    # Higher sensitivity → lower luma threshold → more background cut.
    luma_thr = settings.white_luma - sens * 35.0  # ~245..210
    chroma_thr = 18.0 + (1.0 - sens) * 20.0

    white_score = np.clip((luma - (luma_thr - 20.0)) / 20.0, 0.0, 1.0)
    low_chroma = np.clip(1.0 - chroma / max(chroma_thr, 1.0), 0.0, 1.0)
    bg_likeness = white_score * low_chroma

    # Soft cut only in edge band; leave core untouched.
    cut = bg_likeness * edge_band * (0.45 + 0.55 * sens)
    refined = working * (1.0 - cut)

    # Restore core opacity fully.
    refined = np.where(core > 0.5, np.maximum(refined, mask), refined)

    # Outside solid dilated mask: force transparent unless weakly preserved.
    outside = dil == 0
    refined = np.where(outside, refined * 0.05, refined)

    if settings.keep_floor_shadow:
        # Soft dark-ish region near bottom that rembg partially kept → keep.
        bottom = np.zeros((h, w), dtype=np.float32)
        y0 = int(h * 0.72)
        bottom[y0:, :] = 1.0
        shadow = (
            (luma < 200)
            & (luma > 40)
            & (chroma < 40)
            & (mask > 0.15)
            & (bottom > 0)
        )
        refined = np.where(shadow, np.maximum(refined, mask * 0.85), refined)
    else:
        # Extra pass: remove soft floor spill in bottom band.
        y0 = int(h * 0.78)
        floor = np.zeros((h, w), dtype=bool)
        floor[y0:, :] = True
        floor_spill = floor & (bg_likeness > 0.35) & (mask < 0.85)
        refined = np.where(floor_spill, refined * (1.0 - bg_likeness), refined)

    # Light temporal-friendly smooth (small bilateral-like blur on mask).
    refined_u8 = (np.clip(refined, 0, 1) * 255).astype(np.uint8)
    refined_u8 = cv2.GaussianBlur(refined_u8, (3, 3), 0)
    refined = refined_u8.astype(np.float32) / 255.0

    # Re-boost thin high-confidence rembg strands (cables).
    thin = (mask > 0.55) & (core < 0.5)
    refined = np.where(thin, np.maximum(refined, mask), refined)

    return np.clip(refined, 0.0, 1.0)


def decontaminate_edges(rgba: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove dark/light BG bleed from semi-transparent edge pixels (CG mattes)."""
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    rgb = rgba[..., :3].astype(np.float32)
    # Solid foreground estimate: erode high-alpha region
    solid = (a > 0.92).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    solid = cv2.erode(solid, k, iterations=1)
    # Nearest solid color via blur of masked RGB (cheap approx)
    w = solid.astype(np.float32) / 255.0
    w3 = w[..., None]
    blurred = cv2.GaussianBlur(rgb * w3, (0, 0), 2.0)
    wblur = cv2.GaussianBlur(w, (0, 0), 2.0).astype(np.float32)
    wblur3 = np.maximum(wblur[..., None], 1e-3)
    fg = np.where(wblur3 > 1e-3, blurred / wblur3, rgb)

    edge = (a > 0.02) & (a < 0.95)
    out_rgb = rgb.copy()
    out_rgb[edge] = fg[edge]
    # Harden nearly-transparent fringe to fully clear (kills black halo)
    a2 = a.copy()
    a2[a2 < 0.12] = 0.0
    a2[a2 > 0.88] = 1.0
    out = rgba.copy()
    out[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out[..., 3] = (np.clip(a2, 0, 1) * 255).astype(np.uint8)
    transparent = out[..., 3] == 0
    out[transparent, 0:3] = 0
    return out


def apply_alpha(rgba: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = rgba.copy()
    out[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    # Zero RGB where fully transparent to keep encoders happy.
    transparent = out[..., 3] == 0
    out[transparent, 0:3] = 0
    return out
