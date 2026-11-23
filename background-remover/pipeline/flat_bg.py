"""Flat black/white studio keying for CG / anatomical GIFs.

For pure black/white backgrounds, color+flood keying beats rembg alone
(rembg fringes on illustrations). rembg still protects dark subject parts
(weight plates) that match the background color.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import cv2
import numpy as np

FlatMode = Literal["black", "white"]


def detect_flat_background(
    rgba: np.ndarray,
    *,
    sample: int = 6,
) -> Optional[Tuple[FlatMode, np.ndarray]]:
    """Return ('black'|'white', mean_rgb) if corners look like a flat studio BG."""
    h, w = rgba.shape[:2]
    s = max(1, sample)
    patches = [
        rgba[0:s, 0:s, :3],
        rgba[0:s, w - s : w, :3],
        rgba[h - s : h, 0:s, :3],
        rgba[h - s : h, w - s : w, :3],
    ]
    corners = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0).astype(
        np.float32
    )
    mean = corners.mean(axis=0)
    std = float(corners.std(axis=0).mean())
    if std > 14.0:
        return None
    luma = float(0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2])
    chroma = float(mean.max() - mean.min())
    # Also treat already-transparent corners (RGB zeroed) as black studio
    alpha_corner = float(rgba[0:s, 0:s, 3].mean())
    if alpha_corner < 8 and luma <= 25.0:
        return "black", mean
    if luma <= 18.0 and chroma <= 12.0:
        return "black", mean
    if luma >= 235.0 and chroma <= 18.0:
        return "white", mean
    return None


def soft_color_key(
    rgba: np.ndarray,
    mode: FlatMode,
    seed_rgb: np.ndarray,
    *,
    hard_tol: float = 14.0,
    soft_tol: float = 28.0,
) -> np.ndarray:
    """Return float alpha 0..1 (1=keep). Soft falloff from seed color."""
    rgb = rgba[..., :3].astype(np.float32)
    seed = seed_rgb.astype(np.float32)
    if mode == "black":
        # Luma-dominant key — better for pure black studios
        luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        # 0 at black, 1 when clearly not background
        alpha = np.clip((luma - hard_tol) / max(soft_tol - hard_tol, 1.0), 0.0, 1.0)
    else:
        dist = np.abs(rgb - seed).max(axis=-1)
        # Invert: close to white seed → transparent
        alpha = np.clip((dist - hard_tol) / max(soft_tol - hard_tol, 1.0), 0.0, 1.0)
    return alpha.astype(np.float32)


def flood_flat_mask(
    rgba: np.ndarray,
    mode: FlatMode,
    seed_rgb: np.ndarray,
    *,
    tol: int = 22,
) -> np.ndarray:
    """Float mask 1 = background to remove (border-connected flat color)."""
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.int16)
    seed = seed_rgb.astype(np.int16)
    dist = np.abs(rgb - seed).max(axis=-1)
    similar = (dist <= tol).astype(np.uint8)

    _num, labels = cv2.connectedComponents(similar, connectivity=8)
    border = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border = border[border > 0]
    if border.size == 0:
        return np.zeros((h, w), dtype=np.float32)

    filled = np.isin(labels, border).astype(np.float32)
    soft = cv2.GaussianBlur((filled * 255).astype(np.uint8), (3, 3), 0)
    return soft.astype(np.float32) / 255.0


def combine_studio_alpha(
    rembg_mask: np.ndarray,
    soft_key_alpha: np.ndarray,
    flood_bg: np.ndarray,
    *,
    rembg_protect: float = 0.50,
) -> np.ndarray:
    """Pro matte for flat studio: soft key + flood cut + rembg plate protect."""
    # Start from soft color key (good edges on CG)
    alpha = soft_key_alpha.astype(np.float32).copy()

    # Border-connected flat BG → force out (unless rembg strongly keeps)
    cut = flood_bg * (1.0 - np.clip((rembg_mask - rembg_protect) / 0.30, 0.0, 1.0))
    alpha = alpha * (1.0 - 0.92 * cut)

    force_out = (flood_bg > 0.75) & (rembg_mask < rembg_protect)
    alpha = np.where(force_out, 0.0, alpha)

    # Dark subject (plates/bar) that rembg keeps → opaque
    force_in = rembg_mask > 0.65
    alpha = np.where(force_in, np.maximum(alpha, rembg_mask), alpha)

    # Light despeckle
    a8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    a8 = cv2.medianBlur(a8, 3)
    return np.clip(a8.astype(np.float32) / 255.0, 0.0, 1.0)


# Back-compat name used by older process.py
def combine_with_rembg(
    rembg_mask: np.ndarray,
    flat_bg: np.ndarray,
    *,
    rembg_protect: float = 0.45,
) -> np.ndarray:
    soft = 1.0 - flat_bg  # crude
    return combine_studio_alpha(rembg_mask, soft, flat_bg, rembg_protect=rembg_protect)
