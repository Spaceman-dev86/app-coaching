"""Flat black/white background detection + border-connected keying.

Anatomical GIFs often sit on pure studio black/white. rembg alone leaves
fringes; white-edge refine does nothing on black. Clear border-connected
flat pixels while rembg protects dark subject parts (weight plates).
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
    if std > 12.0:
        return None
    luma = float(0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2])
    chroma = float(mean.max() - mean.min())
    if luma <= 18.0 and chroma <= 12.0:
        return "black", mean
    if luma >= 235.0 and chroma <= 18.0:
        return "white", mean
    return None


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

    num, labels = cv2.connectedComponents(similar, connectivity=8)
    if num <= 1:
        return np.zeros((h, w), dtype=np.float32)

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


def combine_with_rembg(
    rembg_mask: np.ndarray,
    flat_bg: np.ndarray,
    *,
    rembg_protect: float = 0.45,
) -> np.ndarray:
    """Alpha: cut flooded flat BG unless rembg strongly keeps the pixel."""
    alpha = rembg_mask.astype(np.float32).copy()
    cut = flat_bg * (1.0 - np.clip((rembg_mask - rembg_protect) / 0.35, 0.0, 1.0))
    alpha = alpha * (1.0 - cut)
    force_out = (flat_bg > 0.7) & (rembg_mask < rembg_protect)
    alpha = np.where(force_out, 0.0, alpha)
    # Keep rembg core even if dark (black plates / cables)
    force_in = rembg_mask > 0.7
    alpha = np.where(force_in, np.maximum(alpha, rembg_mask), alpha)
    return np.clip(alpha, 0.0, 1.0)
