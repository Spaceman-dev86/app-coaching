"""rembg session wrapper — prefer isnet-general-use, fall back to u2net."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from rembg import new_session, remove

PREFERRED_MODELS = ("isnet-general-use", "u2net")


class Masker:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.session = None
        self.model_name = model_name
        errors: list[str] = []
        candidates = [model_name] if model_name else list(PREFERRED_MODELS)
        for name in candidates:
            if not name:
                continue
            try:
                self.session = new_session(name)
                self.model_name = name
                break
            except Exception as exc:  # noqa: BLE001 — try next model
                errors.append(f"{name}: {exc}")
        if self.session is None:
            raise RuntimeError(
                "Could not load rembg model. Tried: "
                + "; ".join(errors or candidates)
            )

    def alpha_mask(self, rgba: np.ndarray) -> np.ndarray:
        """Return float alpha mask in [0, 1] from rembg."""
        rgb = Image.fromarray(rgba[..., :3], mode="RGB")
        out = remove(
            rgb,
            session=self.session,
            only_mask=True,
            post_process_mask=True,
        )
        if isinstance(out, Image.Image):
            mask = np.array(out.convert("L"), dtype=np.float32) / 255.0
        else:
            arr = np.asarray(out)
            if arr.ndim == 3:
                arr = arr[..., -1] if arr.shape[-1] == 4 else arr[..., 0]
            mask = arr.astype(np.float32)
            if mask.max() > 1.0:
                mask /= 255.0
        return np.clip(mask, 0.0, 1.0)
