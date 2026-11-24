#!/usr/bin/env python3
"""FastAPI server: static HTML UI + rembg pipeline API."""

from __future__ import annotations

import io
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from pipeline.edge_refine import composite_on_color
from pipeline.process import process_batch

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DEFAULT_OUTPUT = ROOT / "output"
UPLOADS = ROOT / "uploads"

ALLOWED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".mp4",
    ".webm",
    ".mov",
    ".m4v",
}

app = FastAPI(title="Background Remover")


def _checkerboard(size: tuple[int, int], cell: int = 12) -> np.ndarray:
    h, w = size
    yy, xx = np.indices((h, w))
    board = ((xx // cell) + (yy // cell)) % 2
    light = np.array([220, 220, 220], dtype=np.uint8)
    dark = np.array([170, 170, 170], dtype=np.uint8)
    return np.where(board[..., None] == 0, light, dark)


def composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    board = _checkerboard(rgba.shape[:2]).astype(np.float32)
    out = rgb * a + board * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _preview_png_bytes(path: Path) -> bytes:
    with Image.open(path) as im:
        rgba = np.array(im.convert("RGBA"), dtype=np.uint8)
    return _png_bytes(composite_on_checkerboard(rgba))


def _write_dual_previews(path: Path, job_out: Path) -> dict:
    """Write white / black / checkerboard previews — the real quality check."""
    with Image.open(path) as im:
        rgba = np.array(im.convert("RGBA"), dtype=np.uint8)
    stem = path.stem
    mapping = {
        "preview_white": composite_on_color(rgba, (255, 255, 255)),
        "preview_black": composite_on_color(rgba, (0, 0, 0)),
        "preview_check": composite_on_checkerboard(rgba),
    }
    urls = {}
    for key, rgb in mapping.items():
        name = f"{stem}_{key}.png"
        (job_out / name).write_bytes(_png_bytes(rgb))
        urls[key] = name
    return urls


@app.on_event("startup")
def _startup() -> None:
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(500, "static/index.html manquant")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "output": str(DEFAULT_OUTPUT),
        "version": "webp-video-1",
        "video_output": "webp",
    }


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/api/process")
async def api_process(
    files: List[UploadFile] = File(...),
    edge_sensitivity: float = Form(0.55),
    keep_floor_shadow: str = Form("true"),
):
    if not files:
        raise HTTPException(400, "Ajoutez au moins un fichier.")

    keep_shadow = _as_bool(keep_floor_shadow)
    edge = max(0.0, min(1.0, float(edge_sensitivity)))

    job_id = uuid.uuid4().hex[:12]
    job_upload = UPLOADS / job_id
    job_out = DEFAULT_OUTPUT / job_id
    job_upload.mkdir(parents=True, exist_ok=True)
    job_out.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    try:
        for uf in files:
            name = Path(uf.filename or "file").name
            suffix = Path(name).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    400,
                    f"Format non supporté: {name} (jpg/png/webp/gif/mp4…)",
                )
            dest = job_upload / name
            with dest.open("wb") as fh:
                shutil.copyfileobj(uf.file, fh)
            saved.append(dest)

        results, zip_path = process_batch(
            saved,
            output_dir=job_out,
            edge_sensitivity=edge,
            keep_floor_shadow=keep_shadow,
            make_zip=True,
        )

        items = []
        for out in results:
            name = out.name
            if name.endswith(".webp"):
                label = "WebP animé (Supabase OK · alpha)"
            elif name.endswith(".gif"):
                label = "compat GIF (qualité limitée)"
            elif name.endswith(".png"):
                label = "PNG alpha"
            else:
                label = out.suffix.lstrip(".")

            preview_url: Optional[str] = None
            preview_white = preview_black = preview_check = None
            # Dual BG previews on pro cutouts (skip GIF compat sidecars)
            if out.suffix.lower() in {".webp", ".png"}:
                try:
                    urls = _write_dual_previews(out, job_out)
                    preview_white = f"/files/{job_id}/{urls['preview_white']}"
                    preview_black = f"/files/{job_id}/{urls['preview_black']}"
                    preview_check = f"/files/{job_id}/{urls['preview_check']}"
                    preview_url = preview_check
                except Exception:  # noqa: BLE001
                    try:
                        preview_name = f"{out.stem}_preview.png"
                        (job_out / preview_name).write_bytes(_preview_png_bytes(out))
                        preview_url = f"/files/{job_id}/{preview_name}"
                    except Exception:  # noqa: BLE001
                        preview_url = None

            items.append(
                {
                    "source": name,
                    "output": name,
                    "label": label,
                    "download_url": f"/files/{job_id}/{name}",
                    "preview_url": preview_url,
                    "preview_white": preview_white,
                    "preview_black": preview_black,
                    "preview_check": preview_check,
                }
            )

        return {
            "job_id": job_id,
            "count": len(items),
            "items": items,
            "zip_url": f"/files/{job_id}/{zip_path.name}" if zip_path else None,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Échec traitement: {exc}") from exc


@app.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str):
    # Prevent path traversal
    safe_job = Path(job_id).name
    safe_name = Path(filename).name
    path = DEFAULT_OUTPUT / safe_job / safe_name
    if not path.is_file():
        raise HTTPException(404, "Fichier introuvable")
    return FileResponse(path, filename=safe_name)


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main() -> None:
    import uvicorn

    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Background Remover (HTML) → http://127.0.0.1:7860", flush=True)
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
