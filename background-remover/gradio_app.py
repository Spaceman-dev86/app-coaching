#!/usr/bin/env python3
"""Optional Gradio UI (same rembg pipeline). Prefer: python main.py (HTML)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from pipeline.process import process_batch

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output"


def _checkerboard(size: Tuple[int, int], cell: int = 12) -> np.ndarray:
    h, w = size
    yy, xx = np.indices((h, w))
    board = ((xx // cell) + (yy // cell)) % 2
    light = np.array([220, 220, 220], dtype=np.uint8)
    dark = np.array([170, 170, 170], dtype=np.uint8)
    rgb = np.where(board[..., None] == 0, light, dark)
    return rgb


def composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    board = _checkerboard(rgba.shape[:2]).astype(np.float32)
    out = rgb * a + board * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_preview_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"), dtype=np.uint8)


def run_batch(
    files: Optional[List],
    edge_sensitivity: float,
    keep_floor_shadow: bool,
    output_dir: str,
    progress=gr.Progress(track_tqdm=False),
):
    if not files:
        raise gr.Error("Drop at least one file (JPG/PNG/WebP/GIF/MP4).")

    paths: List[Path] = []
    for f in files:
        p = Path(f if isinstance(f, (str, Path)) else f.name)
        if not p.exists():
            raise gr.Error(f"Missing file: {p}")
        paths.append(p)

    out_dir = Path(output_dir.strip() or DEFAULT_OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    def on_progress(frac: float, msg: str) -> None:
        progress(frac, desc=msg)

    results, zip_path = process_batch(
        paths,
        output_dir=out_dir,
        edge_sensitivity=float(edge_sensitivity),
        keep_floor_shadow=bool(keep_floor_shadow),
        make_zip=True,
        progress=on_progress,
    )

    src0 = paths[0]
    with Image.open(src0) as im:
        before = np.array(im.convert("RGB"), dtype=np.uint8)
    after_rgba = _load_preview_rgba(results[0])
    after = composite_on_checkerboard(after_rgba)

    gallery = []
    for r in results:
        try:
            gallery.append(composite_on_checkerboard(_load_preview_rgba(r)))
        except Exception:  # noqa: BLE001
            pass

    status = f"Processed {len(results)} file(s) → {out_dir}"
    zip_file = str(zip_path) if zip_path else None
    download_files = [str(r) for r in results]
    if zip_file:
        download_files.append(zip_file)

    return before, after, gallery, status, download_files


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Background Remover") as demo:
        gr.Markdown(
            "# Background Remover (Gradio)\n"
            "Prefer the HTML UI: `python main.py` → http://127.0.0.1:7860"
        )
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(
                    label="Drop files (batch)",
                    file_count="multiple",
                    file_types=["image", ".gif", ".mp4", ".webm", ".mov", ".webp"],
                )
                edge = gr.Slider(
                    0.0,
                    1.0,
                    value=0.55,
                    step=0.01,
                    label="Edge sensitivity",
                )
                shadow = gr.Checkbox(value=True, label="Keep floor shadow")
                out_dir = gr.Textbox(value=str(DEFAULT_OUTPUT), label="Output directory")
                run_btn = gr.Button("Remove backgrounds", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
                downloads = gr.File(label="Downloads", file_count="multiple")
            with gr.Column(scale=1):
                with gr.Row():
                    before = gr.Image(label="Before", type="numpy")
                    after = gr.Image(label="After (checkerboard)", type="numpy")
                gallery = gr.Gallery(label="Batch previews", columns=3, height=320)

        run_btn.click(
            fn=run_batch,
            inputs=[files, edge, shadow, out_dir],
            outputs=[before, after, gallery, status, downloads],
        )

    return demo


def main() -> None:
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Gradio UI on http://127.0.0.1:7861 …", flush=True)
    build_ui().queue().launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
    )


if __name__ == "__main__":
    main()
