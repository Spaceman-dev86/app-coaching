#!/usr/bin/env python3
"""Process bundled samples end-to-end and copy results to validation dirs."""

from __future__ import annotations

import shutil
from pathlib import Path

from pipeline import process_batch

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
APP_OUT = ROOT / "output"
STORE_VAL = Path("/cursor/stores/self/media/validation")


def main() -> None:
    paths = sorted(SAMPLES.glob("*"))
    paths = [p for p in paths if p.is_file() and not p.name.startswith(".")]
    if not paths:
        raise SystemExit(f"No samples in {SAMPLES}")

    APP_OUT.mkdir(parents=True, exist_ok=True)
    STORE_VAL.mkdir(parents=True, exist_ok=True)

    def progress(frac: float, msg: str) -> None:
        print(f"[{frac:5.1%}] {msg}", flush=True)

    results, zip_path = process_batch(
        paths,
        output_dir=APP_OUT,
        edge_sensitivity=0.55,
        keep_floor_shadow=True,
        make_zip=True,
        progress=progress,
    )

    for r in results:
        dest = STORE_VAL / r.name
        shutil.copy2(r, dest)
        print(f"validation copy → {dest}")

    if zip_path and zip_path.exists():
        shutil.copy2(zip_path, STORE_VAL / zip_path.name)
        print(f"validation zip → {STORE_VAL / zip_path.name}")

    print("OK", len(results), "outputs")


if __name__ == "__main__":
    main()
