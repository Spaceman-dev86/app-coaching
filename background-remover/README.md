# Background Remover

Local precision cutout tool for stills, GIFs, and MP4s.

- **Stills** (JPG/PNG/WebP) → PNG with real alpha  
- **GIF** → animated WebP (true alpha, not 1-bit GIF)  
- **MP4** → WebM VP9 + alpha (via `ffmpeg`)

Pipeline: decode frames → `rembg` (`isnet-general-use` / `u2net`) → edge-local soft refine for near-white backgrounds → encode.

## Requirements

- Python 3.10+
- `ffmpeg` on `PATH` (needed for MP4 → WebM)

## Install

```bash
cd background-remover
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads the rembg model weights (~170MB).

## Run

```bash
python main.py
```

Open **http://127.0.0.1:7860**

### CLI (batch without UI)

```bash
python -c "
from pathlib import Path
from pipeline import process_batch
process_batch(list(Path('samples').glob('*')), output_dir='output')
"
```

## UI

- Multi-file drop + batch progress  
- Edge sensitivity slider  
- Keep floor shadow on/off  
- Checkerboard before/after preview  
- Configurable output directory (default `./output`)  
- Download individual cutouts + `cutouts.zip`

## Layout

```
background-remover/
  main.py              # Gradio app
  requirements.txt
  README.md
  pipeline/
    decode.py
    rembg_mask.py
    edge_refine.py
    encode.py
    process.py
  samples/             # optional test assets
  output/              # default write location
```

## Notes

Edge refine applies a soft near-white threshold **only near the mask boundary**, so light-gray body highlights stay intact. Thin structures (cables, fingers) are protected by a light morphological preserve + rembg strand boost. GIF masks are temporally smoothed to reduce flicker.
