# Background Remover

Outil local de détourage précis (photos, GIF, MP4).

## Qualité : quelle méthode ?

| Source | Sortie **pro** (Supabase `exercise-media`) | Notes |
|--------|---------------------------------------------|-------|
| GIF / WebP animé | **WebP** animé (alpha) | GIF optionnel, qualité limitée |
| Photo | PNG alpha | — |
| MP4 / MOV | **WebP** animé (alpha) | Pas de WebM — refusé par le bucket coach |

Le format **GIF ne peut pas** égaler un WebP/PNG : palette 256 couleurs + transparence 1 bit.
Pour un rendu pro, utilisez toujours le **WebP**.

Fonds studio noir/blanc (illustrations anatomiques) : keying soft + rembg (protège les disques noirs).

## Windows (PowerShell)

```powershell
cd C:\workspace\background-remover
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python main.py
```

Ouvrir **http://127.0.0.1:7860**

Ou double-clic `start.bat`.

## MP4

`ffmpeg` doit être sur le `PATH` (déjà OK sur Ordi_Rmi). Lecture via imageio `FFMPEG` / OpenCV / CLI — **pas besoin de pyav**.

## Structure

```
C:\workspace\background-remover\
  main.py / server.py / static\index.html
  pipeline\   # decode, rembg, flat_bg, encode
  samples\ output\
```
