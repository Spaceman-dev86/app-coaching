# Background Remover

Outil local de détourage précis (photos, GIF, MP4).

- **Photos** (JPG/PNG/WebP) → PNG avec alpha  
- **GIF** → WebP animé (vrai alpha)  
- **MP4** → WebM VP9 + alpha (`ffmpeg`)

Pipeline : decode → `rembg` (`isnet-general-use`) → refine bords → encode.

## Windows (PowerShell)

```powershell
cd C:\workspace\background-remover
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python main.py
```

Puis ouvrir le navigateur : **http://127.0.0.1:7860**

Le premier lancement télécharge le modèle rembg (~170 Mo).

### Si `Activate.ps1` est bloqué

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### ffmpeg (pour MP4 → WebM)

Installez `ffmpeg` et ajoutez-le au `PATH`, sinon les photos/GIF fonctionnent déjà.

## Interface HTML

- Multi-fichiers + glisser-déposer  
- Aperçu damier + téléchargement  
- Sensibilité des bords + option ombre au sol  
- Sorties dans `output\<job_id>\` + `cutouts.zip`

Alternative Gradio (port 7861) : `python gradio_app.py`

## Structure

```
C:\workspace\background-remover\
  main.py              # lance le serveur HTML
  server.py            # FastAPI + API /api/process
  static\index.html    # UI glisser-déposer
  pipeline\            # rembg + refine
  samples\             # exemples
  output\              # résultats
```
