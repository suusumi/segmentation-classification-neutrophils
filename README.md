# Neutrophil Segmentation and Classification

Cross-platform ML application for single-cell neutrophil analysis.

Current focus:

```text
single neutrophil image
-> preprocessing
-> nucleus segmentation
-> mask postprocessing
-> morphological feature extraction
-> nucleus segment counting
-> normal / hypersegmentation classification
-> saved artifacts and report
```

The project is structured so that a future YOLO detector can be added for full blood-smear images before passing cropped neutrophils into the same single-cell pipeline.

## Architecture

```text
src/
  api/          FastAPI app and schemas
  core/         project-relative paths and settings
  datasets/     Acevedo dataset loaders and transforms
  detection/    future YOLO detector interface
  models/       U-Net model definition
  pipeline/     preprocessing, segmentation, postprocessing, features, classification
  services/     orchestration, validation, domain errors
  storage/      SQLite repository for analysis metadata and JSON results
  utils/        config, IO, logging, seed helpers

data/
  raw/          raw datasets, for example data/raw/acevedo
  interim/      temporary derived data
  processed/    split CSVs and SQLite metadata

models/         model weights, not committed
outputs/        analysis artifacts, masks, overlays, reports, logs
frontend/       React/Vite client
```

All application paths are built from the project root via `pathlib.Path`. API responses store artifact paths as project-relative POSIX strings, for example `outputs/analyses/<id>/artifacts/nucleus_mask.png`.

## Dataset Layout

Acevedo should stay in folder-per-class form:

```text
data/raw/acevedo/
  basophil/
  eosinophil/
  erythroblast/
  ig/
  lymphocyte/
  monocyte/
  neutrophil/
  platelet/
```

Regenerate portable split CSVs with:

```bash
python scripts/prepare_acevedo.py \
  --input_dir data/raw/acevedo \
  --output_dir data/processed/acevedo_splits
```

The script now writes project-relative paths instead of machine-specific absolute paths.

## Backend

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn src.api.app:app --reload
```

Endpoints:

```text
GET  /health
POST /analysis
GET  /analysis/{analysis_id}
GET  /analysis/{analysis_id}/logs
GET  /analysis/{analysis_id}/report
```

Example upload:

```bash
curl -X POST http://localhost:8000/analysis \
  -F "file=@data/raw/acevedo/neutrophil/NEUTROPHIL_100292.jpg"
```

## Pipeline Status

Implemented now:

- Image upload validation.
- Image persistence under `outputs/analyses/<analysis_id>/input`.
- Baseline nucleus segmentation with classical thresholding.
- Mask postprocessing.
- Morphological feature extraction.
- Rule-based normal / hypersegmentation classification.
- Mask, overlay, JSON report, Markdown report, and log file persistence.
- SQLite metadata store at `data/processed/analysis/analysis.sqlite3`.

## Annotation Workflow

Generate pseudo-label masks for manual review with:

```bash
python scripts/export_pseudo_masks.py
```

The generated annotation dataset is written to `data/processed/nucleus_segmentation/pseudo_labels`.
See `docs/annotation-workflow.md` for the manual mask correction workflow.
For CVAT batch preparation, see `docs/cvat-workflow.md`.

Prepared extension points:

- `src.models.UNet` defines the U-Net architecture.
- `UNetNucleusSegmenter` is ready for trained weights in `models/unet_nucleus.pt`.
- `detection.YOLONeutrophilDetector` defines the future full-smear detection boundary.

## U-Net Nucleus Training

Convert corrected CVAT masks into a curated image/mask dataset:

```bash
python scripts/convert_cvat_nucleus_export.py --overwrite
```

The curated dataset is written to:

```text
data/processed/nucleus_segmentation/curated/
  images/{train,val,test}/
  masks/{train,val,test}/
  manifest.csv
```

Train the first U-Net nucleus segmenter:

```bash
python scripts/train_unet_nucleus.py \
  --epochs 40 \
  --batch-size 8 \
  --image-size 256 \
  --weights-path models/unet_nucleus.pt
```

After `models/unet_nucleus.pt` exists, configure the pipeline with
`SEGMENTER_NAME=unet` to force the trained model instead of the threshold
baseline. By default, `SEGMENTER_NAME=auto`: the application uses
`models/unet_nucleus.pt` when it exists and falls back to the threshold
segmenter when it does not.

On Windows PowerShell:

```powershell
$env:SEGMENTER_NAME = "unet"
$env:UNET_WEIGHTS_PATH = "models/unet_nucleus.pt"
uvicorn src.api.app:app --reload
```

## Database Choice

Use SQLite for the first stage. It is built into Python, cross-platform, requires no server, and is enough for local development, demos, and single-machine deployments.

When the application becomes multi-user or distributed, move analysis metadata to PostgreSQL and keep image artifacts in object storage or a mounted volume. The repository layer in `src/storage` is the boundary to replace.

## Frontend

Run the React app:

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the API at `http://localhost:8000` by default. Override with:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Docker

Run backend and frontend:

```bash
docker compose up --build
```

Services:

```text
API:      http://localhost:8000
Frontend: http://localhost:5173
```

The compose file mounts `data`, `models`, and `outputs` so local datasets, weights, and analysis results survive container restarts.
