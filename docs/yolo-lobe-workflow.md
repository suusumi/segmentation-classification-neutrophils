# YOLO lobe annotation workflow

This workflow trains YOLO-seg to detect separate neutrophil nucleus lobes.

## CVAT labels

Create one polygon label:

```text
nucleus_lobe
```

Do not create labels such as `lobe_1`, `lobe_2`, or `segment_3`. Every visible
lobe uses the same class. The segment count is the number of `nucleus_lobe`
objects on the image.

## Annotation rule

Draw one polygon for each visible nucleus lobe:

```text
one visible nucleus lobe = one nucleus_lobe polygon
```

Practical rules:

- Mark only the dark purple nucleus lobes, not the cytoplasm.
- If two lobes are connected by a thin bridge but visually/morphologically look
  like separate lobes, draw them as separate polygons.
- If the boundary between lobes is uncertain, prefer a consistent decision and
  flag the image for later review outside CVAT.
- Do not annotate erythrocytes, platelets, neighbor cells, stain debris, or
  small artifacts.
- For hypersegmented examples, still use the same `nucleus_lobe` label for every
  lobe. A five-lobe neutrophil has five objects of the same class.

## Export from CVAT

Export annotations as:

```text
COCO 1.0 / COCO Instance Segmentation
```

Unpack the export into:

```text
data/cvat_lobes/
```

The expected JSON is usually:

```text
data/cvat_lobes/annotations/instances_default.json
```

## Convert CVAT to curated project dataset

Run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\convert_cvat_lobes_export.py `
  --cvat-dir data\cvat_lobes `
  --source-images-dir data\raw\acevedo\neutrophil `
  --output-dir data\processed\nucleus_lobes\curated `
  --label nucleus_lobe `
  --overwrite
```

This writes:

```text
data/processed/nucleus_lobes/curated/
  images/{train,val,test}/
  nucleus_masks/{train,val,test}/
  lobe_instance_masks/{train,val,test}/
  manifest.csv
```

## Convert curated dataset to YOLO-seg format

```powershell
.\.venv\Scripts\python.exe scripts\prepare_yolo_lobes_dataset.py `
  --manifest data\processed\nucleus_lobes\curated\manifest.csv `
  --output-dir data\processed\nucleus_lobes\yolo_seg `
  --overwrite
```

YOLO files are written to:

```text
data/processed/nucleus_lobes/yolo_seg/
  images/{train,val,test}/
  labels/{train,val,test}/
  data.yaml
  manifest.csv
```

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_lobes.py `
  --data-yaml data\processed\nucleus_lobes\yolo_seg\data.yaml `
  --model yolo11n-seg.pt `
  --epochs 100 `
  --image-size 640 `
  --batch-size 8 `
  --device 0 `
  --weights-path models\yolo_lobes_seg.pt
```

Recommended current evaluation thresholds:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_yolo_lobes.py `
  --split test `
  --weights-path models\yolo_lobes_seg.pt `
  --output-dir outputs\yolo_lobes_eval\test_conf040_iou030 `
  --conf 0.40 `
  --iou 0.30 `
  --save-plots
```

## Fast annotation with YOLO preannotations

After training a first YOLO model, use it to generate draft polygons:

```powershell
.\.venv\Scripts\python.exe scripts\export_yolo_lobe_preannotations.py `
  --images-dir data\processed\nucleus_segmentation\cvat_batches\batch_001\images `
  --weights-path models\yolo_lobes_seg.pt `
  --output-zip data\processed\nucleus_lobes\yolo_preannotations.zip `
  --conf 0.40 `
  --iou 0.30
```

Import the ZIP into CVAT as COCO Instance Segmentation, then correct the
predicted polygons manually. This is the fastest path: review and fix instead
of drawing every lobe from zero.
