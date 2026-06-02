# Разметка сегментов ядра

## Что размечаем

В CVAT нужно создать отдельную задачу для сегментов ядра.

Label:

```text
nucleus_lobe
```

Правило:

```text
одна доля ядра = один отдельный polygon/shape nucleus_lobe
```

Все доли имеют один и тот же label `nucleus_lobe`, а количество сегментов берется из количества отдельных объектов на изображении.

## Как размечать

Для каждого изображения:

```text
1. Открыть изображение нейтрофила.
2. Выделить каждую видимую долю ядра отдельным polygon.
3. Если доли соединены тонкой перемычкой, разметить их как отдельные доли, если морфологически они выглядят как отдельные сегменты.
4. Если невозможно уверенно решить, лучше пропустить изображение или пометить его как сомнительное вне CVAT.
5. Не размечать эритроциты, тромбоциты, соседние клетки и артефакты.
```

## Экспорт из CVAT

После разметки:

```text
Actions -> Export annotations
Format: COCO 1.0 / COCO Instance Segmentation
```


## Конвертация в датасет проекта

Запуск из корня проекта:

```bash
source .venv/bin/activate
python scripts/convert_cvat_lobes_export.py \
  --cvat-dir data/cvat_lobes \
  --source-images-dir data/raw/acevedo/neutrophil \
  --output-dir data/processed/nucleus_lobes/curated \
  --label nucleus_lobe \
  --overwrite
```

Результат:

```text
data/processed/nucleus_lobes/curated/
  images/
    train/
    val/
    test/
  nucleus_masks/
    train/
    val/
    test/
  lobe_instance_masks/
    train/
    val/
    test/
  manifest.csv
```

`lobe_instance_masks` хранит instance mask:

```text
0 = фон
1 = первая доля ядра
2 = вторая доля ядра
...
N = N-я доля ядра
```

`nucleus_masks` — бинарная маска объединения всех долей. Она нужна для сравнения с общей U-Net маской ядра.

`manifest.csv` содержит `segment_count`, то есть целевое число сегментов для каждой картинки.

## Автоматические preannotations

Чтобы не размечать все с нуля, можно сгенерировать стартовые polygons `nucleus_lobe` из текущего pipeline:

```bash
source .venv/bin/activate
python scripts/export_lobe_preannotations.py \
  --batch-dir data/processed/nucleus_segmentation/cvat_batches/batch_001 \
  --output-zip data/processed/nucleus_lobes/cvat_batches/batch_001/preannotations_coco_instances.zip \
  --segmenter auto
```

Этот файл можно импортировать в CVAT:

```bash
cvat-cli \
  --server-host http://localhost \
  --server-port 8080 \
  --auth admin:admin12345 \
  task import-dataset \
  --format "COCO 1.0" \
  <task_id> \
  data/processed/nucleus_lobes/cvat_batches/batch_001/preannotations_coco_instances.zip
```
