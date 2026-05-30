# Разметка сегментов ядра

Этот этап нужен не для общей маски ядра, а для подсчета отдельных долей нейтрофила.

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

Не нужно создавать labels `lobe_1`, `lobe_2`, `lobe_3`. Все доли имеют один и тот же label `nucleus_lobe`, а количество сегментов берется из количества отдельных объектов на изображении.

## Как размечать

Для каждого изображения:

```text
1. Открой изображение нейтрофила.
2. Выдели каждую видимую долю ядра отдельным polygon.
3. Если доли соединены тонкой перемычкой, размечай их как отдельные доли, если морфологически они выглядят как отдельные сегменты.
4. Если невозможно уверенно решить, лучше пропустить изображение или пометить его как сомнительное вне CVAT.
5. Не размечай эритроциты, тромбоциты, соседние клетки и артефакты.
```

Для обучения не нужно размечать все 3000 изображений сразу. Практичный первый набор:

```text
100-200 изображений:
- нормальные нейтрофилы с 1-4 сегментами
- гиперсегментированные с 5+ сегментами
- сложные случаи с тонкими перемычками
- разные оттенки, масштаб, фон и качество
```

## Экспорт из CVAT

После разметки:

```text
Actions -> Export annotations
Format: COCO 1.0 / COCO Instance Segmentation
```

Скачанный архив распакуй в:

```text
data/cvat_lobes/
```

Внутри должен появиться COCO JSON, обычно:

```text
data/cvat_lobes/annotations/instances_default.json
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

Важно: не импортируй preannotations поверх task, где уже есть ручная работа, если не уверен, как CVAT объединит аннотации. Безопаснее создать отдельную auto-task и сравнить качество.

## Что будет следующим шагом

После появления `data/processed/nucleus_lobes/curated/manifest.csv` можно сделать baseline-оценку:

```text
U-Net nucleus mask -> текущий watershed count -> сравнение с ручным segment_count
```

Если точность подсчета окажется слабой, следующий шаг — обучать отдельный модуль для сегментов:

```text
image + nucleus mask -> lobe/boundary segmentation -> split -> segment_count
```
