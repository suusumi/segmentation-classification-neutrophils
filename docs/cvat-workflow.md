# CVAT workflow

CVAT используем только для ручной проверки и исправления масок. Основной проект продолжает жить отдельно.

## Что готовим для CVAT

Наши pseudo-mask лежат в:

```text
data/processed/nucleus_segmentation/pseudo_labels/
```

CVAT удобнее кормить небольшими партиями. Для первой проверки создай batch на 200 изображений из train split:

```bash
source .venv/bin/activate
python scripts/prepare_cvat_batch.py --batch-name batch_001 --limit 200 --split train
```

Результат:

```text
data/processed/nucleus_segmentation/cvat_batches/batch_001/
  images.zip
  preannotations_segmentation_mask.zip
  labels.json
  manifest.csv
```

`images.zip` — изображения для CVAT task.

`preannotations_segmentation_mask.zip` — стартовые маски в CVAT Segmentation Mask формате.

`labels.json` — label `nucleus`.

## Локальный запуск CVAT

Официальный CVAT поднимается из собственного репозитория, а не из нашего compose-файла.

```bash
cd ..
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
```

Создай superuser:

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Открой:

```text
http://localhost:8080
```

Официальная инструкция CVAT: https://docs.cvat.ai/docs/administration/community/basics/installation/

## Создание первой task

В CVAT:

```text
Tasks -> Create a new task
```

Параметры:

```text
Name: neutrophil-nucleus-batch-001
Labels: импортировать labels.json или создать label nucleus вручную
Data: загрузить images.zip
```

После создания task импортируй стартовые маски:

```text
Actions -> Upload annotations
Format: Segmentation mask 1.1
File: preannotations_segmentation_mask.zip
```

Формат Segmentation Mask описан в документации CVAT:

```text
https://docs.cvat.ai/docs/dataset_management/formats/format-smask/
```

## Создание task через CLI

Если установлен `cvat-cli`, task можно создать без ручной загрузки файлов:

```bash
pip install cvat-cli
```

Создать task с изображениями:

```bash
cvat-cli \
  --server-host http://localhost \
  --server-port 8080 \
  --auth admin:admin12345 \
  task create \
  --labels data/processed/nucleus_segmentation/cvat_batches/batch_001/labels.json \
  --completion_verification_period 5 \
  "neutrophil-nucleus-batch-001" \
  local data/processed/nucleus_segmentation/cvat_batches/batch_001/images.zip
```

Импортировать preannotations:

```bash
cvat-cli \
  --server-host http://localhost \
  --server-port 8080 \
  --auth admin:admin12345 \
  task import-dataset \
  --format "Segmentation mask 1.1" \
  1 \
  data/processed/nucleus_segmentation/cvat_batches/batch_001/preannotations_segmentation_mask.zip
```

Формат должен называться именно `Segmentation mask 1.1`.

## Что править руками

Для U-Net первого этапа нужна только бинарная маска ядра:

```text
background = фон
nucleus = все ядро нейтрофила
```

Правила:

```text
1. В маске должно быть только ядро целевого нейтрофила.
2. Эритроциты, соседние клетки, фон и мусор должны быть background.
3. Если pseudo-mask пропустила часть ядра, дорисовать ее.
4. Если pseudo-mask захватила лишнее, стереть.
5. Если изображение сомнительное, лучше отметить его отдельно в manifest/export notes.
```

Отдельные доли ядра пока не смешиваем с binary mask. Для долей позже заведем отдельный annotation task или отдельный label scheme.

## Экспорт после ручной правки

После правки:

```text
Actions -> Export annotations
Format: Segmentation Mask 1.0
```

Экспортированный zip потом нужно будет конвертировать в наш training dataset:

```text
data/processed/nucleus_segmentation/curated/
  images/
    train/
    val/
    test/
  masks/
    train/
    val/
    test/
  manifest.csv
```

Этот import-back step будет следующим модулем после проверки CVAT workflow.
