# Подготовка масок для ручной разметки

Цель этапа — получить стартовые маски ядер нейтрофилов, вручную проверить их и затем использовать исправленные маски как ground truth для обучения U-Net.

## Что генерирует скрипт

Скрипт берет изображения нейтрофилов из Acevedo и прогоняет текущий baseline:

```text
image -> threshold segmentation -> postprocessing -> pseudo-mask
```

Результат сохраняется в:

```text
data/processed/nucleus_segmentation/pseudo_labels/
  images/          копии исходных изображений
  pseudo_masks/    черновые бинарные маски ядра
  overlays/        исходное изображение + маска поверх него
  manifest.csv     таблица соответствий image/mask/overlay/split
```

Маска имеет формат PNG:

```text
черный = фон
белый = ядро
```

## Генерация всех pseudo-mask

Из корня проекта:

```bash
source .venv/bin/activate
python scripts/export_pseudo_masks.py
```

Быстрая проверка на первых 20 изображениях:

```bash
python scripts/export_pseudo_masks.py --limit 20 --overwrite
```

Перегенерировать все маски:

```bash
python scripts/export_pseudo_masks.py --overwrite
```

Не сохранять overlay-картинки:

```bash
python scripts/export_pseudo_masks.py --no-overlays
```

## Как проверять результат

Открой несколько файлов из:

```text
data/processed/nucleus_segmentation/pseudo_labels/overlays/
```

Если красная область совпадает с ядром нейтрофила и не цепляет чужие клетки, pseudo-mask можно использовать как старт для ручной правки.

Если видны ошибки, это нормально: pseudo-mask — не ground truth. Ее задача — ускорить ручную разметку, а не заменить ее.

## Дальше: ручная разметка

Рекомендуемый инструмент — CVAT.

Минимальный workflow:

```text
1. Импортировать изображения из images/
2. Подгрузить или вручную перенести pseudo_masks/ как стартовую разметку
3. Исправить границы ядра
4. При необходимости отдельно разметить доли ядра
5. Экспортировать проверенные маски
6. Конвертировать экспорт в обучающий датасет для U-Net
```

Для U-Net первого этапа нужна бинарная маска:

```text
image -> nucleus mask
```

Для подсчета сегментов ядра нужна отдельная разметка долей:

```text
0 = фон
1 = первая доля ядра
2 = вторая доля ядра
3 = третья доля ядра
...
```

Эти две задачи связаны, но это разные типы ground truth.

## CVAT batch

Подробная инструкция для CVAT лежит в:

```text
docs/cvat-workflow.md
```

Подготовить первую партию на 200 изображений:

```bash
python scripts/prepare_cvat_batch.py --batch-name batch_001 --limit 200 --split train
```
