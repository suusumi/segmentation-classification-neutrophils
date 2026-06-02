"""Метрики для базовых моделей подсчета долей ядра."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np


@dataclass(frozen=True)
class LobeCountMetrics:
    """Сводные метрики предсказания числа долей."""
    sample_count: int
    exact_accuracy: float
    plus_minus_one_accuracy: float
    binary_hypersegmentation_accuracy: float
    confusion_matrix: list[list[int]]
    count_values: list[int]


def confusion_matrix_for_counts(
    targets: Sequence[int],
    predictions: Sequence[int],
    count_values: Sequence[int],
) -> list[list[int]]:
    """Строит матрицу ошибок, где строки - истинные количества, а столбцы - предсказанные."""
    index_by_count = {count: index for index, count in enumerate(count_values)}
    matrix = np.zeros((len(count_values), len(count_values)), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[index_by_count[int(target)], index_by_count[int(prediction)]] += 1
    return cast(list[list[int]], matrix.tolist())


def compute_lobe_count_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    count_values: Sequence[int],
    hypersegmentation_threshold: int = 5,
) -> LobeCountMetrics:
    """Вычисляет метрики подсчета и бинарной гиперсегментации."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length.")
    if not targets:
        raise ValueError("At least one sample is required to compute metrics.")

    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    exact = target_array == prediction_array
    plus_minus_one = np.abs(target_array - prediction_array) <= 1
    target_hyper = target_array >= hypersegmentation_threshold
    prediction_hyper = prediction_array >= hypersegmentation_threshold

    return LobeCountMetrics(
        sample_count=int(target_array.size),
        exact_accuracy=float(exact.mean()),
        plus_minus_one_accuracy=float(plus_minus_one.mean()),
        binary_hypersegmentation_accuracy=float((target_hyper == prediction_hyper).mean()),
        confusion_matrix=confusion_matrix_for_counts(
            targets=targets,
            predictions=predictions,
            count_values=count_values,
        ),
        count_values=list(count_values),
    )
