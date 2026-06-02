"""Утилиты для воспроизводимых экспериментов."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Устанавливает seed для Python, NumPy и PyTorch.

    Args:
        seed: Значение seed, используемое во всех генераторах случайных чисел.
        deterministic: Если ``True``, включает детерминированное поведение PyTorch
            там, где это возможно. Это может снизить производительность, но улучшает
            воспроизводимость экспериментов.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
