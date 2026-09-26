from __future__ import annotations

import math
import numbers

DEFAULT_N = 3
FIRST_CROSS_WINDOW = 5
ACTION_HORIZON = 10
ACTION_DIM = 32
MAX_EXEC = 10
DENOISE_STEPS = 10


def validate_positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite positive real number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return parsed


def validate_max_exec(value: object, action_horizon: int, name: str = "max_exec") -> int:
    minimum = FIRST_CROSS_WINDOW + 1
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer in [{minimum}, {action_horizon}]")
    parsed = int(value)
    if not minimum <= parsed <= action_horizon:
        raise ValueError(f"{name} must be an integer in [{minimum}, {action_horizon}]")
    return parsed
