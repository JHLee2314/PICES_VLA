from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import numbers

GOOGLE_EMBODIMENT_TAG = "SIMPLER_ENV_GOOGLE"
MODEL_ACTION_HORIZON = 40
MODEL_ACTION_DIM = 132
OFFICIAL_ACTION_HORIZON = 8
VALID_ACTION_DIM = 7
ACTIVE_SCALAR_COUNT = OFFICIAL_ACTION_HORIZON * VALID_ACTION_DIM
RADIUS_REFERENCE = math.sqrt(ACTIVE_SCALAR_COUNT)
SUPPORTED_PROBE_DIRECTION_COUNTS = (1, 2, 3, 4)
METHOD = "multi_iid_keepm_first_cross_mean_adaptive"
DEFAULT_N = 3
SELECTOR_WINDOW = 1
MAX_EXEC = OFFICIAL_ACTION_HORIZON
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


@dataclass(frozen=True)
class MethodSpec:
    method: str
    sigma_values: tuple[float, ...]
    # N: number of nominal candidates.
    N: int = DEFAULT_N
    selector_window: int = SELECTOR_WINDOW
    max_exec: int = MAX_EXEC

    @property
    def keep_m(self) -> int:
        """M = ceil(N / 2), derived from the nominal candidate count."""
        return (self.N + 1) // 2

    @property
    def probe_direction_count(self) -> int:
        return len(self.sigma_values)


def validate_sigma_values(
    values: Sequence[float],
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("sigma_values must be a sequence")
    if len(values) not in SUPPORTED_PROBE_DIRECTION_COUNTS:
        allowed = ", ".join(
            str(value) for value in SUPPORTED_PROBE_DIRECTION_COUNTS
        )
        raise ValueError(f"sigma_values length must be one of {allowed}")
    parsed: list[float] = []
    for index, raw_value in enumerate(values):
        if isinstance(raw_value, bool) or not isinstance(
            raw_value,
            numbers.Real,
        ):
            raise ValueError(
                f"sigma_values[{index}] must be a real number"
            )
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"sigma_values[{index}] must be finite and positive"
            )
        parsed.append(value)
    return tuple(parsed)


def build_method_spec(
    method: str,
    sigma_values: Sequence[float],
    *,
    N: int = DEFAULT_N,
    selector_window: int = SELECTOR_WINDOW,
) -> MethodSpec:
    if method != METHOD:
        raise ValueError(f"method must be {METHOD!r}, got {method!r}")
    N = validate_candidate_count(N)
    return MethodSpec(
        method=method,
        sigma_values=validate_sigma_values(sigma_values),
        N=N,
        selector_window=validate_selector_window(selector_window),
    )


def validate_selector_window(selector_window: int) -> int:
    if isinstance(selector_window, bool) or not isinstance(
        selector_window, numbers.Integral
    ):
        raise ValueError("selector_window must be an integer")
    selector_window = int(selector_window)
    if not 1 <= selector_window <= MAX_EXEC:
        raise ValueError(f"selector_window must be in [1,{MAX_EXEC}]")
    return selector_window


def validate_candidate_count(N: int) -> int:
    if isinstance(N, bool) or not isinstance(N, numbers.Integral):
        raise ValueError("N must be an integer")
    N = int(N)
    if N < 1:
        raise ValueError("N must be at least 1")
    return N
