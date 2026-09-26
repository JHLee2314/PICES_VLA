from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Any

import numpy as np

from groot_valid7_multi_iid_radius_calibrated_rms_core import (
    ACTION_HORIZON,
    validate_sigma_values,
)


DEFAULT_N = 3
DEFAULT_PREFIX_WINDOW = 1
DEFAULT_SIGMA_VALUES = (0.25,)
DEFAULT_MAX_EXEC = 16
EXECUTION_MODE = "g_prefix_keepm_first_cross_mean_sensitivity_g"
RISK_SOURCE = "sensitivity_g"


@dataclass(frozen=True)
class MultiIidKeepMConfig:
    sigma_values: tuple[float, ...] = DEFAULT_SIGMA_VALUES
    # N: number of nominal candidates.
    N: int = DEFAULT_N
    prefix_window: int = DEFAULT_PREFIX_WINDOW
    max_exec: int = DEFAULT_MAX_EXEC

    def validate(self) -> MultiIidKeepMConfig:
        integer_fields = {
            "N": self.N,
            "prefix_window": self.prefix_window,
            "max_exec": self.max_exec,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{name} must be an integer")
        if self.N < 1:
            raise ValueError("N must be positive")
        if not 1 <= self.prefix_window < self.max_exec:
            raise ValueError(
                "prefix_window and max_exec must satisfy "
                "1 <= prefix_window < max_exec"
            )
        if self.max_exec > ACTION_HORIZON:
            raise ValueError(
                f"max_exec must not exceed native horizon {ACTION_HORIZON}"
            )
        validate_sigma_values(self.sigma_values)
        return self

    @property
    def keep_m(self) -> int:
        """M = ceil(N / 2), derived from the nominal candidate count."""
        return (self.N + 1) // 2

    @property
    def probe_direction_count(self) -> int:
        return len(validate_sigma_values(self.sigma_values))

    @property
    def num_solves(self) -> int:
        return (self.probe_direction_count + 1) * int(self.N)


DEFAULT_CONFIG = MultiIidKeepMConfig()


def select_keepm_first_cross_mean(
    sensitivity_g: np.ndarray,
    config: MultiIidKeepMConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = config.validate()
    values = np.asarray(sensitivity_g, dtype=np.float32)
    expected_shape = (1, config.N, ACTION_HORIZON)
    if values.shape != expected_shape:
        raise ValueError(f"sensitivity_g must have shape {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("sensitivity_g contains NaN or Inf")

    prefix_g_mean = values[0, :, : config.prefix_window].mean(
        axis=1,
        dtype=np.float32,
    )
    keep_candidate_indices = np.asarray(
        sorted(
            range(config.N),
            key=lambda index: (float(prefix_g_mean[index]), index),
        )[: config.keep_m],
        dtype=np.int64,
    )

    first_cross_index = np.full(
        config.N,
        config.max_exec,
        dtype=np.int64,
    )
    for candidate in range(config.N):
        later_gt = (
            values[0, candidate, config.prefix_window : config.max_exec]
            > prefix_g_mean[candidate]
        )
        crossing_offsets = np.flatnonzero(later_gt)
        if crossing_offsets.size:
            first_cross_index[candidate] = (
                config.prefix_window + int(crossing_offsets[0])
            )

    selected_index = min(
        keep_candidate_indices.tolist(),
        key=lambda index: (
            -int(first_cross_index[index]),
            float(prefix_g_mean[index]),
            index,
        ),
    )
    n_exec = int(first_cross_index[selected_index])
    sigma_values = [float(value) for value in config.sigma_values]

    return {
        "execution_mode": EXECUTION_MODE,
        "risk_source": RISK_SOURCE,
        "smoothing": "none",
        "N": config.N,
        "probe_direction_count": config.probe_direction_count,
        "sigma_values": sigma_values,
        "keep_m": config.keep_m,
        "prefix_window": config.prefix_window,
        "max_exec": config.max_exec,
        "prefix_g_mean": prefix_g_mean,
        "first_cross_index": first_cross_index,
        "keep_candidate_indices": keep_candidate_indices,
        "keep_prefix_g_mean": prefix_g_mean[keep_candidate_indices],
        "keep_first_cross_index": first_cross_index[keep_candidate_indices],
        "selected_candidate_index": int(selected_index),
        "selected_prefix_g_mean": float(prefix_g_mean[selected_index]),
        "selected_first_cross_index": n_exec,
        "n_exec": n_exec,
        "selected_no_cross": n_exec == config.max_exec,
    }
