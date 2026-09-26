from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gr00t.model.policy import Gr00tPolicy
from groot_valid7_multi_iid_radius_calibrated_rms_core import validate_sigma_values


class Gr00tValid7MultiIidRadiusCalibratedRmsSensitivityGPolicy(Gr00tPolicy):
    def __init__(
        self,
        *,
        N: int,
        sigma_values: Sequence[float],
        **kwargs: Any,
    ) -> None:
        if N < 1:
            raise ValueError("N must be positive")
        validate_sigma_values(sigma_values)
        super().__init__(**kwargs)
