from __future__ import annotations

from dataclasses import dataclass
import numbers

from backends.pi05_libero.src.config import (
    FIRST_CROSS_WINDOW,
    MAX_EXEC,
    DEFAULT_N,
    validate_positive_real,
)

SUPPORTED_PROBE_DIRECTION_COUNTS = frozenset({1, 2, 3, 4})


def validate_candidate_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def validate_window_size(
    value: object,
    name: str = "window_size",
    *,
    max_exec: int | None = None,
) -> int:
    parsed = validate_candidate_count(value, name)
    if max_exec is not None and parsed >= max_exec:
        raise ValueError(f"{name} must be in [1, {max_exec - 1}] (less than max_exec)")
    return parsed


def validate_probe_direction_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be one of: 1, 2, 3, 4")
    parsed = int(value)
    if parsed not in SUPPORTED_PROBE_DIRECTION_COUNTS:
        raise ValueError(f"{name} must be one of: 1, 2, 3, 4")
    return parsed


@dataclass(frozen=True)
class JAXPerturbMultiIidConfig:
    sigma: float = 0.25
    probe_direction_count: int = 1
    # N: number of nominal candidates.
    N: int = DEFAULT_N
    window_size: int = FIRST_CROSS_WINDOW

    def validate(self) -> None:
        validate_positive_real(self.sigma, "sigma")
        validate_probe_direction_count(self.probe_direction_count, "probe_direction_count")
        validate_candidate_count(self.N, "N")
        validate_window_size(self.window_size, max_exec=MAX_EXEC)

    @property
    def keep_m(self) -> int:
        """M = ceil(N / 2), derived from the nominal candidate count."""
        return (self.N + 1) // 2

    @property
    def total_solve_candidates(self) -> int:
        return (self.probe_direction_count + 1) * self.N
