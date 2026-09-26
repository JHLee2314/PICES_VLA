from __future__ import annotations

from typing import Any

import numpy as np

from n17_simplerenv_multi_iid_config import (
    METHOD,
    MODEL_ACTION_HORIZON,
    MethodSpec,
    build_method_spec,
)


def select_keepm_first_cross_mean(
    sensitivity_g: np.ndarray,
    spec: MethodSpec,
) -> dict[str, Any]:
    spec = build_method_spec(
        spec.method,
        spec.sigma_values,
        N=spec.N,
        selector_window=spec.selector_window,
    )
    values = np.asarray(sensitivity_g, dtype=np.float32)
    expected = (1, spec.N, MODEL_ACTION_HORIZON)
    if values.shape != expected:
        raise ValueError(f"sensitivity_g must have shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("sensitivity_g contains NaN or Inf")

    curves = values[0, :, : spec.max_exec]
    prefix_mean = curves[:, : spec.selector_window].mean(
        axis=1,
        dtype=np.float32,
    )
    keep = np.asarray(
        sorted(
            range(spec.N),
            key=lambda index: (float(prefix_mean[index]), index),
        )[: spec.keep_m],
        dtype=np.int64,
    )
    first_cross = np.full(
        spec.N,
        spec.max_exec,
        dtype=np.int64,
    )
    for candidate in range(spec.N):
        offsets = np.flatnonzero(
            curves[candidate, spec.selector_window : spec.max_exec]
            > prefix_mean[candidate]
        )
        if offsets.size:
            first_cross[candidate] = (
                spec.selector_window + int(offsets[0])
            )

    selected = min(
        keep.tolist(),
        key=lambda index: (
            -int(first_cross[index]),
            float(prefix_mean[index]),
            index,
        ),
    )
    selected_n_exec = int(first_cross[selected])
    return {
        "selector": "keepm_first_cross_mean",
        "execution_mode": "adaptive",
        "risk_source": "multi_iid_radius_calibrated_rms_sensitivity_g",
        "smoothing": "none",
        "N": spec.N,
        "probe_direction_count": spec.probe_direction_count,
        "sigma_values": list(spec.sigma_values),
        "keep_m": spec.keep_m,
        "selector_window": spec.selector_window,
        "max_exec": spec.max_exec,
        "candidate_prefix_mean": prefix_mean,
        "first_cross_index": first_cross,
        "keep_candidate_indices": keep,
        "keep_prefix_mean": prefix_mean[keep],
        "keep_first_cross_index": first_cross[keep],
        "selected_candidate_index": int(selected),
        "selected_candidate_prefix_mean": float(prefix_mean[selected]),
        "selected_n_exec": selected_n_exec,
        "nominal_executed_n_exec": selected_n_exec,
        "selected_no_cross": bool(selected_n_exec == spec.max_exec),
    }


def select_method(sensitivity_g: np.ndarray, spec: MethodSpec) -> dict[str, Any]:
    if spec.method != METHOD:
        raise ValueError(f"unsupported method: {spec.method!r}")
    return select_keepm_first_cross_mean(sensitivity_g, spec)
