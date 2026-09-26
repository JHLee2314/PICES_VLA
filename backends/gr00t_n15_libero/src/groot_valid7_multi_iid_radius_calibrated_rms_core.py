from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers
from typing import Any

import torch

from groot_solver_utils import (
    repeat_normalized_input,
    sample_gr00t_with_initial_actions,
)


VALID_ACTION_DIM = 7
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 16
SUPPORTED_PROBE_DIRECTION_COUNTS = (1, 2, 3, 4)
ACTIVE_DIMENSION = ACTION_HORIZON * VALID_ACTION_DIM
RADIUS_REFERENCE = math.sqrt(ACTIVE_DIMENSION)

@dataclass(frozen=True)
class Valid7MultiIidInferenceResult:
    sigma_values: tuple[float, ...]
    z0: torch.Tensor
    z_probes: torch.Tensor
    G_sigma_components: torch.Tensor
    G_radius_calibrated_components: torch.Tensor
    G: torch.Tensor
    delta_iid: torch.Tensor
    delta_iid_valid_l2: torch.Tensor
    input_perturb_valid_l2: torch.Tensor
    g_radius_scale: torch.Tensor
    delta_padding_max_abs: torch.Tensor
    probe_changed_fraction_valid7: torch.Tensor
    probe_changed_fraction_padding25: torch.Tensor
    base_noise_padding_nonzero_fraction: torch.Tensor
    g_component_finite_fraction: torch.Tensor
    g_component_zero_fraction: torch.Tensor
    g_finite_fraction: torch.Tensor
    g_zero_fraction: torch.Tensor

    @property
    def probe_direction_count(self) -> int:
        return len(self.sigma_values)


def validate_sigma(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    sigma = float(value)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return sigma


def validate_sigma_values(
    values: Sequence[float],
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("sigma_values must be a sequence of real numbers")
    if len(values) not in SUPPORTED_PROBE_DIRECTION_COUNTS:
        allowed = ", ".join(str(value) for value in SUPPORTED_PROBE_DIRECTION_COUNTS)
        raise ValueError(f"sigma_values length must be one of {allowed}")
    return tuple(
        validate_sigma(value, f"sigma_values[{index}]")
        for index, value in enumerate(values)
    )


def _validate_model_shape(value: torch.Tensor, name: str) -> None:
    expected_tail = (ACTION_HORIZON, MODEL_ACTION_DIM)
    if value.ndim != 3 or tuple(value.shape[1:]) != expected_tail:
        raise ValueError(
            f"{name} must have shape [N,{ACTION_HORIZON},{MODEL_ACTION_DIM}], "
            f"got {tuple(value.shape)}"
        )


def mask_valid7_delta_without_normalization(delta_raw: torch.Tensor) -> torch.Tensor:
    """Mask padded coordinates while preserving sampled valid7 coordinates."""
    _validate_model_shape(delta_raw, "delta_raw")
    delta_iid = delta_raw.float().clone()
    delta_iid[..., VALID_ACTION_DIM:] = 0.0
    return delta_iid.to(dtype=delta_raw.dtype)


def _validate_delta_stack(
    delta_iid: torch.Tensor,
    *,
    N: int,
    probe_direction_count: int,
) -> None:
    expected = (
        probe_direction_count,
        N,
        ACTION_HORIZON,
        MODEL_ACTION_DIM,
    )
    if delta_iid.ndim != 4 or tuple(delta_iid.shape) != expected:
        raise ValueError(f"delta_iid must have shape {expected}, got {tuple(delta_iid.shape)}")
    if bool((delta_iid[..., VALID_ACTION_DIM:] != 0).any()):
        raise ValueError("delta_iid contains a non-zero padded coordinate")


def make_multi_iid_probe_noise(
    base_noise: torch.Tensor,
    delta_iid: torch.Tensor,
    sigma_values: Sequence[float],
) -> torch.Tensor:
    _validate_model_shape(base_noise, "base_noise")
    sigmas = validate_sigma_values(sigma_values)
    probe_direction_count = len(sigmas)
    N = int(base_noise.shape[0])
    _validate_delta_stack(
        delta_iid,
        N=N,
        probe_direction_count=probe_direction_count,
    )
    if delta_iid.device != base_noise.device or delta_iid.dtype != base_noise.dtype:
        raise ValueError("delta_iid must match base_noise device and dtype")
    sigma_tensor = torch.tensor(
        sigmas,
        dtype=base_noise.dtype,
        device=base_noise.device,
    ).view(probe_direction_count, 1, 1, 1)
    probe_noise = base_noise.unsqueeze(0) + sigma_tensor * delta_iid
    return torch.cat(
        (base_noise, probe_noise.flatten(start_dim=0, end_dim=1)),
        dim=0,
    )


def split_multi_iid_output(
    final_all: torch.Tensor,
    *,
    N: int,
    probe_direction_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if N < 1:
        raise ValueError("N must be positive")
    if probe_direction_count not in SUPPORTED_PROBE_DIRECTION_COUNTS:
        raise ValueError("unsupported probe_direction_count")
    expected = (probe_direction_count + 1) * N
    if final_all.ndim != 3 or final_all.shape[0] != expected:
        raise ValueError(
            f"final_all must have shape [{expected},H,D], got {tuple(final_all.shape)}"
        )
    blocks = final_all.reshape(
        probe_direction_count + 1,
        N,
        final_all.shape[1],
        final_all.shape[2],
    )
    return blocks[0], blocks[1:]


def compute_valid7_multi_iid_radius_calibrated_rms_sensitivity_g(
    z0: torch.Tensor,
    z_probes: torch.Tensor,
    delta_iid: torch.Tensor,
    sigma_values: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return final G, sigma components, calibrated components, radii, scales."""
    sigmas = validate_sigma_values(sigma_values)
    probe_direction_count = len(sigmas)
    _validate_model_shape(z0, "z0")
    N = int(z0.shape[0])
    expected_probe_shape = (
        probe_direction_count,
        N,
        ACTION_HORIZON,
        MODEL_ACTION_DIM,
    )
    if z_probes.ndim != 4 or tuple(z_probes.shape) != expected_probe_shape:
        raise ValueError(
            f"z_probes must have shape {expected_probe_shape}, got {tuple(z_probes.shape)}"
        )
    _validate_delta_stack(
        delta_iid,
        N=N,
        probe_direction_count=probe_direction_count,
    )

    delta_valid = delta_iid[..., :VALID_ACTION_DIM].float()
    active_vectors = delta_valid.flatten(start_dim=2)
    delta_iid_valid_l2 = torch.linalg.vector_norm(
        active_vectors,
        ord=2,
        dim=-1,
    )
    if not bool(torch.isfinite(delta_iid_valid_l2).all()) or bool(
        (delta_iid_valid_l2 <= 0).any()
    ):
        raise FloatingPointError("active Gaussian probe L2 radius is invalid")

    diff_valid = (
        z_probes[..., :VALID_ACTION_DIM].float()
        - z0.unsqueeze(0)[..., :VALID_ACTION_DIM].float()
    )
    sigma_tensor = torch.tensor(
        sigmas,
        dtype=torch.float32,
        device=diff_valid.device,
    ).view(probe_direction_count, 1, 1)
    G_sigma_components = torch.linalg.vector_norm(
        diff_valid,
        ord=2,
        dim=-1,
    ) / sigma_tensor
    g_radius_scale = RADIUS_REFERENCE / delta_iid_valid_l2
    G_radius_calibrated_components = (
        G_sigma_components * g_radius_scale[..., None]
    )
    G = torch.sqrt(
        torch.mean(G_radius_calibrated_components.square(), dim=0)
    ).unsqueeze(0)

    tensors = {
        "G_sigma_components": G_sigma_components,
        "G_radius_calibrated_components": G_radius_calibrated_components,
        "G": G,
        "g_radius_scale": g_radius_scale,
    }
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
    return (
        G,
        G_sigma_components,
        G_radius_calibrated_components,
        delta_iid_valid_l2,
        g_radius_scale,
    )


def compute_probe_diagnostics(
    noise_all: torch.Tensor,
    *,
    N: int,
    probe_direction_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base, probes = split_multi_iid_output(
        noise_all,
        N=N,
        probe_direction_count=probe_direction_count,
    )
    perturbations = probes.float() - base.unsqueeze(0).float()
    valid = perturbations[..., :VALID_ACTION_DIM]
    padding = perturbations[..., VALID_ACTION_DIM:]
    input_perturb_valid_l2 = torch.linalg.vector_norm(
        valid.flatten(start_dim=2),
        ord=2,
        dim=-1,
    )
    changed_valid = (valid != 0).float().flatten(start_dim=2).mean(dim=2)
    changed_padding = (padding != 0).float().flatten(start_dim=2).mean(dim=2)
    return input_perturb_valid_l2, changed_valid, changed_padding


@torch.no_grad()
def infer_valid7_multi_iid_radius_calibrated_rms_sensitivity_g(
    model,
    normalized_input: Mapping[str, Any],
    *,
    N: int,
    sigma_values: Sequence[float],
) -> Valid7MultiIidInferenceResult:
    if N < 1:
        raise ValueError("N must be positive")
    sigmas = validate_sigma_values(sigma_values)
    probe_direction_count = len(sigmas)
    num_solves = (probe_direction_count + 1) * N

    normalized_input_all = repeat_normalized_input(normalized_input, num_solves)
    backbone_inputs, action_inputs = model.prepare_input(normalized_input_all)
    backbone_outputs = model.backbone(backbone_inputs)

    action_head = model.action_head
    backbone_outputs = action_head.process_backbone_output(backbone_outputs)
    features = backbone_outputs.backbone_features
    horizon = int(action_head.config.action_horizon)
    action_dim = int(action_head.config.action_dim)
    if horizon != ACTION_HORIZON or action_dim != MODEL_ACTION_DIM:
        raise ValueError(
            "GR00T N1.5 multi-iid inference requires "
            f"action shape [{ACTION_HORIZON},{MODEL_ACTION_DIM}], "
            f"got [{horizon},{action_dim}]"
        )

    base_noise = torch.randn(
        (N, horizon, action_dim),
        device=features.device,
        dtype=features.dtype,
    )
    delta_iid = torch.stack(
        tuple(
            mask_valid7_delta_without_normalization(
                torch.randn(
                    (N, horizon, action_dim),
                    device=features.device,
                    dtype=features.dtype,
                )
            )
            for _ in sigmas
        ),
        dim=0,
    )
    noise_all = make_multi_iid_probe_noise(base_noise, delta_iid, sigmas)
    action_outputs = sample_gr00t_with_initial_actions(
        action_head,
        backbone_outputs,
        action_inputs,
        initial_actions=noise_all,
        backbone_is_processed=True,
    )
    model.validate_data(action_outputs, backbone_outputs, is_training=False)
    final_all = action_outputs["action_pred"]
    z0, z_probes = split_multi_iid_output(
        final_all,
        N=N,
        probe_direction_count=probe_direction_count,
    )
    (
        G,
        G_sigma_components,
        G_radius_calibrated_components,
        delta_iid_valid_l2,
        g_radius_scale,
    ) = compute_valid7_multi_iid_radius_calibrated_rms_sensitivity_g(
        z0,
        z_probes,
        delta_iid,
        sigmas,
    )
    (
        input_perturb_valid_l2,
        probe_changed_fraction_valid7,
        probe_changed_fraction_padding25,
    ) = compute_probe_diagnostics(
        noise_all,
        N=N,
        probe_direction_count=probe_direction_count,
    )

    delta_padding = delta_iid[..., VALID_ACTION_DIM:]
    delta_padding_max_abs = delta_padding.float().abs().max()
    base_noise_padding_nonzero_fraction = (
        (base_noise[..., VALID_ACTION_DIM:] != 0)
        .float()
        .reshape(N, -1)
        .mean(dim=1)
    )
    g_component_finite_fraction = torch.isfinite(
        G_radius_calibrated_components
    ).float().mean(dim=(1, 2))
    g_component_zero_fraction = (
        G_radius_calibrated_components == 0
    ).float().mean(dim=(1, 2))
    g_finite_fraction = torch.isfinite(G).float().mean(dim=(1, 2))
    g_zero_fraction = (G == 0).float().mean(dim=(1, 2))

    if float(delta_padding_max_abs) != 0.0:
        raise FloatingPointError("multi-iid delta padding is not exactly zero")
    if bool((probe_changed_fraction_padding25 != 0).any()):
        raise FloatingPointError("multi-iid probes changed padded input coordinates")

    return Valid7MultiIidInferenceResult(
        sigma_values=sigmas,
        z0=z0,
        z_probes=z_probes,
        G_sigma_components=G_sigma_components,
        G_radius_calibrated_components=G_radius_calibrated_components,
        G=G,
        delta_iid=delta_iid,
        delta_iid_valid_l2=delta_iid_valid_l2,
        input_perturb_valid_l2=input_perturb_valid_l2,
        g_radius_scale=g_radius_scale,
        delta_padding_max_abs=delta_padding_max_abs,
        probe_changed_fraction_valid7=probe_changed_fraction_valid7,
        probe_changed_fraction_padding25=probe_changed_fraction_padding25,
        base_noise_padding_nonzero_fraction=base_noise_padding_nonzero_fraction,
        g_component_finite_fraction=g_component_finite_fraction,
        g_component_zero_fraction=g_component_zero_fraction,
        g_finite_fraction=g_finite_fraction,
        g_zero_fraction=g_zero_fraction,
    )
