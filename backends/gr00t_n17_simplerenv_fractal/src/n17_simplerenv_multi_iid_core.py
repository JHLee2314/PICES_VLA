from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from n17_simplerenv_multi_iid_config import (
    MODEL_ACTION_DIM,
    MODEL_ACTION_HORIZON,
    OFFICIAL_ACTION_HORIZON,
    RADIUS_REFERENCE,
    SUPPORTED_PROBE_DIRECTION_COUNTS,
    VALID_ACTION_DIM,
    validate_sigma_values,
)


@dataclass(frozen=True)
class SimplerEnvMultiIidInferenceResult:
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
    delta_dimension_padding_max_abs: torch.Tensor
    delta_temporal_padding_max_abs: torch.Tensor
    probe_changed_fraction_active: torch.Tensor
    probe_changed_fraction_dim_padding: torch.Tensor
    probe_changed_fraction_time_padding: torch.Tensor
    base_noise_dim_padding_nonzero_fraction: torch.Tensor
    base_noise_time_padding_nonzero_fraction: torch.Tensor
    g_component_finite_fraction: torch.Tensor
    g_component_zero_fraction: torch.Tensor
    g_finite_fraction: torch.Tensor
    g_zero_fraction: torch.Tensor

    @property
    def probe_direction_count(self) -> int:
        return len(self.sigma_values)


def _validate_model_shape(value: torch.Tensor, name: str) -> None:
    expected_tail = (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM)
    if value.ndim != 3 or tuple(value.shape[1:]) != expected_tail:
        raise ValueError(
            f"{name} must have shape [N,{MODEL_ACTION_HORIZON},"
            f"{MODEL_ACTION_DIM}], got {tuple(value.shape)}"
        )


def mask_trainmask8x7_delta_without_normalization(
    delta_raw: torch.Tensor,
) -> torch.Tensor:
    _validate_model_shape(delta_raw, "delta_raw")
    delta_iid = delta_raw.float().clone()
    delta_iid[:, OFFICIAL_ACTION_HORIZON:, :] = 0.0
    delta_iid[:, :OFFICIAL_ACTION_HORIZON, VALID_ACTION_DIM:] = 0.0
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
        MODEL_ACTION_HORIZON,
        MODEL_ACTION_DIM,
    )
    if delta_iid.ndim != 4 or tuple(delta_iid.shape) != expected:
        raise ValueError(
            f"delta_iid must have shape {expected}, got {tuple(delta_iid.shape)}"
        )
    if bool(
        (
            delta_iid[
                :, :, :OFFICIAL_ACTION_HORIZON, VALID_ACTION_DIM:
            ]
            != 0
        ).any()
    ):
        raise ValueError("delta_iid contains non-zero dimension padding")
    if bool((delta_iid[:, :, OFFICIAL_ACTION_HORIZON:, :] != 0).any()):
        raise ValueError("delta_iid contains non-zero temporal padding")


def sample_request_noise(
    *,
    N: int,
    sigma_values: Sequence[float],
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if N < 1:
        raise ValueError("N must be positive")
    sigmas = validate_sigma_values(sigma_values)
    target_device = torch.device(device)
    shape = (N, MODEL_ACTION_HORIZON, MODEL_ACTION_DIM)
    base_noise = torch.randn(
        shape,
        device=target_device,
        dtype=dtype,
    )
    delta_iid = torch.stack(
        tuple(
            mask_trainmask8x7_delta_without_normalization(
                torch.randn(
                    shape,
                    device=target_device,
                    dtype=dtype,
                )
            )
            for _ in sigmas
        ),
        dim=0,
    )
    return base_noise, delta_iid


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
            f"final_all must have shape [{expected},H,D], "
            f"got {tuple(final_all.shape)}"
        )
    blocks = final_all.reshape(
        probe_direction_count + 1,
        N,
        final_all.shape[1],
        final_all.shape[2],
    )
    return blocks[0], blocks[1:]


def compute_trainmask8x7_multi_iid_radius_calibrated_rms_sensitivity_g(
    z0: torch.Tensor,
    z_probes: torch.Tensor,
    delta_iid: torch.Tensor,
    input_perturb_valid_l2: torch.Tensor,
    sigma_values: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigmas = validate_sigma_values(sigma_values)
    probe_direction_count = len(sigmas)
    _validate_model_shape(z0, "z0")
    N = int(z0.shape[0])
    expected_probe_shape = (
        probe_direction_count,
        N,
        MODEL_ACTION_HORIZON,
        MODEL_ACTION_DIM,
    )
    if z_probes.ndim != 4 or tuple(z_probes.shape) != expected_probe_shape:
        raise ValueError(
            f"z_probes must have shape {expected_probe_shape}, "
            f"got {tuple(z_probes.shape)}"
        )
    _validate_delta_stack(
        delta_iid,
        N=N,
        probe_direction_count=probe_direction_count,
    )

    active_delta = delta_iid[
        :, :, :OFFICIAL_ACTION_HORIZON, :VALID_ACTION_DIM
    ].float()
    delta_iid_valid_l2 = torch.linalg.vector_norm(
        active_delta.reshape(probe_direction_count, N, -1),
        ord=2,
        dim=-1,
    )
    if not bool(torch.isfinite(delta_iid_valid_l2).all()) or bool(
        (delta_iid_valid_l2 <= 0).any()
    ):
        raise FloatingPointError("active Gaussian probe L2 radius is invalid")
    expected_radius_shape = (probe_direction_count, N)
    if tuple(input_perturb_valid_l2.shape) != expected_radius_shape:
        raise ValueError(
            "input_perturb_valid_l2 must have shape "
            f"{expected_radius_shape}, got "
            f"{tuple(input_perturb_valid_l2.shape)}"
        )
    effective_radius = input_perturb_valid_l2.float()
    if not bool(torch.isfinite(effective_radius).all()) or bool(
        (effective_radius <= 0).any()
    ):
        raise FloatingPointError("effective input perturbation L2 is invalid")

    diff_valid = (
        z_probes[..., :VALID_ACTION_DIM].float()
        - z0.unsqueeze(0)[..., :VALID_ACTION_DIM].float()
    )
    sigma_tensor = torch.tensor(
        sigmas,
        dtype=torch.float32,
        device=diff_valid.device,
    ).view(probe_direction_count, 1, 1)
    diff_valid_l2 = torch.linalg.vector_norm(
        diff_valid,
        ord=2,
        dim=-1,
    )
    G_sigma_components = diff_valid_l2 / sigma_tensor
    g_radius_scale = (
        sigma_tensor.squeeze(-1) * RADIUS_REFERENCE / effective_radius
    )
    G_radius_calibrated_components = (
        diff_valid_l2 * RADIUS_REFERENCE / effective_radius[..., None]
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base, probes = split_multi_iid_output(
        noise_all,
        N=N,
        probe_direction_count=probe_direction_count,
    )
    perturbations = probes.float() - base.unsqueeze(0).float()
    active = perturbations[
        :, :, :OFFICIAL_ACTION_HORIZON, :VALID_ACTION_DIM
    ]
    dimension_padding = perturbations[
        :, :, :OFFICIAL_ACTION_HORIZON, VALID_ACTION_DIM:
    ]
    temporal_padding = perturbations[
        :, :, OFFICIAL_ACTION_HORIZON:, :
    ]
    input_perturb_valid_l2 = torch.linalg.vector_norm(
        active.reshape(probe_direction_count, N, -1),
        ord=2,
        dim=-1,
    )
    changed_active = (
        (active != 0)
        .float()
        .reshape(probe_direction_count, N, -1)
        .mean(dim=2)
    )
    changed_dimension_padding = (
        (dimension_padding != 0)
        .float()
        .reshape(probe_direction_count, N, -1)
        .mean(dim=2)
    )
    changed_temporal_padding = (
        (temporal_padding != 0)
        .float()
        .reshape(probe_direction_count, N, -1)
        .mean(dim=2)
    )
    return (
        input_perturb_valid_l2,
        changed_active,
        changed_dimension_padding,
        changed_temporal_padding,
    )


def _repeat_single_batch(value: torch.Tensor, repeats: int, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1:
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        raise ValueError(f"{name} must be a tensor with leading batch 1, got {shape}")
    factors = (repeats,) + (1,) * (value.ndim - 1)
    return value.repeat(factors)


@torch.no_grad()
def sample_n17_with_initial_actions(
    action_head: Any,
    *,
    backbone_features: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    image_mask: torch.Tensor,
    backbone_attention_mask: torch.Tensor,
    initial_actions: torch.Tensor,
) -> torch.Tensor:
    if initial_actions.ndim != 3:
        raise ValueError(
            f"initial_actions must have shape [B,H,D], got {tuple(initial_actions.shape)}"
        )
    batch_size = backbone_features.shape[0]
    expected_shape = (
        batch_size,
        int(action_head.action_horizon),
        int(action_head.action_dim),
    )
    if tuple(initial_actions.shape) != expected_shape:
        raise ValueError(
            f"initial_actions must have shape {expected_shape}, "
            f"got {tuple(initial_actions.shape)}"
        )
    if initial_actions.device != backbone_features.device:
        raise ValueError("initial_actions and backbone_features must share a device")
    if initial_actions.dtype != backbone_features.dtype:
        raise ValueError("initial_actions and backbone_features must share a dtype")

    actions = initial_actions
    num_steps = int(action_head.num_inference_timesteps)
    if num_steps < 1:
        raise ValueError("num_inference_timesteps must be positive")
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_cont = step / float(num_steps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(batch_size,),
            fill_value=t_discretized,
            device=actions.device,
        )
        action_features = action_head.action_encoder(
            actions, timesteps_tensor, embodiment_id
        )
        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=actions.device,
            )
            action_features = (
                action_features
                + action_head.position_embedding(pos_ids).unsqueeze(0)
            )
        sa_embs = torch.cat((state_features, action_features), dim=1)
        if action_head.config.use_alternate_vl_dit:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=backbone_features,
                timestep=timesteps_tensor,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=backbone_features,
                timestep=timesteps_tensor,
            )
        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        actions = actions + dt * pred_velocity
    return actions


@torch.no_grad()
def infer_n17_simplerenv_multi_iid(
    model: Any,
    model_inputs: dict[str, Any],
    *,
    N: int,
    sigma_values: Sequence[float],
) -> SimplerEnvMultiIidInferenceResult:
    if N < 1:
        raise ValueError("N must be positive")
    sigmas = validate_sigma_values(sigma_values)
    probe_direction_count = len(sigmas)
    num_solves = (probe_direction_count + 1) * N

    backbone_inputs, action_inputs = model.prepare_input(dict(model_inputs))
    if "action" in action_inputs:
        raise ValueError("SimplerEnv multi-IID inference does not support RTC action input")
    backbone_outputs = model.backbone(backbone_inputs)
    action_head = model.action_head
    features = action_head._encode_features(backbone_outputs, action_inputs)

    horizon = int(action_head.action_horizon)
    action_dim = int(action_head.action_dim)
    if (horizon, action_dim) != (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM):
        raise ValueError(
            "N1.7 SimplerEnv inference requires model action shape "
            f"[{MODEL_ACTION_HORIZON},{MODEL_ACTION_DIM}], "
            f"got [{horizon},{action_dim}]"
        )

    backbone_features = _repeat_single_batch(
        features.backbone_features, num_solves, "backbone_features"
    )
    state_features = _repeat_single_batch(
        features.state_features, num_solves, "state_features"
    )
    embodiment_id = _repeat_single_batch(
        action_inputs.embodiment_id, num_solves, "embodiment_id"
    )
    image_mask = _repeat_single_batch(
        backbone_outputs.image_mask, num_solves, "image_mask"
    )
    backbone_attention_mask = _repeat_single_batch(
        backbone_outputs.backbone_attention_mask,
        num_solves,
        "backbone_attention_mask",
    )

    base_noise, delta_iid = sample_request_noise(
        N=N,
        sigma_values=sigmas,
        device=backbone_features.device,
        dtype=backbone_features.dtype,
    )
    noise_all = make_multi_iid_probe_noise(base_noise, delta_iid, sigmas)
    (
        input_perturb_valid_l2,
        probe_changed_fraction_active,
        probe_changed_fraction_dim_padding,
        probe_changed_fraction_time_padding,
    ) = compute_probe_diagnostics(
        noise_all,
        N=N,
        probe_direction_count=probe_direction_count,
    )
    final_all = sample_n17_with_initial_actions(
        action_head,
        backbone_features=backbone_features,
        state_features=state_features,
        embodiment_id=embodiment_id,
        image_mask=image_mask,
        backbone_attention_mask=backbone_attention_mask,
        initial_actions=noise_all,
    )
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
    ) = compute_trainmask8x7_multi_iid_radius_calibrated_rms_sensitivity_g(
        z0,
        z_probes,
        delta_iid,
        input_perturb_valid_l2,
        sigmas,
    )

    delta_dimension_padding_max_abs = (
        delta_iid[
            :, :, :OFFICIAL_ACTION_HORIZON, VALID_ACTION_DIM:
        ].float().abs().max()
    )
    delta_temporal_padding_max_abs = (
        delta_iid[:, :, OFFICIAL_ACTION_HORIZON:, :].float().abs().max()
    )
    base_noise_dim_padding_nonzero_fraction = (
        (base_noise[:, :OFFICIAL_ACTION_HORIZON, VALID_ACTION_DIM:] != 0)
        .float()
        .reshape(N, -1)
        .mean(dim=1)
    )
    base_noise_time_padding_nonzero_fraction = (
        (base_noise[:, OFFICIAL_ACTION_HORIZON:, :] != 0)
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

    if float(delta_dimension_padding_max_abs) != 0.0:
        raise FloatingPointError("delta dimension padding is not exactly zero")
    if float(delta_temporal_padding_max_abs) != 0.0:
        raise FloatingPointError("delta temporal padding is not exactly zero")
    if bool((probe_changed_fraction_dim_padding != 0).any()):
        raise FloatingPointError("probes changed dimension-padding inputs")
    if bool((probe_changed_fraction_time_padding != 0).any()):
        raise FloatingPointError("probes changed temporal-padding inputs")

    return SimplerEnvMultiIidInferenceResult(
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
        delta_dimension_padding_max_abs=delta_dimension_padding_max_abs,
        delta_temporal_padding_max_abs=delta_temporal_padding_max_abs,
        probe_changed_fraction_active=probe_changed_fraction_active,
        probe_changed_fraction_dim_padding=probe_changed_fraction_dim_padding,
        probe_changed_fraction_time_padding=probe_changed_fraction_time_padding,
        base_noise_dim_padding_nonzero_fraction=(
            base_noise_dim_padding_nonzero_fraction
        ),
        base_noise_time_padding_nonzero_fraction=(
            base_noise_time_padding_nonzero_fraction
        ),
        g_component_finite_fraction=g_component_finite_fraction,
        g_component_zero_fraction=g_component_zero_fraction,
        g_finite_fraction=g_finite_fraction,
        g_zero_fraction=g_zero_fraction,
    )
