from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

def _repeat_single_batch(value: Any, repeats: int, path: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim < 1:
            raise ValueError(f"normalized input tensor {path} must have at least one dimension")
        # Most tensors have leading batch 1. Eagle flattens the cameras into the
        # leading dimension (for LIBERO, pixel_values is [2,C,H,W]); repeating the
        # complete leading block preserves [sample0_cam0, sample0_cam1, ...] order.
        factors = (repeats,) + (1,) * (value.ndim - 1)
        return value.repeat(factors)
    if isinstance(value, Mapping):
        return {key: _repeat_single_batch(item, repeats, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_repeat_single_batch(item, repeats, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, list):
        return [_repeat_single_batch(item, repeats, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"normalized input {path} has unsupported type {type(value).__name__}")

def repeat_normalized_input(normalized_input: Mapping[str, Any], repeats: int) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    state = normalized_input.get("state")
    if not isinstance(state, torch.Tensor) or state.ndim < 1 or state.shape[0] != 1:
        shape = tuple(state.shape) if isinstance(state, torch.Tensor) else None
        raise ValueError(f"iid-Gaussian inference requires one normalized observation, got state {shape}")
    return {
        key: _repeat_single_batch(value, repeats, key)
        for key, value in normalized_input.items()
    }

# Copied from FlowmatchingActionHead.get_action in:
# gr00t/model/action_head/flow_matching_action_head.py
# Isaac-GR00T snapshot 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96.
# The behavioral extensions are externally supplied initial_actions and an internal
# flag that avoids processing the same backbone output twice in iid-Gaussian inference.
@torch.no_grad()
def sample_gr00t_with_initial_actions(
    action_head,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    initial_actions: torch.Tensor | None = None,
    *,
    backbone_is_processed: bool = False,
) -> BatchFeature:
    if not backbone_is_processed:
        backbone_output = action_head.process_backbone_output(backbone_output)

    vl_embs = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    state_features = action_head.state_encoder(action_input.state, embodiment_id)

    batch_size = vl_embs.shape[0]
    device = vl_embs.device
    expected_shape = (
        batch_size,
        action_head.config.action_horizon,
        action_head.config.action_dim,
    )
    if initial_actions is None:
        actions = torch.randn(size=expected_shape, dtype=vl_embs.dtype, device=device)
    else:
        if tuple(initial_actions.shape) != expected_shape:
            raise ValueError(
                f"initial_actions must have shape {expected_shape}, got {tuple(initial_actions.shape)}"
            )
        if initial_actions.device != device or initial_actions.dtype != vl_embs.dtype:
            raise ValueError(
                "initial_actions must match backbone device/dtype: "
                f"expected {device}/{vl_embs.dtype}, got "
                f"{initial_actions.device}/{initial_actions.dtype}"
            )
        actions = initial_actions

    num_steps = action_head.num_inference_timesteps
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(f"num_inference_timesteps must be positive, got {num_steps!r}")
    dt = 1.0 / num_steps

    for t in range(num_steps):
        t_cont = t / float(num_steps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device
        )
        action_features = action_head.action_encoder(
            actions, timesteps_tensor, embodiment_id
        )
        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        model_output = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        actions = actions + dt * pred_velocity
    return BatchFeature(data={"action_pred": actions})


