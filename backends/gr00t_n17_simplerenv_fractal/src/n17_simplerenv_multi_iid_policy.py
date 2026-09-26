from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

from n17_simplerenv_multi_iid_core import infer_n17_simplerenv_multi_iid
from n17_simplerenv_multi_iid_selectors import select_method
from n17_simplerenv_multi_iid_config import (
    ACTION_KEYS,
    GOOGLE_EMBODIMENT_TAG,
    MODEL_ACTION_DIM,
    MODEL_ACTION_HORIZON,
    DEFAULT_N,
    OFFICIAL_ACTION_HORIZON,
    SELECTOR_WINDOW,
    build_method_spec,
)


def validate_loaded_policy_contract(
    action_head: Any,
    action_config: Any,
) -> None:
    expected_head = (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM, 4)
    actual_head = (
        int(action_head.action_horizon),
        int(action_head.action_dim),
        int(action_head.num_inference_timesteps),
    )
    if actual_head != expected_head:
        raise ValueError(
            "expected N1.7 action head [H,D,steps]="
            f"{expected_head}, got {actual_head}"
        )
    if list(action_config.delta_indices) != list(
        range(OFFICIAL_ACTION_HORIZON)
    ):
        raise ValueError("SimplerEnv action.delta_indices must be range(8)")
    if tuple(action_config.modality_keys) != ACTION_KEYS:
        raise ValueError(
            f"SimplerEnv action keys must be {ACTION_KEYS}, "
            f"got {tuple(action_config.modality_keys)}"
        )


class Gr00tN17SimplerEnvMultiIidPolicy(Gr00tPolicy):
    def __init__(
        self,
        *,
        method: str,
        sigma_values: Sequence[float],
        N: int = DEFAULT_N,
        selector_window: int = SELECTOR_WINDOW,
        **kwargs: Any,
    ) -> None:
        embodiment_tag = str(
            kwargs.get("embodiment_tag", GOOGLE_EMBODIMENT_TAG)
        )
        if embodiment_tag != GOOGLE_EMBODIMENT_TAG:
            raise ValueError("multi-IID experiment supports Fractal Google only")
        self.multi_iid_spec = build_method_spec(
            method,
            sigma_values,
            N=N,
            selector_window=selector_window,
        )
        super().__init__(**kwargs)
        validate_loaded_policy_contract(
            self.model.action_head,
            self.modality_configs["action"],
        )

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        unbatched_observations = self._unbatch_observation(observation)
        if len(unbatched_observations) != 1:
            raise ValueError("multi-IID inference requires one observation")

        processed_inputs = []
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [
                {
                    "type": MessageType.EPISODE_STEP.value,
                    "content": vla_step_data,
                }
            ]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        if set(collated_inputs) != {"inputs"}:
            raise ValueError(
                f"unexpected N1.7 collator keys: {sorted(collated_inputs)}"
            )
        with torch.inference_mode():
            result = infer_n17_simplerenv_multi_iid(
                self.model,
                collated_inputs["inputs"],
                N=self.multi_iid_spec.N,
                sigma_values=self.multi_iid_spec.sigma_values,
            )
        sensitivity_g_np = self._to_fp32(result.G)
        selector = select_method(sensitivity_g_np, self.multi_iid_spec)
        selected_index = int(selector["selected_candidate_index"])
        normalized_action = result.z0[selected_index : selected_index + 1].float()
        batched_states = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in self.modality_configs["state"].modality_keys
        }
        decoded = self.processor.decode_action(
            normalized_action.cpu().numpy(),
            self.embodiment_tag,
            batched_states,
        )
        if tuple(decoded) != ACTION_KEYS:
            raise ValueError(
                f"decoded action keys must be {ACTION_KEYS}, got {tuple(decoded)}"
            )
        casted_action = {
            key: value.astype(np.float32) for key, value in decoded.items()
        }
        for key, value in casted_action.items():
            if value.shape != (1, OFFICIAL_ACTION_HORIZON, 1):
                raise ValueError(
                    f"decoded action {key!r} must have shape "
                    f"(1,{OFFICIAL_ACTION_HORIZON},1), got {value.shape}"
                )
        return casted_action, {
            "n_exec": int(selector["selected_n_exec"]),
            "selected_candidate_index": selected_index,
        }

    @staticmethod
    def _to_fp32(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
