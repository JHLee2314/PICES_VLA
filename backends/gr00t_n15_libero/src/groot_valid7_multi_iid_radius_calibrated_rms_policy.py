from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gr00t.model.policy import (
    COMPUTE_DTYPE,
    squeeze_dict_values,
    unsqueeze_dict_values,
)

from groot_valid7_multi_iid_radius_calibrated_rms_core import (
    infer_valid7_multi_iid_radius_calibrated_rms_sensitivity_g,
)
from groot_valid7_multi_iid_radius_calibrated_rms_sensitivity_g_policy import (
    Gr00tValid7MultiIidRadiusCalibratedRmsSensitivityGPolicy,
)
from multi_iid_keepm_first_cross_selector import (
    DEFAULT_CONFIG,
    MultiIidKeepMConfig,
    select_keepm_first_cross_mean,
)


class Gr00tValid7MultiIidRadiusCalibratedRmsPolicy(
    Gr00tValid7MultiIidRadiusCalibratedRmsSensitivityGPolicy
):
    def __init__(
        self,
        *,
        selector_config: MultiIidKeepMConfig = DEFAULT_CONFIG,
        **kwargs: Any,
    ) -> None:
        self.selector_config = selector_config.validate()
        super().__init__(
            N=self.selector_config.N,
            sigma_values=self.selector_config.sigma_values,
            **kwargs,
        )

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        for key, value in obs_copy.items():
            if not isinstance(value, np.ndarray):
                obs_copy[key] = np.array(value)

        normalized_input = self.apply_transforms(obs_copy)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=COMPUTE_DTYPE,
        ):
            result = infer_valid7_multi_iid_radius_calibrated_rms_sensitivity_g(
                self.model,
                normalized_input,
                N=self.selector_config.N,
                sigma_values=self.selector_config.sigma_values,
            )

        normalized_action, selection = self._select_candidate(result)
        unnormalized_action = self._get_unnormalized_action(normalized_action)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        unnormalized_action["n_exec"] = int(selection["n_exec"])
        unnormalized_action["selected_candidate_index"] = int(selection["selected_candidate_index"])
        return unnormalized_action

    def _select_candidate(self, result) -> tuple[torch.Tensor, dict[str, Any]]:
        selection = select_keepm_first_cross_mean(
            result.G.detach().cpu().numpy(),
            self.selector_config,
        )
        selected_index = int(selection["selected_candidate_index"])
        normalized_action = result.z0[selected_index : selected_index + 1].float()
        return normalized_action, selection
