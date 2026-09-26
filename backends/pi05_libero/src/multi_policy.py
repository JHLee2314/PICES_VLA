from __future__ import annotations

# ruff: noqa: SLF001
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi.models import model as _model
from typing_extensions import override

from backends.pi05_libero.src.config import ACTION_DIM, ACTION_HORIZON, DENOISE_STEPS, MAX_EXEC
from backends.pi05_libero.src.multi_config import JAXPerturbMultiIidConfig
from backends.pi05_libero.src.multi_method import (
    build_multi_probe_batch,
    compute_multi_g_and_select,
    split_multi_solution_batch,
)


class JAXPerturbMultiIidPolicy(_base_policy.BasePolicy):
    def __init__(self, policy, config: JAXPerturbMultiIidConfig):
        config.validate()
        if getattr(policy, "_is_pytorch_model", None) is not False:
            raise TypeError("JAX multi-IID requires a native JAX OpenPI policy")
        required_attrs = ("_model", "_sample_actions", "_sample_kwargs", "_rng", "_input_transform", "_output_transform")
        missing = [name for name in required_attrs if not hasattr(policy, name)]
        if missing:
            raise TypeError(f"OpenPI policy is missing required attributes: {', '.join(missing)}")
        model = policy._model
        if (int(model.action_horizon), int(model.action_dim)) != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"pi05 LIBERO requires model action shape [{ACTION_HORIZON},{ACTION_DIM}]")
        self._policy = policy
        self._config = config
        self._action_horizon = int(model.action_horizon)
        self._action_dim = int(model.action_dim)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        obs_copy = dict(obs)
        inputs = self._policy._input_transform(jax.tree.map(lambda x: x, obs_copy))
        inputs_1 = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)

        self._policy._rng, call_rng = jax.random.split(self._policy._rng)
        noise_all, _base_noise, delta_iid, solve_rng = build_multi_probe_batch(
            call_rng,
            self._config.sigma,
            self._config.probe_direction_count,
            N=self._config.N,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
        )
        total_solve_candidates = self._config.total_solve_candidates
        inputs_all = jax.tree.map(lambda x: jnp.repeat(x, total_solve_candidates, axis=0), inputs_1)
        observation_all = _model.Observation.from_dict(inputs_all)

        sample_kwargs = dict(self._policy._sample_kwargs)
        sample_kwargs.update(noise=noise_all, num_steps=DENOISE_STEPS)
        z_all = self._policy._sample_actions(solve_rng, observation_all, **sample_kwargs)
        z0, z_probes = split_multi_solution_batch(
            z_all,
            self._config.probe_direction_count,
            N=self._config.N,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
        )
        stats = compute_multi_g_and_select(
            z0,
            z_probes,
            delta_iid,
            self._config.sigma,
            N=self._config.N,
            max_exec=MAX_EXEC,
            window_size=self._config.window_size,
        )
        selected_idx = int(np.asarray(jax.device_get(stats["selected_index"])).item())
        n_exec = int(np.asarray(jax.device_get(stats["n_exec"])).item())

        outputs = {
            "state": inputs_1["state"],
            "actions": z0[selected_idx : selected_idx + 1],
        }
        outputs = jax.tree.map(lambda x: np.asarray(jax.device_get(x[0, ...])), outputs)
        outputs = self._policy._output_transform(outputs)
        full_actions = np.asarray(outputs["actions"])
        outputs["actions"] = full_actions[:n_exec]
        outputs["n_exec"] = n_exec
        outputs["selected_candidate_index"] = selected_idx
        return outputs

    @property
    @override
    def metadata(self) -> dict:
        return dict(self._policy.metadata)


def apply_jax_perturb_multi_iid_to_openpi_policy(
    policy,
    config: JAXPerturbMultiIidConfig | None = None,
):
    return JAXPerturbMultiIidPolicy(
        policy, JAXPerturbMultiIidConfig() if config is None else config
    )
