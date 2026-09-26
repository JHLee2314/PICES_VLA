from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from backends.pi05_libero.src.config import ACTION_DIM
from backends.pi05_libero.src.config import ACTION_HORIZON
from backends.pi05_libero.src.config import FIRST_CROSS_WINDOW
from backends.pi05_libero.src.config import MAX_EXEC
from backends.pi05_libero.src.config import DEFAULT_N
from backends.pi05_libero.src.config import validate_max_exec
from backends.pi05_libero.src.config import validate_positive_real
from backends.pi05_libero.src.multi_config import validate_candidate_count
from backends.pi05_libero.src.multi_config import validate_probe_direction_count
from backends.pi05_libero.src.multi_config import validate_window_size


def build_multi_probe_batch(
    call_rng: jax.Array,
    sigma: float,
    probe_direction_count: int,
    *,
    N: int = DEFAULT_N,
    action_horizon: int = ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build a shared-base batch with independent one-sided IID perturbations."""
    sigma = validate_positive_real(sigma, "sigma")
    count = validate_probe_direction_count(probe_direction_count, "probe_direction_count")
    candidate_count = validate_candidate_count(N, "N")
    return _build_multi_probe_batch_jit(
        call_rng,
        sigma,
        probe_direction_count=count,
        N=candidate_count,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )


@partial(
    jax.jit,
    static_argnames=("probe_direction_count", "N", "action_horizon", "action_dim"),
)
def _build_multi_probe_batch_jit(
    call_rng: jax.Array,
    sigma: float,
    *,
    probe_direction_count: int,
    N: int,
    action_horizon: int,
    action_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    keys = jax.random.split(call_rng, probe_direction_count + 2)
    shape = (N, action_horizon, action_dim)
    base_noise = jax.random.normal(keys[0], shape, dtype=jnp.float32)
    delta_iid = jax.vmap(lambda key: jax.random.normal(key, shape, dtype=jnp.float32))(
        keys[1 : probe_direction_count + 1]
    )
    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    probe_noise = base_noise[None, ...] + sigma * delta_iid
    noise_all = jnp.concatenate(
        (base_noise, probe_noise.reshape(-1, action_horizon, action_dim)),
        axis=0,
    )
    return noise_all, base_noise, delta_iid, keys[-1]


def split_multi_solution_batch(
    z_all: jax.Array,
    probe_direction_count: int,
    *,
    N: int = DEFAULT_N,
    action_horizon: int = ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
) -> tuple[jax.Array, jax.Array]:
    count = validate_probe_direction_count(probe_direction_count, "probe_direction_count")
    candidate_count = validate_candidate_count(N, "N")
    expected_shape = ((count + 1) * candidate_count, action_horizon, action_dim)
    if tuple(z_all.shape) != expected_shape:
        raise ValueError(f"Expected solution shape {expected_shape}, got {tuple(z_all.shape)}")
    blocks = z_all.reshape(count + 1, candidate_count, action_horizon, action_dim)
    return blocks[0], blocks[1:]


def compute_multi_radius_calibrated_rms_g(
    z0: jax.Array,
    z_probes: jax.Array,
    delta_iid: jax.Array,
    sigma: float,
    *,
    N: int = DEFAULT_N,
) -> dict[str, jax.Array]:
    """Compute equal-sigma, per-direction radius-calibrated RMS sensitivity G."""
    candidate_count = validate_candidate_count(N, "N")
    expected_base_shape = tuple(z0.shape)
    if len(expected_base_shape) != 3 or expected_base_shape[0] != candidate_count:
        raise ValueError(
            f"z0 must have shape [{candidate_count},H,D], got {expected_base_shape}"
        )
    if z_probes.ndim != 4:
        raise ValueError(
            "z_probes must have shape "
            f"[*,{candidate_count},H,D], got {tuple(z_probes.shape)}"
        )
    count = validate_probe_direction_count(z_probes.shape[0], "probe_direction_count")
    expected_probe_shape = (count, *expected_base_shape)
    for name, value in (("z_probes", z_probes), ("delta_iid", delta_iid)):
        if tuple(value.shape) != expected_probe_shape:
            raise ValueError(f"{name} must have shape {expected_probe_shape}, got {tuple(value.shape)}")

    sigma = validate_positive_real(sigma, "sigma")
    stats = _compute_multi_radius_calibrated_rms_g_jit(z0, z_probes, delta_iid, sigma)
    _validate_g_stats(stats)
    return stats


@jax.jit
def _compute_multi_radius_calibrated_rms_g_jit(
    z0: jax.Array,
    z_probes: jax.Array,
    delta_iid: jax.Array,
    sigma: float,
) -> dict[str, jax.Array]:
    count = z_probes.shape[0]
    N = z0.shape[0]
    delta_f32 = delta_iid.astype(jnp.float32)
    radius = jnp.linalg.norm(delta_f32.reshape(count, N, -1), axis=-1)
    radius_reference = jnp.sqrt(jnp.asarray(z0.shape[1] * z0.shape[2], dtype=jnp.float32))

    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    sigma_values = jnp.full((count,), sigma, dtype=jnp.float32)
    sigma_components = jnp.linalg.norm(
        z_probes.astype(jnp.float32) - z0.astype(jnp.float32)[None, ...],
        axis=-1,
    ) / sigma

    radius_scale = radius_reference / radius
    radius_calibrated_components = sigma_components * radius_scale[..., None]
    sensitivity_g = jnp.sqrt(jnp.mean(jnp.square(radius_calibrated_components), axis=0))

    return {
        "G": sensitivity_g,
        "sigma_values": sigma_values,
        "delta_l2": radius,
        "input_perturb_l2": sigma_values[:, None] * radius,
        "radius_reference": radius_reference,
        "radius_scale": radius_scale,
        "G_sigma_components": sigma_components,
        "G_radius_calibrated_components": radius_calibrated_components,
    }


def _validate_g_stats(stats: dict[str, jax.Array]) -> None:
    radius = np.asarray(jax.device_get(stats["delta_l2"]), dtype=np.float32)
    if not np.all(np.isfinite(radius)) or not np.all(radius > 0.0):
        raise RuntimeError("invalid iid Gaussian probe L2 radius")
    for name in (
        "input_perturb_l2",
        "radius_reference",
        "radius_scale",
        "G_sigma_components",
        "G_radius_calibrated_components",
        "G",
    ):
        value = np.asarray(jax.device_get(stats[name]))
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite multi-IID sensitivity G statistic: {name}")


def select_multi_from_sensitivity_g(
    sensitivity_g: jax.Array,
    *,
    N: int = DEFAULT_N,
    max_exec: int = MAX_EXEC,
    window_size: int = FIRST_CROSS_WINDOW,
) -> dict[str, jax.Array]:
    candidate_count = validate_candidate_count(N, "N")
    if sensitivity_g.ndim != 2 or sensitivity_g.shape[0] != candidate_count:
        raise ValueError(f"sensitivity G must have shape [{candidate_count},H], got {sensitivity_g.shape}")
    parsed_max_exec = validate_max_exec(max_exec, sensitivity_g.shape[1])
    window = validate_window_size(window_size, max_exec=parsed_max_exec)
    return _select_multi_from_sensitivity_g_jit(
        sensitivity_g,
        max_exec=parsed_max_exec,
        window_size=window,
    )


@partial(jax.jit, static_argnames=("max_exec", "window_size"))
def _select_multi_from_sensitivity_g_jit(
    sensitivity_g: jax.Array,
    *,
    max_exec: int,
    window_size: int,
) -> dict[str, jax.Array]:
    N = sensitivity_g.shape[0]
    M = (N + 1) // 2  # ceil(N / 2); not a configurable parameter.
    candidate_idx = jnp.arange(N, dtype=jnp.int32)
    prefix_mean = jnp.mean(sensitivity_g[:, :window_size], axis=1)

    prefix_order = jnp.lexsort((candidate_idx, prefix_mean))
    kept_indices = prefix_order[:M]

    later_gt = sensitivity_g[:, window_size:max_exec] > prefix_mean[:, None]
    has_crossing = jnp.any(later_gt, axis=1)
    first_relative = jnp.argmax(later_gt.astype(jnp.int32), axis=1)
    first_cross_step = jnp.where(
        has_crossing,
        window_size + first_relative,
        jnp.full((N,), max_exec, dtype=jnp.int32),
    )

    kept_cross = first_cross_step[kept_indices]
    kept_prefix = prefix_mean[kept_indices]
    kept_order = jnp.lexsort((kept_indices, kept_prefix, -kept_cross))
    selected_index = kept_indices[kept_order[0]]
    n_exec = first_cross_step[selected_index]

    return {
        "prefix_mean": prefix_mean,
        "first_cross_step": first_cross_step,
        "kept_indices": kept_indices,
        "selected_index": selected_index,
        "n_exec": n_exec,
    }


def compute_multi_g_and_select(
    z0: jax.Array,
    z_probes: jax.Array,
    delta_iid: jax.Array,
    sigma: float,
    *,
    N: int = DEFAULT_N,
    max_exec: int = MAX_EXEC,
    window_size: int = FIRST_CROSS_WINDOW,
) -> dict[str, jax.Array]:
    candidate_count = validate_candidate_count(N, "N")
    stats = compute_multi_radius_calibrated_rms_g(
        z0,
        z_probes,
        delta_iid,
        sigma,
        N=candidate_count,
    )
    selection = select_multi_from_sensitivity_g(
        stats["G"],
        N=candidate_count,
        max_exec=max_exec,
        window_size=window_size,
    )
    return {**stats, **selection}


