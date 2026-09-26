# Integrating PICES with upstream models

Use the policies below at the point where your existing evaluator creates its model. The original evaluator handles simulator creation, task management, and success criteria. PICES computes candidate actions and an execution length for a single observation. See [SETUP.md](SETUP.md) for the required environments and checkpoints.

Run each example in its corresponding model's Python environment. Set `PICES_ROOT` to your PICES installation path. Add only the PICES source directories to `PYTHONPATH`; use the previously installed upstream packages for `openpi`, `openpi_client`, and `gr00t`. Use separate environments for the N1.5 and N1.7 versions of `gr00t`.

The nominal candidate count is configured with `N`. The retained count is always `M = ceil(N / 2)`; there is no separate M setting.

## pi05–LIBERO

```bash
export PICES_ROOT=/path/to/PICES
export PYTHONPATH="$PICES_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

```python
from openpi.policies import policy_config
from backends.pi05_libero.src.model_config import make_pi05_libero_config
from backends.pi05_libero.src.multi_config import JAXPerturbMultiIidConfig
from backends.pi05_libero.src.multi_policy import (
    apply_jax_perturb_multi_iid_to_openpi_policy,
)

base_policy = policy_config.create_trained_policy(
    make_pi05_libero_config(),
    "/path/to/pi05_libero",  # Directory containing JAX params/ and checkpoint assets/
)
policy = apply_jax_perturb_multi_iid_to_openpi_policy(
    base_policy,
    JAXPerturbMultiIidConfig(
        sigma=0.25,
        N=3,  # Nominal candidates; M = ceil(N / 2) is derived automatically
        window_size=5,
    ),
)

# Build observation with the existing LIBERO preprocessing, following the input contract below.
response = policy.infer(observation)
prefix = response["actions"]  # Environment actions already truncated to (n_exec, 7)
assert len(prefix) == response["n_exec"]
```

`observation` is the native OpenPI LIBERO policy input. `observation/image` and `observation/wrist_image` are RGB images after the existing evaluator's rotation and resizing. `observation/state` is an 8-dimensional state: position 3 + axis-angle 3 + gripper 2. `prompt` is the task string. The native policy's input transform performs normalization and tokenization, so do not apply them again before the call.

[model_config.py](../backends/pi05_libero/src/model_config.py) sets `action_horizon=10`, `discrete_state_input=False`, and `extra_delta_transform=False`. It does not require a separate `pi05_libero_h10` entry in the upstream registry. The adapter uses the native JAX policy's sampler, preprocessing, postprocessing, and current RNG, advancing the key on every call. The model space is 10×32, with 10 denoising steps.

The response contains `actions`, `n_exec`, `selected_candidate_index`, and any fields retained by the native output transform. The selected actions have already passed through the native output transform.

## GR00T N1.5–LIBERO

```bash
export PICES_ROOT=/path/to/PICES
export PYTHONPATH="$PICES_ROOT/backends/gr00t_n15_libero/src${PYTHONPATH:+:$PYTHONPATH}"
```

```python
from custom_data_config import LiberoDataConfig, LiberoDataConfigMeanStd
from groot_valid7_multi_iid_radius_calibrated_rms_policy import (
    Gr00tValid7MultiIidRadiusCalibratedRmsPolicy,
)
from multi_iid_keepm_first_cross_selector import MultiIidKeepMConfig
from observation_transforms import prepare_observation, action_chunk

# This example uses the Spatial checkpoint. For Goal, use LiberoDataConfigMeanStd().
data_config = LiberoDataConfig()
policy = Gr00tValid7MultiIidRadiusCalibratedRmsPolicy(
    model_path="/path/to/gr00t-n1.5-libero-spatial-posttrain",
    embodiment_tag="new_embodiment",
    modality_config=data_config.modality_config(),
    modality_transform=data_config.transform(),
    denoising_steps=4,
    device="cuda:0",
    selector_config=MultiIidKeepMConfig(
        sigma_values=(0.25,),
        N=3,  # Nominal candidates; M = ceil(N / 2) is derived automatically
        prefix_window=1,
        max_exec=16,
    ),
)

# obs is the raw LIBERO observation dict; language is the task string.
model_observation = prepare_observation(obs, language)
response = policy.get_action(model_observation)
actions = action_chunk(response)  # (16, 7), including the LIBERO gripper conversion
prefix = actions[:response["n_exec"]]
```

[prepare_observation](../backends/gr00t_n15_libero/src/observation_transforms.py) takes `agentview_image`, `robot0_eye_in_hand_image`, `robot0_eef_pos`, `robot0_eef_quat`, and `robot0_gripper_qpos`. It rotates images by 180 degrees, converts quaternions to axis-angle, and assembles the `video.*`, `state.*`, and language inputs. The native data transform handles model normalization and image processing.

Each `action.x/y/z/roll/pitch/yaw/gripper` response field has shape `(16, 1)`. `n_exec` and `selected_candidate_index` are integers. `action_chunk` combines the columns and converts the gripper to LIBERO's ±1 format, so do not apply that conversion again. This example processes a single observation.

Spatial, Object, and Long use `LiberoDataConfig`; Goal uses `LiberoDataConfigMeanStd`. Goal also uses min-max normalization for the gripper. Preserve the full [suite–checkpoint mapping](SETUP.md#n15-normalization-by-suite).

## GR00T N1.7–SimplerEnv Fractal

```bash
export PICES_ROOT=/path/to/PICES
export PYTHONPATH="$PICES_ROOT/backends/gr00t_n17_simplerenv_fractal/src${PYTHONPATH:+:$PYTHONPATH}"
```

```python
import numpy as np
from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
from n17_simplerenv_multi_iid_config import ACTION_KEYS, GOOGLE_EMBODIMENT_TAG, METHOD
from n17_simplerenv_multi_iid_policy import Gr00tN17SimplerEnvMultiIidPolicy

core_policy = Gr00tN17SimplerEnvMultiIidPolicy(
    model_path="/path/to/GR00T-N1.7-SimplerEnv-Fractal",
    embodiment_tag=GOOGLE_EMBODIMENT_TAG,
    device="cuda:0",
    strict=True,
    method=METHOD,
    sigma_values=(0.25,),
    N=3,  # Nominal candidates; M = ceil(N / 2) is derived automatically
    selector_window=1,
)
policy = Gr00tSimPolicyWrapper(core_policy, strict=True)

# observation is the flat model observation from the existing Google/Fractal preprocessing.
action_fields, info = policy.get_action(observation)
actions = np.concatenate(
    [action_fields[f"action.{key}"][0] for key in ACTION_KEYS],
    axis=-1,
)  # (8, 7), model actions after processor.decode_action
prefix = actions[:info["n_exec"]]
```

Fractal inputs consist of a `video.image` RGB array with shape `(1, 1, H, W, 3)`, scalar `state.x/y/z/rx/ry/rz/rw/gripper` arrays with shape `(1, 1, 1)`, and a `task` string list `[language]`. Use `uint8` for images and `float32` for state. The exact modality keys and temporal indices follow the checkpoint processor's configuration. The example above uses a batch size of 1.

`Gr00tSimPolicyWrapper` converts the flat input to the native format. The PICES policy uses the upstream processor for normalization and action decoding. Each `action.*` response array has shape `(1, 8, 1)`, and `info` contains `n_exec` and `selected_candidate_index`. Apply the Google environment's translation, rotation, and gripper postprocessing in the existing evaluator as before.

The policy checks for a 40×132 model space, an official action horizon of 8, and 4 denoising steps. Sensitivity G uses the actual `probe-base` radius in the active 8×7 region after BF16 quantization. The core, selector, and policy directly import the [single configuration module](../backends/gr00t_n17_simplerenv_fractal/src/n17_simplerenv_multi_iid_config.py) in the same `src/` directory. No per-request seed or metadata version selection is required.

## Execution length in the evaluator

Put the `prefix` from each example into the existing action queue. **Consume exactly `n_exec` steps**, then run inference again with a new observation. Discard any remaining queued actions at the end of an episode. Truncating to a fixed horizon or running inference at every step would bypass the execution length selected by PICES.

PICES does not fix the global RNG or reset it for each episode. The JAX adapter advances the base policy's RNG, and the Torch backends generate independent Gaussian probes from the normal RNG state. The caller's model environment manages RNG initialization.

Cleanup validation covered computation comparisons, upstream imports, and adapter input/output checks with mock models. Actual checkpoint loading, GPU inference, and simulator rollouts need to be checked separately in the new environment.
