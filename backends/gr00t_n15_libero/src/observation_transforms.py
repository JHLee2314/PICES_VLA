"""LIBERO observation and action conversions; no simulator dependency."""
import math
import numpy as np

ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    wrist_img = obs["robot0_eye_in_hand_image"]
    wrist_img = wrist_img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing

    return img, wrist_img


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].

    Normalization formula: y = 1 - 2 * (x - orig_low) / (orig_high - orig_low)
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action


def prepare_observation(obs: dict, language: str) -> dict:
    xyz = np.asarray(obs["robot0_eef_pos"])
    rpy = quat2axisangle(np.asarray(obs["robot0_eef_quat"]).copy())
    gripper = np.asarray(obs["robot0_gripper_qpos"])
    image, wrist_image = get_libero_image(obs)
    return {
        "video.image": image[None, ...],
        "video.wrist_image": wrist_image[None, ...],
        "state.x": np.array([[xyz[0]]]),
        "state.y": np.array([[xyz[1]]]),
        "state.z": np.array([[xyz[2]]]),
        "state.roll": np.array([[rpy[0]]]),
        "state.pitch": np.array([[rpy[1]]]),
        "state.yaw": np.array([[rpy[2]]]),
        "state.gripper": gripper[None, ...],
        "annotation.human.action.task_description": [language],
    }


def action_chunk(response: dict, native_action_horizon: int = 16) -> np.ndarray:
    columns = []
    for name in ACTION_KEYS:
        key = f"action.{name}"
        if key not in response:
            raise KeyError(f"GR00T response is missing {key}")
        value = np.asarray(response[key])
        if value.ndim == 0 or value.shape[0] != native_action_horizon:
            raise ValueError(
                f"{key} must have leading horizon {native_action_horizon}, got {value.shape}"
            )
        value = value.reshape(native_action_horizon, -1)
        if value.shape[1] != 1:
            raise ValueError(f"{key} must be scalar per action step, got {value.shape}")
        columns.append(value[:, 0])
    action = np.stack(columns, axis=-1).astype(np.float32, copy=False)
    if not np.isfinite(action).all():
        raise ValueError("GR00T action chunk contains NaN or Inf")
    normalize_gripper_action(action, binarize=True)
    if action.shape != (native_action_horizon, 7):
        raise ValueError(f"expected action shape ({native_action_horizon}, 7), got {action.shape}")
    return action
