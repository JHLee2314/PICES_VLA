"""The pi05 LIBERO inference configuration used by PICES."""
from dataclasses import replace

from openpi.models import pi0_config
from openpi.training import config as training_config


def make_pi05_libero_config():
    return replace(
        training_config.get_config("pi05_libero"),
        name="pi05_libero_h10",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
        ),
        data=training_config.LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=training_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
    )
