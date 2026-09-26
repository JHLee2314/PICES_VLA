# Source attribution and licenses for retained code

PICES is licensed under the [Apache License, Version 2.0](../LICENSE). Third-party code retains the original notices listed below.

This directory preserves the original licenses for upstream code used in PICES model integration, preprocessing, postprocessing, and solvers. Source revisions for public code and model checkpoints are listed in [SETUP.md](../docs/SETUP.md). Weights are not included and remain subject to the terms of their distributors.

| Source | Use in PICES | License |
|---|---|---|
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Code used for pi05 model configuration and native policy integration | [Apache-2.0 license text](OPENPI_LICENSE) |
| [NVIDIA/Isaac-GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) | Code extracted and adapted from the LIBERO data config, preprocessing, postprocessing, and flow-matching solver | [Apache-2.0 license text](GROOT_N15_LICENSE) |
| [NVIDIA/Isaac-GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) | Code extracted and adapted from native processor integration and the flow-matching loop | [Apache-2.0 license text](GROOT_N17_LICENSE) |
| [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) | `quat2axisangle` in the N1.5 `observation_transforms.py` | [MIT license and included-code notices](ROBOSUITE_LICENSE) |

PICES adapters and samplers include changes that connect multi-IID probe batches, sensitivity G computation, and candidate/execution-length selection. Cleanup removed server and logging dependencies and fixed-seed contracts, and extracted model configuration and input/output transforms into each backend's `src/` directory. Copyright and SPDX notices in copied files are preserved.
