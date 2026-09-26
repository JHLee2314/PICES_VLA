# PICES

Model code that applies **PICES** to pi05 and GR00T. It includes IID probe generation, sensitivity G computation, candidate and execution-length selection, minimal policy adapters, and model input/output configuration.

Create an PICES adapter in your prepared upstream model environment, then use the existing evaluator to execute the returned action prefix. The upstream project loads the base model and checkpoint.

- [Model integration](docs/INTEGRATION.md): model construction and inference examples for the three backends, including how to use `n_exec`.
- [Environment and weight setup](docs/SETUP.md): upstream and checkpoint sources, revisions, and normalization for each suite.

## Core code

| Target | Computation | Model integration and configuration |
|---|---|---|
| pi05–LIBERO | [multi_method.py](backends/pi05_libero/src/multi_method.py) | [policy](backends/pi05_libero/src/multi_policy.py), [method configuration](backends/pi05_libero/src/multi_config.py), [h10 model configuration](backends/pi05_libero/src/model_config.py) |
| GR00T N1.5–LIBERO | [core](backends/gr00t_n15_libero/src/groot_valid7_multi_iid_radius_calibrated_rms_core.py), [selector](backends/gr00t_n15_libero/src/multi_iid_keepm_first_cross_selector.py) | [policy](backends/gr00t_n15_libero/src/groot_valid7_multi_iid_radius_calibrated_rms_policy.py), [data config](backends/gr00t_n15_libero/src/custom_data_config.py), [input/output transforms](backends/gr00t_n15_libero/src/observation_transforms.py) |
| GR00T N1.7–SimplerEnv Fractal | [core](backends/gr00t_n17_simplerenv_fractal/src/n17_simplerenv_multi_iid_core.py), [selector](backends/gr00t_n17_simplerenv_fractal/src/n17_simplerenv_multi_iid_selectors.py) | [policy](backends/gr00t_n17_simplerenv_fractal/src/n17_simplerenv_multi_iid_policy.py), [single configuration module](backends/gr00t_n17_simplerenv_fractal/src/n17_simplerenv_multi_iid_config.py) |

Each backend's essential source code is retained as physical files under `backends/<backend>/src/`. N1.7 configuration and validation functions are also consolidated in the same `src/` directory.

## Default settings

The configurable defaults are **sigma=0.25 and N=3**. N is the number of nominal candidates.

| Backend | W | Maximum execution length |
|---|---:|---:|
| pi05–LIBERO | 5 | 10 |
| GR00T N1.5–LIBERO | 1 | 16 |
| GR00T N1.7–Fractal | 1 | 8 |

Random sampling and JAX key splitting are preserved. PICES does not fix seeds or reset the RNG at episode boundaries.

## Repository scope

This repository provides model code, minimal documentation, and [source attribution and licenses](licenses/README.md). It does not include model servers, RPC clients, evaluators, full upstream source trees, weights, Python environments, or evaluation records. It does not automatically locate, run, or import specific local experiment checkouts.

The cleanup was checked through computation comparisons with explicit input tensors, imports against prepared upstream packages, and adapter input/output checks with mock models. Inference with actual checkpoints and full evaluations were not rerun after the cleanup. In a new environment, first check weight loading and a short inference run using the [integration examples](docs/INTEGRATION.md).

## License

PICES is licensed under the [Apache License, Version 2.0](LICENSE). Third-party code retains its original copyright and license notices; see [source attribution and licenses](licenses/README.md).
