# Model environment and checkpoint setup

Prepare a Python environment and checkpoint for each model using the corresponding upstream project's installation instructions. Add the PICES source to that environment using the path settings in [INTEGRATION.md](INTEGRATION.md). The caller specifies the checkpoint path; checkpoints do not need to be copied into PICES.

## Upstream sources

| Target | Model project | Revision at source extraction |
|---|---|---|
| pi05–LIBERO | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `40972b37e2f38cc3a65d7034c4e2f0a8113423bc` |
| GR00T N1.5–LIBERO | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.5 | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` |
| GR00T N1.7–Fractal | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.7 | `b9955401d50c92a29258732e3ad6ccd579f1bdc0` |

These revisions identify the sources. The checkouts contained local modifications when the code was extracted, so matching the public SHA alone does not guarantee an identical environment. Compatibility with the latest main branch is not assumed. Use the corresponding model integration in the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) or [SimplerEnv](https://github.com/simpler-env/SimplerEnv) evaluator.

## Model environments

The versions below were used in the existing model environments. Full dependency locks and Python environment copies are not included. N1.5 and N1.7 use different `gr00t` APIs and require separate environments.

| Backend | Python | Inference library | Transformers | NumPy |
|---|---|---|---|---|
| pi05–LIBERO | 3.11.15 | JAX/jaxlib 0.5.3 | 4.53.2 | 1.26.4 |
| N1.5–LIBERO | 3.10.20 | Torch 2.9.1+cu130 | 4.51.3 | 1.26.4 |
| N1.7–Fractal | 3.12.3 | Torch 2.9.0+cu128 | 4.57.3 | 1.26.4 |

The pi05 environment must have OpenPI and `openpi_client` installed. Each GR00T environment must have its corresponding `gr00t` version and model dependencies installed. The model code itself does not import a simulator. The version table alone does not reproduce an identical binary environment.

## Weight sources

| Model | Public source | Source revision |
|---|---|---|
| pi05–LIBERO | [OpenPI checkpoint documentation](https://github.com/Physical-Intelligence/openpi#model-checkpoints), `gs://openpi-assets/checkpoints/pi05_libero` | GCS release path; no Git revision |
| N1.5 Spatial | [youliangtan/gr00t-n1.5-libero-spatial-posttrain](https://huggingface.co/youliangtan/gr00t-n1.5-libero-spatial-posttrain) | `03294ae3de7d4870be8ae6e27cbdeffcb46b6425` |
| N1.5 Goal | [youliangtan/gr00t-n1.5-libero-goal-posttrain](https://huggingface.co/youliangtan/gr00t-n1.5-libero-goal-posttrain) | `17be45f5bf8a543594761b263c68cf33f33847bd` |
| N1.5 Object | [youliangtan/gr00t-n1.5-libero-object-posttrain](https://huggingface.co/youliangtan/gr00t-n1.5-libero-object-posttrain) | `f422266b016fd917127d3a0348062a5cd750df43` |
| N1.5 Long | [youliangtan/gr00t-n1.5-libero-long-posttrain](https://huggingface.co/youliangtan/gr00t-n1.5-libero-long-posttrain) | `aa49078d5cc9ce72917bc4312f1ef12771f277de` |
| N1.7 Fractal | [nvidia/GR00T-N1.7-SimplerEnv-Fractal](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Fractal) | `fb8357ff66e3cc8dec369e2bf7dbe9e6740511b2` |
| N1.7 Cosmos dependency | [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | `9ce19a195e423419c349abfc86fd07178b230561` |

Prepare the complete checkpoint layout required by upstream, including model configuration, normalization statistics, tokenizer and processor files, as well as tensor files. pi05 requires JAX `params/`, the checkpoint's `assets/`, and OpenPI tokenizer assets. N1.5 requires the `experiment_cfg/` directory for each suite. For N1.7, the Fractal processor and Cosmos model/tokenizer references must resolve in the prepared environment.

PICES does not download checkpoints or modify their internal paths. If a relocated checkpoint contains old local absolute paths, update those settings for the model environment you are using. Weights and file hash inventories are not included in this repository.

## N1.5 normalization by suite

| LIBERO suite | Weight | PICES data config | Action normalization |
|---|---|---|---|
| `libero_spatial` | Spatial | `LiberoDataConfig` | Min-max for all dimensions |
| `libero_goal` | Goal | `LiberoDataConfigMeanStd` | Mean-std for x/y/z/roll/pitch/yaw; min-max for gripper |
| `libero_object` | Object | `LiberoDataConfig` | Min-max for all dimensions |
| `libero_10` | Long | `LiberoDataConfig` | Min-max for all dimensions |

The configurations are in [custom_data_config.py](../backends/gr00t_n15_libero/src/custom_data_config.py). Select each checkpoint together with its matching data config. All suites use `new_embodiment`, a model horizon of 16, and 4 denoising steps.

## Model integration code extracted into PICES

- The pi05 [model configuration](../backends/pi05_libero/src/model_config.py) constructs the h10 and continuous-state settings. The adapter uses the native JAX policy's `_sample_actions`, `_rng`, preprocessing, and postprocessing interfaces.
- The N1.5 [solver](../backends/gr00t_n15_libero/src/groot_solver_utils.py) passes an initial noise batch into the original flow-matching loop. It calls `prepare_input`, `process_backbone_output`, and the action encoder/decoder without requiring a separate upstream sampler patch. The data config and LIBERO input/output transforms are also retained in PICES.
- The N1.7 [core](../backends/gr00t_n17_simplerenv_fractal/src/n17_simplerenv_multi_iid_core.py) uses `_encode_features` and the action head to compute the probe batch. It requires the native `Gr00tPolicy` processor, collator, and `decode_action` interfaces, along with `Gr00tSimPolicyWrapper`. Google/Fractal action settings must use a 40×132 model space, 8-step decoding, and 4 denoising steps.

During cleanup, imports and adapters with mock models were checked against prepared upstream packages, and computation results were compared using explicit tensors. Checkpoint loading and actual GPU inference were not rerun. In a new upstream installation, first verify these interfaces and a short inference run using the [integration examples](INTEGRATION.md).
