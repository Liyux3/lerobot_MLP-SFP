# Streaming Flow Policy with MLP Velocity Networks

Implementation of [Streaming Flow Policy](https://arxiv.org/abs/2505.11018) (SFP) for robot manipulation, built on the [LeRobot](https://github.com/huggingface/lerobot) framework. Instead of iterative diffusion denoising, SFP learns a velocity field and integrates it with a single Euler pass, making inference significantly faster. This fork replaces the original UNet velocity network with lightweight MLP variants and benchmarks 7 architectures on a real-robot push-T task.

## What's Different from Upstream LeRobot

All SFP-specific code lives in `src/lerobot/policies/sfp/`:

- **`configuration_sfp.py`** configures SFP hyperparameters (flow coupling constant *k*, noise scale *sigma_0*, integration steps, velocity net type)
- **`modeling_sfp.py`** (~450 lines) implements the full SFP policy: observation encoding via a shared ResNet18 backbone, flow-matching loss, and Euler integration at inference
- **`velocity_nets.py`** (~660 lines) defines 7 velocity network architectures, from simple MLPs to conditional UNets
- **`processor_sfp.py`** adds Savitzky-Golay smoothing for demonstration velocity labels

Additional files: `train_sfp_batch.sh` for parallel multi-GPU ablation, `train_sfp_frozen_encoder.py` for encoder transfer from a pretrained Diffusion Policy, `run_policy.py` for real-robot deployment on SO-101.

## Velocity Network Architectures

| ID | Architecture | Update (ms) | Final Loss | Notes |
|----|-------------|------------|-----------|-------|
| v1 | `bottleneck_skip` | 40 | 0.025 | Fixed-dim MLP with bottleneck + skip |
| v2 | `bottleneck_skip_auto` | 40 | 0.025 | Auto-scaled dims from input size |
| v3 | `unet_dp` | 77 | 0.017 | Diffusion Policy's original 1D UNet |
| v4 | `pure_mlp` | 40 | 0.025 | Vanilla MLP, no skip or bottleneck |
| v5 | `cond_every_layer` | 40 | 0.022 | Time embedding injected at every layer |
| v6 | `cond_residual` | 40 | 0.021 | Residual blocks with FiLM conditioning |
| v7 | `unet` | 52 | 0.019 | Standalone conditional 1D UNet |
| v8 | frozen encoder | 39 | 0.030 | Encoder transferred from pretrained DP |

All variants trained for 50K steps on a real-robot push-T dataset (2x RTX 4090, batch size 32, cosine schedule with 500-step warmup).

**Key finding:** MLP variants (v5, v6) achieve comparable loss to UNets at nearly half the per-step compute. The `cond_residual` network (v6) hits the best accuracy-speed tradeoff, with FiLM-conditioned residual blocks capturing temporal structure without the overhead of transposed convolutions.

## How SFP Works

Standard diffusion policies iterate 10-100 denoising steps at inference. SFP reformulates this as a flow matching problem: learn a velocity field *v(x, t)* that transports noise to actions in one integration pass.

```
Training:  sample t ~ U[0,1], compute flow velocity, regress v_theta against it
Inference: x_0 ~ N(0, sigma^2 I), then x_1 = x_0 + integral(v_theta(x_t, t), dt)
           with k-step Euler integration (default k=4)
```

The coupling constant *k* controls how strongly the flow attaches to the last predicted action (streaming behavior), enabling smooth transitions between action chunks without temporal discontinuities.

## Quick Start

```bash
pip install -e ".[dev]"

# Train SFP with default bottleneck_skip MLP
lerobot-train \
    --policy.type=sfp \
    --policy.velocity_net_type=bottleneck_skip \
    --dataset.repo_id=your_dataset \
    --output_dir=./outputs/sfp_v1

# Run all 7 architectures in parallel (needs 2 GPUs)
bash train_sfp_batch.sh

# Transfer encoder from pretrained Diffusion Policy
python train_sfp_frozen_encoder.py

# Deploy on SO-101 robot
python run_policy.py
```

## Training Configuration

Key hyperparameters in `SFPConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `velocity_net_type` | `bottleneck_skip` | Which velocity network to use |
| `k` | 2.0 | Flow coupling constant |
| `sigma_0` | 1.0 | Initial noise scale |
| `integration_steps` | 4 | Euler steps at inference |
| `n_obs_steps` | 10 | Observation history length |
| `n_action_steps` | 20 | Action chunk size |
| `use_smooth_velocity` | False | Savitzky-Golay velocity smoothing |

## Project Structure

```
src/lerobot/policies/sfp/
    configuration_sfp.py    # SFPConfig dataclass
    modeling_sfp.py         # SFPPolicy, SFPModel (encoder + velocity net)
    velocity_nets.py        # 7 velocity network architectures
    processor_sfp.py        # Smooth velocity preprocessing
train_sfp_batch.sh          # Multi-GPU ablation script
train_sfp_frozen_encoder.py # Encoder transfer from Diffusion Policy
run_policy.py               # Real robot inference
HKU1.stl                    # 3D-printed robot mount
```

## Citation

```bibtex
@inproceedings{shi2025sfp,
    title={Streaming Flow Policy for Robot Manipulation},
    author={Shi, Hao and others},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2025}
}
```

## Acknowledgments

Built on [HuggingFace LeRobot](https://github.com/huggingface/lerobot). The Diffusion Policy encoder transfer (v8) uses weights from the original [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) implementation.
