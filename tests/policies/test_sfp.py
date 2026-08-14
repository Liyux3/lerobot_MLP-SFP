import math

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.sfp.configuration_sfp import SFPConfig
from lerobot.policies.sfp.modeling_sfp import (
    SFPModel,
    SFPPolicy,
    _future_action_trajectory,
    _integrate_velocity_field,
    _interpolate_trajectory,
    _sample_cfm_inputs_and_targets,
)
from lerobot.policies.sfp.processor_sfp import _sfp_batch_to_transition, _sfp_transition_to_batch
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


class ConstantVelocity(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.times = []

    def forward(self, sample, timestep, global_cond):
        del global_cond
        self.times.append(timestep.detach().clone())
        return torch.full_like(sample, self.value)


def test_sfp_defaults_match_official_pusht_notebook():
    config = SFPConfig()

    assert config.k == 10.0
    assert config.sigma_0 == 0.4
    assert config.integration_steps == 1


def test_sfp_factory_registration():
    config = make_policy_config("sfp", device="cpu")

    assert isinstance(config, SFPConfig)
    assert get_policy_class("sfp") is SFPPolicy


def test_sfp_processor_preserves_velocity_cache_indices():
    batch = {
        OBS_STATE: torch.randn(2, 3),
        "episode_index": torch.tensor([4, 7]),
        "frame_index": torch.tensor([11, 13]),
    }

    restored = _sfp_transition_to_batch(_sfp_batch_to_transition(batch))

    torch.testing.assert_close(restored["episode_index"], batch["episode_index"])
    torch.testing.assert_close(restored["frame_index"], batch["frame_index"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_obs_steps": 0},
        {"n_obs_steps": 10, "horizon": 10},
        {"n_obs_steps": 2, "horizon": 16, "n_action_steps": 16},
        {"integration_steps": 0},
        {"k": 0.0},
        {"sigma_0": -0.1},
    ],
)
def test_sfp_rejects_invalid_temporal_and_flow_parameters(kwargs):
    with pytest.raises(ValueError):
        SFPConfig(**kwargs)


def test_future_action_trajectory_drops_history_aligned_actions():
    trajectory = torch.arange(2 * 6 * 1, dtype=torch.float32).reshape(2, 6, 1)

    future = _future_action_trajectory(trajectory, n_obs_steps=3)

    torch.testing.assert_close(future, trajectory[:, 2:])


def test_interpolation_uses_normalized_time_derivative():
    trajectory = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [3.0, 6.0]]])
    t = torch.tensor([0.25])

    xi_t, dxi_dt = _interpolate_trajectory(trajectory, t)

    torch.testing.assert_close(xi_t, torch.tensor([[0.5, 1.0]]))
    torch.testing.assert_close(dxi_dt, torch.tensor([[2.0, 4.0]]))


def test_cfm_target_matches_stabilized_flow_equations():
    xi_t = torch.tensor([[0.5, 1.0]])
    dxi_dt = torch.tensor([[2.0, 4.0]])
    t = torch.tensor([0.25])
    noise = torch.ones_like(xi_t)

    action_t, target = _sample_cfm_inputs_and_targets(xi_t, dxi_dt, t, k=2.0, sigma_0=0.4, noise=noise)

    error = 0.4 * math.exp(-0.5)
    torch.testing.assert_close(action_t, xi_t + error)
    torch.testing.assert_close(target, dxi_dt - 2.0 * error)


def test_streaming_integrator_returns_post_chunk_continuation():
    velocity_net = ConstantVelocity(2.0)
    start = torch.zeros(1, 1, 2)

    actions, continuation = _integrate_velocity_field(
        velocity_net=velocity_net,
        global_cond=torch.zeros(1, 3),
        start_action=start,
        action_steps=3,
        trajectory_steps=5,
        integration_steps=2,
    )

    expected_actions = torch.tensor([[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]])
    torch.testing.assert_close(actions, expected_actions)
    torch.testing.assert_close(continuation, torch.tensor([[[1.5, 1.5]]]))
    assert len(velocity_net.times) == 6
    torch.testing.assert_close(
        torch.stack(velocity_net.times).flatten(),
        torch.tensor([0.0, 0.125, 0.25, 0.375, 0.5, 0.625]),
    )


def test_sfp_model_training_and_streaming_paths_are_compatible():
    config = SFPConfig(
        n_obs_steps=2,
        n_action_steps=3,
        horizon=5,
        velocity_net_type="pure_mlp",
        time_embed_dim=8,
        hidden_dim=16,
        n_layers=2,
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    model = SFPModel(config)
    batch = {
        OBS_STATE: torch.randn(2, 2, 3),
        OBS_ENV_STATE: torch.randn(2, 2, 4),
        ACTION: torch.randn(2, 5, 2),
    }

    loss = model.compute_loss(batch)
    loss.backward()
    actions, continuation = model.generate_action_chunk(batch)
    next_actions, _ = model.generate_action_chunk(batch, last_action=continuation)

    assert loss.isfinite()
    assert actions.shape == (2, 3, 2)
    assert continuation.shape == (2, 1, 2)
    torch.testing.assert_close(next_actions[:, 0], continuation[:, 0])
    assert all(parameter.grad is not None for parameter in model.velocity_net.parameters())
