"""Streaming Flow Policy implementation for LeRobot.

Based on "Streaming Flow Policy for Robot Manipulation" (CoRL 2025).
Uses Euler integration of a learned velocity field instead of diffusion denoising.
"""

from collections import deque

import einops
import torch
from torch import Tensor, nn
from torch.nn import functional

from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sfp.configuration_sfp import SFPConfig
from lerobot.policies.sfp.velocity_nets import make_velocity_net
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


def _future_action_trajectory(trajectory: Tensor, n_obs_steps: int) -> Tensor:
    """Drop actions aligned with historical observations.

    LeRobot samples ``horizon`` actions starting at ``1 - n_obs_steps``. SFP
    integrates from the current timestep, so its training curve starts at index
    ``n_obs_steps - 1`` of that observation-aligned window.
    """
    start = n_obs_steps - 1
    if trajectory.ndim != 3:
        raise ValueError(f"Expected trajectory with shape (B, T, A), got {trajectory.shape}")
    if start >= trajectory.shape[1] - 1:
        raise ValueError(
            "SFP needs at least two current/future actions after alignment, got "
            f"trajectory length {trajectory.shape[1]} and {n_obs_steps=}"
        )
    return trajectory[:, start:]


def _interpolate_trajectory(trajectory: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Linearly interpolate an action curve on normalized time ``[0, 1]``."""
    batch_size, horizon, _ = trajectory.shape
    if horizon < 2:
        raise ValueError("SFP trajectory needs at least two actions")

    scaled_t = t.clamp(0.0, 1.0 - torch.finfo(t.dtype).eps) * (horizon - 1)
    lower = scaled_t.floor().long().clamp(max=horizon - 2)
    upper = lower + 1
    weight = (scaled_t - lower).unsqueeze(-1)
    batch_idx = torch.arange(batch_size, device=trajectory.device)
    xi_lower = trajectory[batch_idx, lower]
    xi_upper = trajectory[batch_idx, upper]
    xi_t = xi_lower + weight * (xi_upper - xi_lower)
    dxi_dt = (xi_upper - xi_lower) * (horizon - 1)
    return xi_t, dxi_dt


def _sample_cfm_inputs_and_targets(
    xi_t: Tensor,
    dxi_dt: Tensor,
    t: Tensor,
    k: float,
    sigma_0: float,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample the stabilized CFM input and target from SFP equations 2 and 3."""
    if noise is None:
        noise = torch.randn_like(xi_t)
    sampled_error = sigma_0 * torch.exp(-k * t).unsqueeze(-1) * noise
    action_t = xi_t + sampled_error
    velocity_target = dxi_dt - k * sampled_error
    return action_t, velocity_target


def _integrate_velocity_field(
    velocity_net: nn.Module,
    global_cond: Tensor,
    start_action: Tensor,
    action_steps: int,
    trajectory_steps: int,
    integration_steps: int,
) -> tuple[Tensor, Tensor]:
    """Euler-integrate a streamed action chunk and return its continuation state."""
    if trajectory_steps < 2:
        raise ValueError(f"trajectory_steps must be at least 2, got {trajectory_steps}")
    if not 1 <= action_steps <= trajectory_steps - 1:
        raise ValueError(
            "action_steps must fit in the trajectory integration intervals, got "
            f"{action_steps=} and {trajectory_steps=}"
        )
    if integration_steps < 1:
        raise ValueError(f"integration_steps must be positive, got {integration_steps}")

    action = start_action
    actions = []
    interval = 1.0 / (trajectory_steps - 1)
    dt = interval / integration_steps
    batch_size = action.shape[0]

    for action_idx in range(action_steps):
        actions.append(action.squeeze(1))
        for substep_idx in range(integration_steps):
            scalar_t = (action_idx + substep_idx / integration_steps) * interval
            t = torch.full((batch_size,), scalar_t, device=action.device, dtype=action.dtype)
            velocity = velocity_net(sample=action, timestep=t, global_cond=global_cond)
            action = action + velocity * dt

    return torch.stack(actions, dim=1), action


class SFPPolicy(PreTrainedPolicy):
    """Streaming Flow Policy for robot manipulation."""

    config_class = SFPConfig
    name = "sfp"

    def __init__(self, config: SFPConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.sfp_model = SFPModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.sfp_model.parameters()

    def reset(self):
        """Clear observation and action queues. Called on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

        self._last_action = None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions, self._last_action = self.sfp_model.generate_action_chunk(
            batch, last_action=self._last_action
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations."""
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Compute training loss."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        loss = self.sfp_model.compute_loss(batch)
        return loss, None


class SFPModel(nn.Module):
    """Core SFP model with vision encoder and velocity network."""

    def __init__(self, config: SFPConfig):
        super().__init__()
        self.config = config

        global_cond_dim = config.robot_state_feature.shape[0]

        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [self._make_rgb_encoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = self._make_rgb_encoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images

        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        total_cond_dim = global_cond_dim * config.n_obs_steps

        action_dim = config.action_feature.shape[0]
        self.velocity_net = make_velocity_net(
            velocity_net_type=config.velocity_net_type,
            action_dim=action_dim,
            obs_dim=total_cond_dim,
            time_embed_dim=config.time_embed_dim,
            expand_dim=config.expand_dim,
            bottleneck_dim=config.bottleneck_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            sin_embedding_scale=config.sin_embedding_scale,
        )

        # Smooth velocity cache (initialized by precompute_smooth_velocities)
        self._velocity_cache = None
        self._episode_lengths = None

    def _make_rgb_encoder(self, config: SFPConfig) -> DiffusionRgbEncoder:
        """Create RGB encoder, reusing diffusion's implementation."""
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

        temp_config = DiffusionConfig(
            input_features=config.input_features,
            output_features=config.output_features,
            vision_backbone=config.vision_backbone,
            crop_shape=config.crop_shape,
            crop_is_random=config.crop_is_random,
            pretrained_backbone_weights=config.pretrained_backbone_weights,
            use_group_norm=config.use_group_norm,
            spatial_softmax_num_keypoints=config.spatial_softmax_num_keypoints,
        )
        return DiffusionRgbEncoder(temp_config)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations into conditioning vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(
        self,
        batch: dict[str, Tensor],
        last_action: Tensor | None = None,
    ) -> Tensor:
        """Generate an action chunk while keeping the legacy tensor-only API."""
        actions, _ = self.generate_action_chunk(batch, last_action=last_action)
        return actions

    def generate_action_chunk(
        self,
        batch: dict[str, Tensor],
        last_action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate a streamed action chunk and its next-chunk continuation."""

        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        action_dim = self.config.action_feature.shape[0]

        global_cond = self._prepare_global_conditioning(batch)

        if last_action is None:
            current_state = batch[OBS_STATE][:, -1, :action_dim]
            start_action = current_state.unsqueeze(1).to(device=device, dtype=dtype)
        else:
            start_action = last_action.to(device=device, dtype=dtype)

        trajectory_steps = self.config.horizon - self.config.n_obs_steps + 1
        return _integrate_velocity_field(
            velocity_net=self.velocity_net,
            global_cond=global_cond,
            start_action=start_action,
            action_steps=self.config.n_action_steps,
            trajectory_steps=trajectory_steps,
            integration_steps=self.config.integration_steps,
        )

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute CFM training loss."""
        device = get_device_from_parameters(self)

        global_cond = self._prepare_global_conditioning(batch)

        trajectory = _future_action_trajectory(batch[ACTION], self.config.n_obs_steps)
        batch_size = trajectory.shape[0]

        t = torch.rand(batch_size, device=device)

        # Use smooth velocity if enabled and cache is ready
        if self.config.use_smooth_velocity and self._velocity_cache is not None:
            episode_indices = batch["episode_index"]
            frame_indices = batch["frame_index"]
            smooth_vel = self._get_smooth_velocity_chunk(
                episode_indices, frame_indices, chunk_size=trajectory.shape[1]
            )
            smooth_vel = smooth_vel.to(device=device, dtype=trajectory.dtype)
            xi_t, dxi_dt = self._interpolate_trajectory_smooth(trajectory, t, smooth_vel)
        else:
            xi_t, dxi_dt = self._interpolate_trajectory(trajectory, t)

        a_t, v_target = _sample_cfm_inputs_and_targets(
            xi_t=xi_t,
            dxi_dt=dxi_dt,
            t=t,
            k=self.config.k,
            sigma_0=self.config.sigma_0,
        )

        v_pred = self.velocity_net(
            sample=a_t.unsqueeze(1),
            timestep=t,
            global_cond=global_cond,
        ).squeeze(1)

        return functional.mse_loss(v_pred, v_target)

    def _interpolate_trajectory(self, trajectory: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Linear interpolation of trajectory at time t."""
        return _interpolate_trajectory(trajectory, t)

    def precompute_smooth_velocities(self, dataset) -> None:
        """Precompute Savitzky-Golay smoothed velocities for all episodes.

        Call this once before training starts.
        """
        from scipy.signal import savgol_filter

        dataset._ensure_hf_dataset_loaded()
        episodes = dataset.meta.episodes
        num_episodes = len(episodes["episode_index"])

        self._velocity_cache = {}
        self._episode_lengths = {}

        window = self.config.smooth_window_length
        polyorder = self.config.smooth_polyorder

        print(f"Precomputing smooth velocities for {num_episodes} episodes...")

        # Get normalization stats
        if "action" not in dataset.meta.stats:
            raise ValueError("Dataset stats must contain 'action'")

        action_min = torch.tensor(dataset.meta.stats["action"]["min"], dtype=torch.float32)
        action_max = torch.tensor(dataset.meta.stats["action"]["max"], dtype=torch.float32)

        print(f"  Action normalization: min={action_min.numpy()}, max={action_max.numpy()}")

        for ep_idx in range(num_episodes):
            from_idx = episodes["dataset_from_index"][ep_idx]
            to_idx = episodes["dataset_to_index"][ep_idx]

            # Batch read all actions in this episode
            subset = dataset.hf_dataset.select(range(from_idx, to_idx))
            actions = torch.stack(list(subset["action"]))  # (T_ep, action_dim)

            # Normalize to [-1, 1]
            actions_normalized = (actions - action_min) / (action_max - action_min) * 2.0 - 1.0

            if ep_idx == 0:
                print(
                    f"  Episode 0 normalized range: [{actions_normalized.min():.3f}, {actions_normalized.max():.3f}]"
                )

            ep_len = actions.shape[0]
            self._episode_lengths[ep_idx] = ep_len

            # Handle short episodes
            actual_window = min(window, ep_len)
            if actual_window % 2 == 0:
                actual_window -= 1
            if actual_window < 3:
                # Too short for savgol, use simple diff
                smooth_vel = torch.zeros_like(actions)
                smooth_vel[:-1] = actions[1:] - actions[:-1]
                smooth_vel[-1] = smooth_vel[-2]
            else:
                actual_polyorder = min(polyorder, actual_window - 1)
                smooth_vel = savgol_filter(
                    actions_normalized.numpy(),  # Use normalized actions
                    window_length=actual_window,
                    polyorder=actual_polyorder,
                    deriv=1,
                    axis=0,
                )
                #                 print(f"Savgol raw range: [{smooth_vel.min():.4f}, {smooth_vel.max():.4f}]")
                #                 print(f"After ×15 range: [{(smooth_vel * 14).min():.4f}, {(smooth_vel * 14).max():.4f}]")
                smooth_vel = torch.from_numpy(smooth_vel).float()

            self._velocity_cache[ep_idx] = smooth_vel

        print(
            f"Smooth velocity cache ready. Memory: {sum(v.numel() * 4 for v in self._velocity_cache.values()) / 1024 / 1024:.2f} MB"
        )

    def _get_smooth_velocity_chunk(
        self, episode_indices: Tensor, frame_indices: Tensor, chunk_size: int = None
    ) -> Tensor:
        """Get smoothed velocity chunks from cache.

        Args:
            episode_indices: (B,) episode index for each sample
            frame_indices: (B,) frame index within episode for each sample
            chunk_size: number of frames per chunk (default 16)

        Returns:
            (B, chunk_size, action_dim) smoothed velocities
        """
        if chunk_size is None:
            chunk_size = self.config.horizon

        batch_size = episode_indices.shape[0]
        action_dim = self._velocity_cache[0].shape[-1]
        result = torch.zeros(batch_size, chunk_size, action_dim)

        for i in range(batch_size):
            ep_idx = episode_indices[i].item()
            frame_idx = frame_indices[i].item()
            ep_len = self._episode_lengths[ep_idx]

            indices = []
            for delta in range(chunk_size):
                idx = frame_idx + delta
                idx = max(0, min(ep_len - 1, idx))  # clamp to valid range
                indices.append(idx)

            result[i] = self._velocity_cache[ep_idx][indices]

        return result

    def _interpolate_trajectory_smooth(
        self, trajectory: Tensor, t: Tensor, smooth_vel: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Interpolate trajectory using precomputed smooth velocities.

        Args:
            trajectory: (B, T, A) action trajectory
            t: (B,) normalized time in [0, 1]
            smooth_vel: (B, T, A) precomputed smooth velocities

        Returns:
            xi_t: (B, A) interpolated position
            dxi_dt: (B, A) smoothed velocity at time t
        """
        batch_size, horizon, _ = trajectory.shape

        scaled_t = t * (horizon - 1)
        lower = scaled_t.floor().long().clamp(0, horizon - 2)
        upper = (lower + 1).clamp(0, horizon - 1)
        weight = (scaled_t - lower.float()).unsqueeze(-1)

        batch_idx = torch.arange(batch_size, device=trajectory.device)

        # Position: interpolate from original trajectory
        xi_lower = trajectory[batch_idx, lower]
        xi_upper = trajectory[batch_idx, upper]
        xi_t = xi_lower + weight * (xi_upper - xi_lower)

        # Velocity: interpolate from smooth_vel, then scale
        # smooth_vel is per-frame change, multiply by (T-1) for normalized time
        velocity_lower = smooth_vel[batch_idx, lower]
        velocity_upper = smooth_vel[batch_idx, upper]
        dxi_dt = (velocity_lower + weight * (velocity_upper - velocity_lower)) * (horizon - 1)

        # if torch.rand(1).item() < 0.001:
        #      print(f"smooth dxi_dt range: [{dxi_dt.min().item():.3f}, {dxi_dt.max().item():.3f}]")

        return xi_t, dxi_dt
