"""Streaming Flow Policy implementation for LeRobot.

Based on "Streaming Flow Policy for Robot Manipulation" (CoRL 2025).
Uses Euler integration of a learned velocity field instead of diffusion denoising.
"""
import time
from collections import deque

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

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
        actions = self.sfp_model.generate_actions(batch, last_action=self._last_action)
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
            self._last_action = actions[:, -1:, :]
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
                img_features_list = torch.cat([
                    encoder(images)
                    for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ])
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
        """Generate action chunk via Euler integration."""
        # TODO: make it streaming actions instead of returning a list of actions

        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        
        batch_size = batch[OBS_STATE].shape[0]
        action_dim = self.config.action_feature.shape[0]
        
        global_cond = self._prepare_global_conditioning(batch)
        
        # if last_action is not None:
        #     a = last_action.to(device=device, dtype=dtype)
        # else:
            # Initialize from current robot state (like SFP notebook)
            # batch[OBS_STATE] is (B, n_obs_steps, state_dim), take latest
        current_state = batch[OBS_STATE][:, -1, :action_dim]  # (B, action_dim)
        a = current_state.unsqueeze(1).to(device=device, dtype=dtype)  # (B, 1, action_dim)
        
        actions = []
        # SFP notebook: dt = 1 / (pred_horizon - obs_horizon)
        # For horizon=16, n_obs_steps=2: dt = 1/14
        # t ranges from 0 to 0.5 for n_action_steps=8
        dt = 1.0 / (self.config.horizon - self.config.n_obs_steps)
        
        for i in range(self.config.n_action_steps):
            actions.append(a.squeeze(1))
            
            t = torch.tensor(i * dt, device=device, dtype=dtype)
            v = self.velocity_net(
                sample=a,
                timestep=t,
                global_cond=global_cond,
            )

            a = a + v * dt
            
        actions = torch.stack(actions, dim=1)
        return actions
    
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute CFM training loss."""
        device = get_device_from_parameters(self)
        
        global_cond = self._prepare_global_conditioning(batch)
        
        trajectory = batch[ACTION]
        # print(f"DEBUG: trajectory range in compute_loss: [{trajectory.min():.4f}, {trajectory.max():.4f}]") 
        batch_size = trajectory.shape[0]
        
        t = torch.rand(batch_size, device=device)
        
        # Use smooth velocity if enabled and cache is ready
        if self.config.use_smooth_velocity and self._velocity_cache is not None:
            episode_indices = batch["episode_index"]
            frame_indices = batch["frame_index"]
            smooth_vel = self._get_smooth_velocity_chunk(episode_indices, frame_indices, chunk_size=trajectory.shape[1])
            smooth_vel = smooth_vel.to(device=device, dtype=trajectory.dtype)
            xi_t, dxi_dt = self._interpolate_trajectory_smooth(trajectory, t, smooth_vel)
        else:
            xi_t, dxi_dt = self._interpolate_trajectory(trajectory, t)
        
        k = self.config.k
        sigma_0 = self.config.sigma_0
        noise = torch.randn_like(xi_t)
        sampled_error = sigma_0 * torch.exp(-k * t.unsqueeze(-1)) * noise
        a_t = xi_t + sampled_error
        
        v_target = -k * sampled_error + dxi_dt


        v_pred = self.velocity_net(
            sample=a_t.unsqueeze(1),
            timestep=t,
            global_cond=global_cond,
        ).squeeze(1)
        
        # loss = F.mse_loss(v_pred, v_target) # original navie loss

        k_base = self.config.k
        # Variable k: ramps up from 0 to k_base following inverse of noise decay
        k_effective = k_base * (1 - torch.exp(-k_base * t))
        k_effective = k_effective.unsqueeze(-1)  # (B, 1)
        
        v_target_full = dxi_dt - k_effective * (a_t - xi_t)
        v_target_ff = dxi_dt
        
        loss_ff = F.mse_loss(v_pred, v_target_ff)
        loss_full = F.mse_loss(v_pred, v_target_full)
        loss = 0.7 * loss_ff + 0.3 * loss_full

        return loss
    
    def _interpolate_trajectory(self, trajectory: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Linear interpolation of trajectory at time t."""
        B, T, A = trajectory.shape
        
        scaled_t = t * (T - 1)
        l = scaled_t.floor().long().clamp(0, T - 2)
        u = (l + 1).clamp(0, T - 1)
        lam = (scaled_t - l.float()).unsqueeze(-1)
        
        batch_idx = torch.arange(B, device=trajectory.device)
        xi_l = trajectory[batch_idx, l]
        xi_u = trajectory[batch_idx, u]
        
        xi_t = xi_l + lam * (xi_u - xi_l)
        dxi_dt = (xi_u - xi_l) * (T - 1)

        return xi_t, dxi_dt
    
    def precompute_smooth_velocities(self, dataset) -> None:
        """Precompute Savitzky-Golay smoothed velocities for all episodes.
        
        Call this once before training starts.
        """
        from scipy.signal import savgol_filter
        
        dataset._ensure_hf_dataset_loaded()
        episodes = dataset.meta.episodes
        num_episodes = len(episodes['episode_index'])
        
        self._velocity_cache = {}
        self._episode_lengths = {}
        
        window = self.config.smooth_window_length
        polyorder = self.config.smooth_polyorder
        
        print(f"Precomputing smooth velocities for {num_episodes} episodes...")


        # Get normalization stats
        if 'action' not in dataset.meta.stats:
            raise ValueError("Dataset stats must contain 'action'")
        
        action_min = torch.tensor(dataset.meta.stats['action']['min'], dtype=torch.float32)
        action_max = torch.tensor(dataset.meta.stats['action']['max'], dtype=torch.float32)
        
        print(f"  Action normalization: min={action_min.numpy()}, max={action_max.numpy()}")

        for ep_idx in range(num_episodes):
            from_idx = episodes['dataset_from_index'][ep_idx]
            to_idx = episodes['dataset_to_index'][ep_idx]
            
            # Batch read all actions in this episode
            subset = dataset.hf_dataset.select(range(from_idx, to_idx))
            actions = torch.stack([a for a in subset['action']])  # (T_ep, action_dim)

            # Normalize to [-1, 1]
            actions_normalized = (actions - action_min) / (action_max - action_min) * 2.0 - 1.0
            
            if ep_idx == 0:
                print(f"  Episode 0 normalized range: [{actions_normalized.min():.3f}, {actions_normalized.max():.3f}]")
            
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
                    actions_normalized.numpy(), # Use normalized actions
                    window_length=actual_window,
                    polyorder=actual_polyorder,
                    deriv=1,
                    axis=0
                )
#                 print(f"Savgol raw range: [{smooth_vel.min():.4f}, {smooth_vel.max():.4f}]")
#                 print(f"After ×15 range: [{(smooth_vel * 14).min():.4f}, {(smooth_vel * 14).max():.4f}]")
                smooth_vel = torch.from_numpy(smooth_vel).float()
            
            self._velocity_cache[ep_idx] = smooth_vel
        
        print(f"Smooth velocity cache ready. Memory: {sum(v.numel() * 4 for v in self._velocity_cache.values()) / 1024 / 1024:.2f} MB")
    
    def _get_smooth_velocity_chunk(
        self, 
        episode_indices: Tensor, 
        frame_indices: Tensor,
        chunk_size: int = None
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

        B = episode_indices.shape[0]
        action_dim = self._velocity_cache[0].shape[-1]
        result = torch.zeros(B, chunk_size, action_dim)
        
        for i in range(B):
            ep_idx = episode_indices[i].item()
            frame_idx = frame_indices[i].item()
            ep_len = self._episode_lengths[ep_idx]
            
            # delta_timestamps = [-1/30, 0, 1/30, ..., 14/30]
            # corresponds to frame_idx + [-1, 0, 1, ..., 14]
            indices = []
            for delta in range(-1, chunk_size - 1):  # -1 to 14, total 16
                idx = frame_idx + delta
                idx = max(0, min(ep_len - 1, idx))  # clamp to valid range
                indices.append(idx)
            
            result[i] = self._velocity_cache[ep_idx][indices]
        
        return result
    
    def _interpolate_trajectory_smooth(
        self, 
        trajectory: Tensor, 
        t: Tensor, 
        smooth_vel: Tensor
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
        B, T, A = trajectory.shape
        
        scaled_t = t * (T - 1)
        l = scaled_t.floor().long().clamp(0, T - 2)
        u = (l + 1).clamp(0, T - 1)
        lam = (scaled_t - l.float()).unsqueeze(-1)
        
        batch_idx = torch.arange(B, device=trajectory.device)
        
        # Position: interpolate from original trajectory
        xi_l = trajectory[batch_idx, l]
        xi_u = trajectory[batch_idx, u]
        xi_t = xi_l + lam * (xi_u - xi_l)
        
        # Velocity: interpolate from smooth_vel, then scale
        # smooth_vel is per-frame change, multiply by (T-1) for normalized time
        vel_l = smooth_vel[batch_idx, l]
        vel_u = smooth_vel[batch_idx, u]
        dxi_dt = (vel_l + lam * (vel_u - vel_l)) * (T - 1)

        # if torch.rand(1).item() < 0.001:
        #      print(f"smooth dxi_dt range: [{dxi_dt.min().item():.3f}, {dxi_dt.max().item():.3f}]")
        
        return xi_t, dxi_dt
