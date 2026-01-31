"""Velocity network architectures for Streaming Flow Policy.

All networks follow the same interface:
    forward(sample, timestep, global_cond) -> velocity
"""

import math
from typing import Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time conditioning."""
    
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
    
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.scale
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def _auto_scale_dims(input_dim: int, expand_ratio: float = 2.0, bottleneck_ratio: float = 0.5):
    """Compute auto-scaled dimensions based on input size."""
    expand_dim = int((input_dim * expand_ratio) // 64) * 64
    expand_dim = max(expand_dim, 256)
    bottleneck_dim = int((input_dim * bottleneck_ratio) // 32) * 32
    bottleneck_dim = max(bottleneck_dim, 64)
    return expand_dim, bottleneck_dim


# =============================================================================
# MLP Architectures
# =============================================================================

class BottleneckSkipMLP(nn.Module):
    """MLP with bottleneck + skip connection. Fixed dims for v1 checkpoint compat."""
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        time_embed_dim: int = 256,
        expand_dim: int = 512,
        bottleneck_dim: int = 128,
        sin_embedding_scale: float = 100.0,
    ):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim, scale=sin_embedding_scale),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        input_dim = action_dim + obs_dim + time_embed_dim
        
        self.expand = nn.Sequential(
            nn.Linear(input_dim, expand_dim),
            nn.ReLU(),
            nn.Linear(expand_dim, expand_dim),
            nn.ReLU(),
        )
        
        self.compress = nn.Linear(expand_dim, bottleneck_dim)
        
        self.process = nn.Sequential(
            nn.Linear(bottleneck_dim + expand_dim, expand_dim),
            nn.ReLU(),
            nn.Linear(expand_dim, expand_dim),
            nn.ReLU(),
        )
        
        self.output_layer = nn.Linear(expand_dim, action_dim)
        
        print(f"BottleneckSkipMLP: input={input_dim}, expand={expand_dim}, bottleneck={bottleneck_dim}, params={sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        has_seq_dim = (sample.dim() == 3)
        if has_seq_dim:
            sample = sample.squeeze(1)
        
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(sample.shape[0])
        t_emb = self.time_embed(timestep)
        
        x = torch.cat([sample, global_cond, t_emb], dim=-1)
        
        expanded = self.expand(x)
        bottleneck = F.relu(self.compress(expanded))
        combined = torch.cat([bottleneck, expanded], dim=-1)
        processed = self.process(combined)
        out = self.output_layer(processed)
        
        if has_seq_dim:
            out = out.unsqueeze(1)
        
        return out


class BottleneckSkipMLPAuto(nn.Module):
    """MLP with bottleneck + skip connection. Auto-scaled dims."""
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        time_embed_dim: int = 256,
        sin_embedding_scale: float = 100.0,
    ):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim, scale=sin_embedding_scale),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        input_dim = action_dim + obs_dim + time_embed_dim
        expand_dim, bottleneck_dim = _auto_scale_dims(input_dim)
        
        self.expand = nn.Sequential(
            nn.Linear(input_dim, expand_dim),
            nn.ReLU(),
            nn.Linear(expand_dim, expand_dim),
            nn.ReLU(),
        )
        
        self.compress = nn.Linear(expand_dim, bottleneck_dim)
        
        self.process = nn.Sequential(
            nn.Linear(bottleneck_dim + expand_dim, expand_dim),
            nn.ReLU(),
            nn.Linear(expand_dim, expand_dim),
            nn.ReLU(),
        )
        
        self.output_layer = nn.Linear(expand_dim, action_dim)
        
        print(f"BottleneckSkipMLPAuto: input={input_dim}, expand={expand_dim}, bottleneck={bottleneck_dim}, params={sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        has_seq_dim = (sample.dim() == 3)
        if has_seq_dim:
            sample = sample.squeeze(1)
        
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(sample.shape[0])
        t_emb = self.time_embed(timestep)
        
        x = torch.cat([sample, global_cond, t_emb], dim=-1)
        
        expanded = self.expand(x)
        bottleneck = F.relu(self.compress(expanded))
        combined = torch.cat([bottleneck, expanded], dim=-1)
        processed = self.process(combined)
        out = self.output_layer(processed)
        
        if has_seq_dim:
            out = out.unsqueeze(1)
        
        return out


class PureMLP(nn.Module):
    """Simple sequential MLP. Auto-scaled."""
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        time_embed_dim: int = 256,
        n_layers: int = 4,
        sin_embedding_scale: float = 100.0,
    ):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim, scale=sin_embedding_scale),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        input_dim = action_dim + obs_dim + time_embed_dim
        hidden_dim, _ = _auto_scale_dims(input_dim)
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
        
        print(f"PureMLP: input={input_dim}, hidden={hidden_dim}, layers={n_layers}, params={sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        has_seq_dim = (sample.dim() == 3)
        if has_seq_dim:
            sample = sample.squeeze(1)
        
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(sample.shape[0])
        t_emb = self.time_embed(timestep)
        
        x = torch.cat([sample, global_cond, t_emb], dim=-1)
        out = self.net(x)
        
        if has_seq_dim:
            out = out.unsqueeze(1)
        
        return out


class CondEveryLayerMLP(nn.Module):
    """MLP with obs+time conditioning concatenated at every layer input."""
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        time_embed_dim: int = 256,
        n_layers: int = 4,
        sin_embedding_scale: float = 100.0,
    ):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim, scale=sin_embedding_scale),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        cond_dim = obs_dim + time_embed_dim
        input_dim = action_dim + cond_dim
        hidden_dim, _ = _auto_scale_dims(input_dim)
        
        self.cond_dim = cond_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim + cond_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        print(f"CondEveryLayerMLP: input={input_dim}, hidden={hidden_dim}, layers={n_layers}, params={sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        has_seq_dim = (sample.dim() == 3)
        if has_seq_dim:
            sample = sample.squeeze(1)
        
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(sample.shape[0])
        t_emb = self.time_embed(timestep)
        
        cond = torch.cat([global_cond, t_emb], dim=-1)
        
        x = torch.cat([sample, cond], dim=-1)
        x = F.relu(self.input_proj(x))
        
        for layer in self.layers:
            x = torch.cat([x, cond], dim=-1)
            x = F.relu(layer(x))
        
        out = self.output_layer(x)
        
        if has_seq_dim:
            out = out.unsqueeze(1)
        
        return out


class CondResidualMLP(nn.Module):
    """MLP with conditioning at every layer + residual connections."""
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        time_embed_dim: int = 256,
        n_layers: int = 4,
        sin_embedding_scale: float = 100.0,
    ):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim, scale=sin_embedding_scale),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        cond_dim = obs_dim + time_embed_dim
        input_dim = action_dim + cond_dim
        hidden_dim, _ = _auto_scale_dims(input_dim)
        
        self.cond_dim = cond_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim + cond_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        print(f"CondResidualMLP: input={input_dim}, hidden={hidden_dim}, layers={n_layers}, params={sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        has_seq_dim = (sample.dim() == 3)
        if has_seq_dim:
            sample = sample.squeeze(1)
        
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(sample.shape[0])
        t_emb = self.time_embed(timestep)
        
        cond = torch.cat([global_cond, t_emb], dim=-1)
        
        x = torch.cat([sample, cond], dim=-1)
        x = F.relu(self.input_proj(x))
        
        for layer in self.layers:
            x_in = torch.cat([x, cond], dim=-1)
            x = x + F.relu(layer(x_in))  # residual
        
        out = self.output_layer(x)
        
        if has_seq_dim:
            out = out.unsqueeze(1)
        
        return out


# =============================================================================
# UNet Architecture (from SFP notebook)
# =============================================================================

class LinearDownsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor):
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = x.view(batch_size, channels, seq_len)
        return x


class LinearUpsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor):
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = x.view(batch_size, channels, seq_len)
        return x


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """UNet for SFP, with Linear up/downsampling for T=1."""
    
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        sin_embedding_scale: float = 100.0,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed, scale=sin_embedding_scale),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            downsample_layer = LinearDownsample1d(dim_out) if not is_last else nn.Identity()
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                downsample_layer,
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            upsample_layer = LinearUpsample1d(dim_in) if not is_last else nn.Identity()
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                upsample_layer,
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        print(f"ConditionalUnet1D: input_dim={input_dim}, global_cond_dim={global_cond_dim}, params={sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        sample: Tensor,
        timestep: Union[Tensor, float, int],
        global_cond: Tensor = None,
    ) -> Tensor:
        """
        sample: (B, T, input_dim) or (B, 1, input_dim)
        timestep: (B,) or scalar, float in [0, 1]
        global_cond: (B, global_cond_dim)
        output: (B, T, input_dim)
        """
        # Handle (B, input_dim) case
        has_seq_dim = (sample.dim() == 3)
        if not has_seq_dim:
            sample = sample.unsqueeze(1)
        
        # (B, T, C) -> (B, C, T)
        sample = sample.moveaxis(-1, -2)

        # Time embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B, C, T) -> (B, T, C)
        x = x.moveaxis(-1, -2)
        
        if not has_seq_dim:
            x = x.squeeze(1)
        
        return x


# =============================================================================
# Factory
# =============================================================================

def make_velocity_net(
    velocity_net_type: str,
    action_dim: int,
    obs_dim: int,
    time_embed_dim: int = 256,
    expand_dim: int = 512,
    bottleneck_dim: int = 128,
    hidden_dim: int = 512,
    n_layers: int = 4,
    sin_embedding_scale: float = 100.0,
    down_dims: list = None,
) -> nn.Module:
    """Factory function to create velocity network by type.
    
    Args:
        velocity_net_type: One of "bottleneck_skip", "bottleneck_skip_auto", 
            "pure_mlp", "cond_every_layer", "cond_residual", "unet", "unet_dp"
        down_dims: For unet types, channel dims at each level.
            unet default: [256, 512, 1024] (~65M params)
            unet_dp default: [512, 1024, 2048] (~250M params, matches DP)
    """
    
    if velocity_net_type == "bottleneck_skip":
        return BottleneckSkipMLP(
            action_dim=action_dim,
            obs_dim=obs_dim,
            time_embed_dim=time_embed_dim,
            expand_dim=expand_dim,
            bottleneck_dim=bottleneck_dim,
            sin_embedding_scale=sin_embedding_scale,
        )
    elif velocity_net_type == "bottleneck_skip_auto":
        return BottleneckSkipMLPAuto(
            action_dim=action_dim,
            obs_dim=obs_dim,
            time_embed_dim=time_embed_dim,
            sin_embedding_scale=sin_embedding_scale,
        )
    elif velocity_net_type == "pure_mlp":
        return PureMLP(
            action_dim=action_dim,
            obs_dim=obs_dim,
            time_embed_dim=time_embed_dim,
            n_layers=n_layers,
            sin_embedding_scale=sin_embedding_scale,
        )
    elif velocity_net_type == "cond_every_layer":
        return CondEveryLayerMLP(
            action_dim=action_dim,
            obs_dim=obs_dim,
            time_embed_dim=time_embed_dim,
            n_layers=n_layers,
            sin_embedding_scale=sin_embedding_scale,
        )
    elif velocity_net_type == "cond_residual":
        return CondResidualMLP(
            action_dim=action_dim,
            obs_dim=obs_dim,
            time_embed_dim=time_embed_dim,
            n_layers=n_layers,
            sin_embedding_scale=sin_embedding_scale,
        )
    elif velocity_net_type == "unet":
        # SFP notebook dims (~65M params)
        if down_dims is None:
            down_dims = [256, 512, 1024]
        return ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            sin_embedding_scale=sin_embedding_scale,
            down_dims=down_dims,
        )
    elif velocity_net_type == "unet_dp":
        # DP dims (~250M params)
        if down_dims is None:
            down_dims = [512, 1024, 2048]
        return ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            sin_embedding_scale=sin_embedding_scale,
            down_dims=down_dims,
        )
    else:
        raise ValueError(f"Unknown velocity_net_type: {velocity_net_type}")
