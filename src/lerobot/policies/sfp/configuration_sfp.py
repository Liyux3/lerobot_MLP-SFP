from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("sfp")
@dataclass
class SFPConfig(PreTrainedConfig):
    """Configuration class for Streaming Flow Policy.
    
    Based on "Streaming Flow Policy for Robot Manipulation" (CoRL 2025).
    Uses Euler integration of a learned velocity field instead of diffusion denoising.
    """
    
    # Temporal structure
    n_obs_steps: int = 10
    n_action_steps: int = 20
    horizon: int = 50
    
    # SFP-specific parameters
    k: float = 2.0
    sigma_0: float = 1.0
    integration_steps: int = 4
    sin_embedding_scale: float = 100.0
    
    # Smooth velocity (for filtering hand jitter in demonstrations)
    use_smooth_velocity: bool = False
    smooth_window_length: int = 9 # 9 frames = 0.3s , 21 frames = 0.9s at 30fps
    smooth_polyorder: int = 2
    
    # Velocity network architecture
    velocity_net_type: str = "bottleneck_skip"
    time_embed_dim: int = 256
    expand_dim: int = 512
    bottleneck_dim: int = 128
    hidden_dim: int = 512
    n_layers: int = 4
    
    # Vision backbone
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    
    # Optimizer
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    
    # Scheduler
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    
    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    
    def __post_init__(self):
        super().__post_init__()
        
        valid_types = ("bottleneck_skip", "bottleneck_skip_auto", "pure_mlp", "cond_every_layer", "cond_residual", "unet", "unet_dp")
        if self.velocity_net_type not in valid_types:
            raise ValueError(f"velocity_net_type must be one of {valid_types}, got {self.velocity_net_type}")
    
    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )
    
    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )
    
    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))
    
    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))
    
    @property
    def reward_delta_indices(self) -> None:
        return None
    
    def validate_features(self) -> None:
        """Validate that required features are present."""
        if not self.input_features or not self.output_features:
            return
        
        if not self.robot_state_feature:
            raise ValueError("SFP requires 'observation.state' in input_features")
        
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "SFP requires at least one image feature (observation.image.*) "
                "or environment state (observation.environment_state)"
            )
        
        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )
