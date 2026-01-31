"""Train SFP with frozen DP encoder."""

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.sfp import SFPConfig, SFPPolicy
from lerobot.configs.types import PolicyFeature, FeatureType

def create_sfp_with_dp_encoder():
    """Create SFP model with frozen DP encoder."""
    
    # Load DP
    dp = DiffusionPolicy.from_pretrained("./outputs/pusht_diffusion_v5/checkpoints/050000/pretrained_model")
    
    # Create SFP config matching DP's encoder settings
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    
    config = SFPConfig(
        input_features=input_features,
        output_features=output_features,
        use_group_norm=True,
        pretrained_backbone_weights=None,  # We'll copy from DP
        velocity_net_type="bottleneck_skip_auto",
        device="cuda",
    )
    
    # Create SFP
    sfp = SFPPolicy(config)
    
    # Copy DP encoder weights
    sfp.sfp_model.rgb_encoder.load_state_dict(dp.diffusion.rgb_encoder.state_dict())
    print("Copied DP encoder weights")
    
    # Freeze encoder
    for param in sfp.sfp_model.rgb_encoder.parameters():
        param.requires_grad = False
    print("Froze encoder parameters")
    
    # Verify
    encoder_params = sum(p.numel() for p in sfp.sfp_model.rgb_encoder.parameters())
    trainable_params = sum(p.numel() for p in sfp.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in sfp.parameters())
    
    print(f"\nParameter counts:")
    print(f"  Encoder (frozen): {encoder_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Total: {total_params:,}")
    
    return sfp

if __name__ == "__main__":
    sfp = create_sfp_with_dp_encoder()
    sfp.save_pretrained("./outputs/pusht_sfp_frozen_enc_init")
    print("\nSaved initial model to ./outputs/pusht_sfp_frozen_enc_init")
