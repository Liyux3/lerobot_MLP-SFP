from typing import Any

import torch

from lerobot.policies.sfp.configuration_sfp import SFPConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import (
    batch_to_transition,
    transition_to_batch,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def _sfp_batch_to_transition(batch: dict[str, Any]):
    """Convert batch to transition, preserving episode_index and frame_index."""
    transition = batch_to_transition(batch)
    
    # Preserve episode_index and frame_index in complementary_data
    if transition.get(TransitionKey.COMPLEMENTARY_DATA) is None:
        transition[TransitionKey.COMPLEMENTARY_DATA] = {}
    
    if "episode_index" in batch:
        transition[TransitionKey.COMPLEMENTARY_DATA]["episode_index"] = batch["episode_index"]
    if "frame_index" in batch:
        transition[TransitionKey.COMPLEMENTARY_DATA]["frame_index"] = batch["frame_index"]
    
    return transition


def _sfp_transition_to_batch(transition) -> dict[str, Any]:
    """Convert transition to batch, restoring episode_index and frame_index."""
    batch = transition_to_batch(transition)
    
    # Restore episode_index and frame_index from complementary_data
    complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if "episode_index" in complementary:
        batch["episode_index"] = complementary["episode_index"]
    if "frame_index" in complementary:
        batch["frame_index"] = complementary["frame_index"]
    
    return batch


def make_sfp_pre_post_processors(
    config: SFPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for SFP policy.
    
    Similar to diffusion processor, but preserves episode_index and frame_index
    for smooth velocity computation.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=_sfp_batch_to_transition,
            to_output=_sfp_transition_to_batch,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
