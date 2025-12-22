"""Vision Token Pruning Method

对抗训练的视觉token剪枝方法。
"""

from .models.generator import Generator
from .models.discriminator import Discriminator
from .models.token_merger import LearnableTokenMerger, LearnableTokenMergerV2, LearnableTokenMergerV3
from .models.layer_pruner import LayerSpecificPruner, VisionPrunerHead, VisionPrunerHeadSimple
from .training import train_step
from .training_batch import train_step_batch
from .evaluation import eval_step
from .utils import (
    get_vision_features_and_question_embedding,
    apply_mask_to_embeddings,
    batch_forward_with_masks,
    extract_target_hidden_states,
    compute_task_loss,
    get_current_sparsity_weight,
    update_generator_temperature,
    get_target_token_num,
    # Multi-layer hook utilities
    create_layer_pruning_modifier,
    register_multi_layer_hooks,
    remove_hooks,
    replace_vision_tokens_in_embeddings,
    update_temperature_for_all
)

__all__ = [
    # Models
    'Generator',
    'Discriminator',
    'LearnableTokenMerger',
    'LearnableTokenMergerV2',
    'LearnableTokenMergerV3',
    'LayerSpecificPruner',
    'VisionPrunerHead',
    'VisionPrunerHeadSimple',
    # Training & Evaluation
    'train_step',
    'train_step_batch',
    'eval_step',
    # Utils
    'get_vision_features_and_question_embedding',
    'apply_mask_to_embeddings',
    'batch_forward_with_masks',
    'extract_target_hidden_states',
    'compute_task_loss',
    'get_current_sparsity_weight',
    'update_generator_temperature',
    'get_target_token_num',
    # Multi-layer hook utilities
    'create_layer_pruning_modifier',
    'register_multi_layer_hooks',
    'remove_hooks',
    'replace_vision_tokens_in_embeddings',
    'update_temperature_for_all',
]
