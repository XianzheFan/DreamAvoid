import logging
import os
import pathlib
from typing import Any
import jax.numpy as jnp
import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

def create_trained_policy(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path | str, *, repack_transforms: transforms.Group | None=None, sample_kwargs: dict[str, Any] | None=None, default_prompt: str | None=None, norm_stats: dict[str, transforms.NormStats] | None=None, pytorch_device: str | None=None) -> _policy.Policy:
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    weight_path = os.path.join(checkpoint_dir, 'model.safetensors')
    is_pytorch = os.path.exists(weight_path)
    logging.info('Loading model...')
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / 'params', dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError('Asset id is required to load norm stats.')
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / 'assets', data_config.asset_id)
    if is_pytorch and pytorch_device is None:
        try:
            import torch
            pytorch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            pytorch_device = 'cpu'
    effective_sample_kwargs = dict(sample_kwargs or {})
    if getattr(train_config.model, 'switch_head', False) and 'return_switch' not in effective_sample_kwargs:
        effective_sample_kwargs['return_switch'] = True
    return _policy.Policy(model, transforms=[*repack_transforms.inputs, transforms.InjectDefaultPrompt(default_prompt), *data_config.data_transforms.inputs, transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm), *data_config.model_transforms.inputs], output_transforms=[*data_config.model_transforms.outputs, transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm), *data_config.data_transforms.outputs, *repack_transforms.outputs], sample_kwargs=effective_sample_kwargs, metadata=train_config.policy_metadata, is_pytorch=is_pytorch, pytorch_device=pytorch_device if is_pytorch else None)
