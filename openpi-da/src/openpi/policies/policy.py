from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias
import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
BasePolicy: TypeAlias = _base_policy.BasePolicy

class Policy(BasePolicy):

    def __init__(self, model: _model.BaseModel, *, rng: at.KeyArrayLike | None=None, transforms: Sequence[_transforms.DataTransformFn]=(), output_transforms: Sequence[_transforms.DataTransformFn]=(), sample_kwargs: dict[str, Any] | None=None, metadata: dict[str, Any] | None=None, pytorch_device: str='cpu', is_pytorch: bool=False):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            static_argnames = tuple((k for k, v in self._sample_kwargs.items() if isinstance(v, (str, bool, int, float)) or v is None))
            self._sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=static_argnames if static_argnames else None)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None=None) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs['noise'] = noise
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        raw_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        if isinstance(raw_result, tuple):
            actions, switch_prob = raw_result
            outputs = {'state': inputs['state'], 'actions': actions, 'switch': switch_prob}
        else:
            outputs = {'state': inputs['state'], 'actions': raw_result}
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs['policy_timing'] = {'infer_ms': model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

class PolicyRecorder(_base_policy.BasePolicy):

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy
        logging.info(f'Dumping policy records to: {record_dir}')
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:
        results = self._policy.infer(obs)
        data = {'inputs': obs, 'outputs': results}
        data = flax.traverse_util.flatten_dict(data, sep='/')
        output_path = self._record_dir / f'step_{self._record_step}'
        self._record_step += 1
        np.save(output_path, np.asarray(data))
        return results
