from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
_orig_lerobot_init = lerobot_dataset.LeRobotDataset.__init__

def _patched_lerobot_init(self, *args, **kwargs):
    _orig_lerobot_init(self, *args, **kwargs)

def _patched_init_wrapper(self, *args, **kwargs):
    _orig_stack = torch.stack

    def _safe_stack(tensors, *a, **kw):
        if not isinstance(tensors, (list, tuple)):
            tensors = list(tensors)
        return _orig_stack(tensors, *a, **kw)
    torch.stack = _safe_stack
    try:
        _orig_lerobot_init(self, *args, **kwargs)
    finally:
        torch.stack = _orig_stack
lerobot_dataset.LeRobotDataset.__init__ = _patched_init_wrapper
_orig_get_query_timestamps = lerobot_dataset.LeRobotDataset._get_query_timestamps
_orig_query_hf_dataset = lerobot_dataset.LeRobotDataset._query_hf_dataset

def _patched_get_query_timestamps(self, current_ts, query_indices=None):
    query_timestamps = {}
    for key in self.meta.video_keys:
        if query_indices is not None and key in query_indices:
            timestamps = self.hf_dataset.select(query_indices[key])['timestamp']
            query_timestamps[key] = torch.stack(list(timestamps)).tolist()
        else:
            query_timestamps[key] = [current_ts]
    return query_timestamps

def _patched_query_hf_dataset(self, query_indices):
    return {key: torch.stack(list(self.hf_dataset.select(q_idx)[key])) for key, q_idx in query_indices.items() if key not in self.meta.video_keys}
lerobot_dataset.LeRobotDataset._get_query_timestamps = _patched_get_query_timestamps
lerobot_dataset.LeRobotDataset._query_hf_dataset = _patched_query_hf_dataset
import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms
T_co = TypeVar('T_co', covariant=True)

class Dataset(Protocol[T_co]):

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError('Subclasses of Dataset should implement __getitem__.')

    def __len__(self) -> int:
        raise NotImplementedError('Subclasses of Dataset should implement __len__.')

class IterableDataset(Protocol[T_co]):

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError('Subclasses of IterableDataset should implement __iter__.')

    def __len__(self) -> int:
        raise NotImplementedError('Subclasses of Dataset should implement __len__.')

class DataLoader(Protocol[T_co]):

    def data_config(self) -> _config.DataConfig:
        raise NotImplementedError('Subclasses of DataLoader should implement data_config.')

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError('Subclasses of DataLoader should implement __iter__.')

class TransformedDataset(Dataset[T_co]):

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)

class IterableTransformedDataset(IterableDataset[T_co]):

    def __init__(self, dataset: IterableDataset, transforms: Sequence[_transforms.DataTransformFn], *, is_batched: bool=False):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                batch_size = next((v.shape[0] for v in sample.values()))
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]
                transformed = [self._transform(s) for s in individual_samples]
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)

class FakeDataset(Dataset):

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)
        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)
        return {**observation.to_dict(), 'actions': action}

    def __len__(self) -> int:
        return self._num_samples

def create_torch_dataset(data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig) -> Dataset:
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError('Repo ID is not set. Cannot create dataset.')
    if repo_id == 'fake':
        return FakeDataset(model_config, num_samples=1024)
    local_dirs = data_config.local_dirs
    if local_dirs:
        datasets = []
        first_meta = None
        for local_dir in local_dirs:
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=local_dir)
            if first_meta is None:
                first_meta = dataset_meta
            ds = lerobot_dataset.LeRobotDataset(repo_id, root=local_dir, delta_timestamps={key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys}, video_backend='pyav')
            datasets.append(ds)
        dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
        if data_config.prompt_from_task and first_meta is not None:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(first_meta.tasks)])
    else:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, delta_timestamps={key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys}, video_backend='pyav')
        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset

def create_rlds_dataset(data_config: _config.DataConfig, action_horizon: int, batch_size: int, *, shuffle: bool=False) -> Dataset:
    return DroidRldsDataset(data_dir=data_config.rlds_data_dir, batch_size=batch_size, shuffle=shuffle, action_chunk_size=action_horizon, action_space=data_config.action_space, datasets=data_config.datasets)

def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool=False) -> Dataset:
    norm_stats = {}
    if data_config.repo_id != 'fake' and (not skip_norm_stats):
        if data_config.norm_stats is None:
            raise ValueError('Normalization stats not found. Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.')
        norm_stats = data_config.norm_stats
    return TransformedDataset(dataset, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm), *data_config.model_transforms.inputs])

def transform_iterable_dataset(dataset: IterableDataset, data_config: _config.DataConfig, *, skip_norm_stats: bool=False, is_batched: bool=False) -> IterableDataset:
    norm_stats = {}
    if data_config.repo_id != 'fake' and (not skip_norm_stats):
        if data_config.norm_stats is None:
            raise ValueError('Normalization stats not found. Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.')
        norm_stats = data_config.norm_stats
    return IterableTransformedDataset(dataset, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm), *data_config.model_transforms.inputs], is_batched=is_batched)

def create_data_loader(config: _config.TrainConfig, *, sharding: jax.sharding.Sharding | None=None, shuffle: bool=False, num_batches: int | None=None, skip_norm_stats: bool=False, framework: Literal['jax', 'pytorch']='jax') -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f'data_config: {data_config}')
    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(data_config, action_horizon=config.model.action_horizon, batch_size=config.batch_size, sharding=sharding, shuffle=shuffle, num_batches=num_batches, skip_norm_stats=skip_norm_stats, framework=framework)
    return create_torch_data_loader(data_config, model_config=config.model, action_horizon=config.model.action_horizon, batch_size=config.batch_size, sharding=sharding, shuffle=shuffle, num_batches=num_batches, num_workers=config.num_workers, seed=config.seed, skip_norm_stats=skip_norm_stats, framework=framework)

def create_torch_data_loader(data_config: _config.DataConfig, model_config: _model.BaseModelConfig, action_horizon: int, batch_size: int, *, sharding: jax.sharding.Sharding | None=None, skip_norm_stats: bool=False, shuffle: bool=False, num_batches: int | None=None, num_workers: int=0, seed: int=0, framework: str='jax') -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    sampler = None
    if framework == 'pytorch':
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(), shuffle=shuffle, drop_last=True)
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
    logging.info(f'local_batch_size: {local_batch_size}')
    data_loader = TorchDataLoader(dataset, local_batch_size=local_batch_size, sharding=None if framework == 'pytorch' else sharding, shuffle=sampler is None and shuffle, sampler=sampler, num_batches=num_batches, num_workers=num_workers, seed=seed, framework=framework)
    return DataLoaderImpl(data_config, data_loader)

def create_rlds_data_loader(data_config: _config.DataConfig, action_horizon: int, batch_size: int, *, sharding: jax.sharding.Sharding | None=None, skip_norm_stats: bool=False, shuffle: bool=False, num_batches: int | None=None, framework: str='jax') -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    if framework == 'pytorch':
        raise NotImplementedError('PyTorch RLDS data loader is not supported yet')
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)
    data_loader = RLDSDataLoader(dataset, sharding=sharding, num_batches=num_batches)
    return DataLoaderImpl(data_config, data_loader)

class TorchDataLoader:

    def __init__(self, dataset, local_batch_size: int, *, sharding: jax.sharding.Sharding | None=None, shuffle: bool=False, sampler: torch.utils.data.Sampler | None=None, num_batches: int | None=None, num_workers: int=0, seed: int=0, framework: str='jax'):
        if jax.process_count() > 1:
            raise NotImplementedError('Data loading with multiple processes is not supported.')
        if len(dataset) < local_batch_size:
            raise ValueError(f'Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).')
        self._sharding = sharding
        if sharding is None and framework == 'jax':
            self._sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ('B',)), jax.sharding.PartitionSpec('B'))
        self._num_batches = num_batches
        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context('spawn')
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(typing.cast(torch.utils.data.Dataset, dataset), batch_size=local_batch_size, shuffle=sampler is None and shuffle, sampler=sampler, num_workers=num_workers, multiprocessing_context=mp_context, persistent_workers=num_workers > 0, collate_fn=_collate_fn, worker_init_fn=_worker_init_fn, drop_last=True, generator=generator)

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)

def _collate_fn(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)

def _worker_init_fn(worker_id: int) -> None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

class RLDSDataLoader:

    def __init__(self, dataset: DroidRldsDataset, *, sharding: jax.sharding.Sharding | None=None, num_batches: int | None=None):
        self._dataset = dataset
        self._num_batches = num_batches
        if jax.process_count() > 1:
            raise NotImplementedError('Data loading with multiple processes is not supported.')
        if sharding is None:
            sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ('B',)), jax.sharding.PartitionSpec('B'))
        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)

class DataLoaderImpl(DataLoader):

    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield (_model.Observation.from_dict(batch), batch['actions'])
