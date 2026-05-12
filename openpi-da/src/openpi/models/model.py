import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar
import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at
logger = logging.getLogger('openpi')
ArrayT = TypeVar('ArrayT', bound=jax.Array | torch.Tensor | np.ndarray)

class ModelType(enum.Enum):
    PI0 = 'pi0'
    PI0_FAST = 'pi0_fast'
    PI0_SDE = 'pi0_sde'
    PI05_SDE = 'pi05_sde'
    PI05 = 'pi05'
    PI0_RLT = 'pi0_rlt'
    PI05_RLT = 'pi05_rlt'
IMAGE_KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')
IMAGE_RESOLUTION = (224, 224)

@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    images: dict[str, at.Float[ArrayT, '*b h w c']]
    image_masks: dict[str, at.Bool[ArrayT, '*b']]
    state: at.Float[ArrayT, '*b s']
    tokenized_prompt: at.Int[ArrayT, '*b l'] | None = None
    tokenized_prompt_mask: at.Bool[ArrayT, '*b l'] | None = None
    token_ar_mask: at.Int[ArrayT, '*b l'] | None = None
    token_loss_mask: at.Bool[ArrayT, '*b l'] | None = None
    switch_label: at.Float[ArrayT, '*b'] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> 'Observation[ArrayT]':
        if ('tokenized_prompt' in data) != ('tokenized_prompt_mask' in data):
            raise ValueError('tokenized_prompt and tokenized_prompt_mask must be provided together.')
        for key in data['image']:
            if data['image'][key].dtype == np.uint8:
                data['image'][key] = data['image'][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data['image'][key], 'dtype') and data['image'][key].dtype == torch.uint8:
                data['image'][key] = data['image'][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(images=data['image'], image_masks=data['image_mask'], state=data['state'], tokenized_prompt=data.get('tokenized_prompt'), tokenized_prompt_mask=data.get('tokenized_prompt_mask'), token_ar_mask=data.get('token_ar_mask'), token_loss_mask=data.get('token_loss_mask'), switch_label=data.get('switch_label'))

    def to_dict(self) -> at.PyTree[ArrayT]:
        result = dataclasses.asdict(self)
        result['image'] = result.pop('images')
        result['image_mask'] = result.pop('image_masks')
        return result
Actions = at.Float[ArrayT, '*b ah ad']

def preprocess_observation(rng: at.KeyArrayLike | None, observation: Observation, *, train: bool=False, image_keys: Sequence[str]=IMAGE_KEYS, image_resolution: tuple[int, int]=IMAGE_RESOLUTION) -> Observation:
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f'images dict missing keys: expected {image_keys}, got {list(observation.images)}')
    batch_shape = observation.state.shape[:-1]
    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f'Resizing image {key} from {image.shape[1:3]} to {image_resolution}')
            image = image_tools.resize_with_pad(image, *image_resolution)
        if train:
            image = image / 2.0 + 0.5
            transforms = []
            if 'wrist' not in key:
                height, width = image.shape[1:3]
                transforms += [augmax.RandomCrop(int(width * 0.95), int(height * 0.95)), augmax.Resize(width, height), augmax.Rotate((-5, 5))]
            transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)
            image = image * 2.0 - 1.0
        out_images[key] = image
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])
    return Observation(images=out_images, image_masks=out_masks, state=observation.state, tokenized_prompt=observation.tokenized_prompt, tokenized_prompt_mask=observation.tokenized_prompt_mask, token_ar_mask=observation.token_ar_mask, token_loss_mask=observation.token_loss_mask, switch_label=observation.switch_label)

@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    action_dim: int
    action_horizon: int
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        pass

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> 'BaseModel':
        pass

    def load(self, params: at.Params, *, remove_extra_params: bool=True) -> 'BaseModel':
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f'train_config: {train_config}')
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int=1) -> tuple[Observation, Actions]:
        pass

    def fake_obs(self, batch_size: int=1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int=1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)

@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(self, rng: at.KeyArrayLike, observation: Observation, actions: Actions, *, train: bool=False) -> at.Float[at.Array, '*b ah']:
        ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions:
        ...

def restore_params(params_path: pathlib.Path | str, *, restore_type: type[np.ndarray] | type[jax.Array]=jax.Array, dtype: jnp.dtype | None=None, sharding: jax.sharding.Sharding | None=None) -> at.Params:
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith('gs://') else params_path
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ('x',))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {'params': metadata['params']}
        params = ckptr.restore(params_path, ocp.args.PyTreeRestore(item=item, restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item)))['params']
    flat_params = traverse_util.flatten_dict(params)
    if all((kp[-1] == 'value' for kp in flat_params)):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
