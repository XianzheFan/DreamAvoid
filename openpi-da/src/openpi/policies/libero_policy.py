import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

def make_libero_example() -> dict:
    return {'observation/state': np.random.rand(8), 'observation/image': np.random.randint(256, size=(224, 224, 3), dtype=np.uint8), 'observation/wrist_image': np.random.randint(256, size=(224, 224, 3), dtype=np.uint8), 'prompt': 'do something'}

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, 'c h w -> h w c')
    return image

@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data['observation/image'])
        wrist_image = _parse_image(data['observation/wrist_image'])
        inputs = {'state': data['observation/state'], 'image': {'base_0_rgb': base_image, 'left_wrist_0_rgb': wrist_image, 'right_wrist_0_rgb': np.zeros_like(base_image)}, 'image_mask': {'base_0_rgb': np.True_, 'left_wrist_0_rgb': np.True_, 'right_wrist_0_rgb': np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_}}
        if 'actions' in data:
            inputs['actions'] = data['actions']
        if 'prompt' in data:
            inputs['prompt'] = data['prompt']
        if 'switch_label' in data:
            inputs['switch_label'] = data['switch_label']
        return inputs

@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        result = {'actions': np.asarray(data['actions'][:, :7])}
        if 'switch' in data:
            result['switch'] = data['switch']
        return result
