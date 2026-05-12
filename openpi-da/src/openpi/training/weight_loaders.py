import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable
import flax.traverse_util
import numpy as np
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
logger = logging.getLogger(__name__)

@runtime_checkable
class WeightLoader(Protocol):

    def load(self, params: at.Params) -> at.Params:
        pass

@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):

    def load(self, params: at.Params) -> at.Params:
        return params

@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex='.*lora.*')

@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download('gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz', gs={'token': 'anon'})
        with path.open('rb') as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {'PaliGemma': flax.traverse_util.unflatten_dict(flat_params, sep='/')['params']}
        return _merge_params(loaded_params, params, missing_regex='.*')

def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    flat_ref = flax.traverse_util.flatten_dict(params, sep='/')
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep='/')
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
    flat_loaded.clear()
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]
    return flax.traverse_util.unflatten_dict(result, sep='/')
