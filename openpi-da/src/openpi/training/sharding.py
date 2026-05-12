import contextlib
import logging
import jax
import numpy as np
BATCH_AXIS = 'batch'
FSDP_AXIS = 'fsdp'
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)

class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None

def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(f'Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}.')
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))

@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    if _MeshState.active_mesh is not None:
        raise ValueError('Cannot nest set_mesh context managers.')
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None

def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS)))

def fsdp_sharding(pytree, mesh: jax.sharding.Mesh, *, min_size_mbytes: int=4, log: bool=False):
    min_size_bytes = min_size_mbytes * 2 ** 20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if not hasattr(array, 'shape'):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if (arr_size := (np.prod(array.shape) * np.dtype(array.dtype).itemsize)) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(f'Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2 ** 20:.2f} MiB) along axis {i}')
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
        if log:
            logging.warning(f'Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}')
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
