import asyncio
import dataclasses
import http
import logging
import os
import pathlib
import socket
import time
import traceback
from typing import Any
import jax.numpy as jnp
from openpi_client import msgpack_numpy
import tyro
import websockets.asyncio.server as _server
import websockets.frames
from openpi.models import model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

@dataclasses.dataclass
class Args:
    policy_config: str
    policy_dir: str
    port: int = 8000
    sde_noise_level: float = 0.3
    sde_num_steps: int = 3
    ode_num_steps: int = 10
    default_prompt: str | None = None

def _build_policies(args: Args) -> tuple[_policy.Policy, _policy.Policy, dict[str, Any]]:
    train_config = _config.get_config(args.policy_config)
    checkpoint_dir = pathlib.Path(download.maybe_download(str(args.policy_dir)))
    logging.info('Loading model weights once for ODE+SDE share...')
    weight_path = checkpoint_dir / 'params'
    params = _model.restore_params(weight_path, dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError('Asset id is required to load norm stats.')
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / 'assets', data_config.asset_id)
    input_transforms = [transforms.InjectDefaultPrompt(args.default_prompt), *data_config.data_transforms.inputs, transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm), *data_config.model_transforms.inputs]
    output_transforms = [*data_config.model_transforms.outputs, transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm), *data_config.data_transforms.outputs]
    has_switch = bool(getattr(train_config.model, 'switch_head', False))
    ode_kwargs: dict[str, Any] = {'mode': 'ode', 'num_steps': args.ode_num_steps}
    if has_switch:
        ode_kwargs['return_switch'] = True
    ode = _policy.Policy(model, transforms=input_transforms, output_transforms=output_transforms, sample_kwargs=ode_kwargs, metadata=train_config.policy_metadata)
    sde_kwargs: dict[str, Any] = {'mode': 'sde', 'num_steps': args.sde_num_steps, 'noise_level': args.sde_noise_level}
    sde = _policy.Policy(model, transforms=input_transforms, output_transforms=output_transforms, sample_kwargs=sde_kwargs, metadata=train_config.policy_metadata)
    return (ode, sde, dict(train_config.policy_metadata or {}))

class CombinedPolicyServer:

    def __init__(self, ode_policy: _policy.Policy, sde_policy: _policy.Policy, metadata: dict[str, Any], host: str='0.0.0.0', port: int=8000):
        self._ode = ode_policy
        self._sde = sde_policy
        self._metadata = metadata
        self._host = host
        self._port = port

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None, process_request=_health_check) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logging.info(f'Connection from {websocket.remote_address} opened')
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                mode = obs.pop('mode', 'ode')
                policy = self._sde if mode == 'sde' else self._ode
                infer_time = time.monotonic()
                action = policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                action['server_timing'] = {'infer_ms': infer_time * 1000, 'mode': mode}
                if prev_total_time is not None:
                    action['server_timing']['prev_total_ms'] = prev_total_time * 1000
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logging.info(f'Connection from {websocket.remote_address} closed')
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason='Internal server error. Traceback included in previous frame.')
                raise

def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == '/healthz':
        return connection.respond(http.HTTPStatus.OK, 'OK\n')
    return None

def main(args: Args) -> None:
    ode_policy, sde_policy, metadata = _build_policies(args)
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = '?'
    logging.info("Creating combined server on port %d (host: %s, ip: %s). Routes: mode=='sde' -> SDE (noise_level=%.3f, num_steps=%d); else -> ODE (num_steps=%d).", args.port, hostname, local_ip, args.sde_noise_level, args.sde_num_steps, args.ode_num_steps)
    server = CombinedPolicyServer(ode_policy=ode_policy, sde_policy=sde_policy, metadata=metadata, host='0.0.0.0', port=args.port)
    server.serve_forever()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
