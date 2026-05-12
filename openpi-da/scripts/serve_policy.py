import dataclasses
import enum
import logging
import socket
import tyro
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

class EnvMode(enum.Enum):
    ALOHA = 'aloha'
    ALOHA_SIM = 'aloha_sim'
    DROID = 'droid'
    LIBERO = 'libero'

@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str

@dataclasses.dataclass
class Default:
    pass

@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {EnvMode.ALOHA: Checkpoint(config='pi05_aloha', dir='gs://openpi-assets/checkpoints/pi05_base'), EnvMode.ALOHA_SIM: Checkpoint(config='pi0_aloha_sim', dir='gs://openpi-assets/checkpoints/pi0_aloha_sim'), EnvMode.DROID: Checkpoint(config='pi05_droid', dir='gs://openpi-assets/checkpoints/pi05_droid'), EnvMode.LIBERO: Checkpoint(config='pi05_libero', dir='gs://openpi-assets/checkpoints/pi05_libero')}

def create_default_policy(env: EnvMode, *, default_prompt: str | None=None) -> _policy.Policy:
    if (checkpoint := DEFAULT_CHECKPOINT.get(env)):
        return _policy_config.create_trained_policy(_config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt)
    raise ValueError(f'Unsupported environment mode: {env}')

def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(_config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt)
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)

def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, 'policy_records')
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info('Creating server (host: %s, ip: %s)', hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host='0.0.0.0', port=args.port, metadata=policy_metadata)
    server.serve_forever()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
