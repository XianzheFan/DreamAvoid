import dataclasses
import enum
import logging
import socket
import tyro
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from openpi.models.pi0_sde import Pi0SDEConfig

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
    noise_level: float = 0.3
    num_steps: int = 5
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

def create_sde_policy(args: Args) -> _policy.Policy:
    config_name = args.policy.config if isinstance(args.policy, Checkpoint) else DEFAULT_CHECKPOINT[args.env].config
    checkpoint_dir = args.policy.dir if isinstance(args.policy, Checkpoint) else DEFAULT_CHECKPOINT[args.env].dir
    full_config = _config.get_config(config_name)
    original_model_config = full_config.model
    model_kwargs = dataclasses.asdict(original_model_config)
    new_model_config = Pi0SDEConfig(**model_kwargs, noise_method='flow_sde', noise_level=args.noise_level, num_steps=args.num_steps)
    full_config = dataclasses.replace(full_config, model=new_model_config)
    return _policy_config.create_trained_policy(full_config, checkpoint_dir, default_prompt=args.default_prompt)

def main(args: Args) -> None:
    policy = create_sde_policy(args)
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
