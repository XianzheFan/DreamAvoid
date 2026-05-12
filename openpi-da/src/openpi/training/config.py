import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None

@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ('actions',)
    prompt_from_task: bool = False
    local_dirs: Sequence[str] = ()
    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

class GroupFactory(Protocol):

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        pass

@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI0_SDE | _model.ModelType.PI0_RLT:
                return _transforms.Group(inputs=[_transforms.InjectDefaultPrompt(self.default_prompt), _transforms.ResizeImages(224, 224), _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len)), _transforms.PadStatesAndActions(model_config.action_dim)])
            case _model.ModelType.PI05 | _model.ModelType.PI05_SDE | _model.ModelType.PI05_RLT:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(inputs=[_transforms.InjectDefaultPrompt(self.default_prompt), _transforms.ResizeImages(224, 224), _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len), discrete_state_input=model_config.discrete_state_input), _transforms.PadStatesAndActions(model_config.action_dim)])
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = _tokenizer.FASTTokenizer if model_config.fast_model_tokenizer is None else model_config.fast_model_tokenizer
                tokenizer_kwargs = {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                return _transforms.Group(inputs=[_transforms.InjectDefaultPrompt(self.default_prompt), _transforms.ResizeImages(224, 224), _transforms.TokenizeFASTInputs(tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs))], outputs=[_transforms.ExtractFASTActions(tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs), action_horizon=model_config.action_horizon, action_dim=model_config.action_dim)])

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        pass

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(self.base_config or DataConfig(), repo_id=repo_id, asset_id=asset_id, norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id), use_quantile_norm=model_config.model_type != ModelType.PI0)

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f'Loaded norm stats from {data_assets_dir}')
            return norm_stats
        except FileNotFoundError:
            logging.info(f'Norm stats not found in {data_assets_dir}, skipping.')
        return None

@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = 'fake'

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)

@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), data_transforms=self.data_transforms(model_config), model_transforms=self.model_transforms(model_config))

@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default=_transforms.Group(inputs=[_transforms.RepackTransform({'images': {'cam_high': 'observation.images.top'}, 'state': 'observation.state', 'actions': 'action'})]))
    action_sequence_keys: Sequence[str] = ('action',)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)], outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)])
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(inputs=[_transforms.DeltaActions(delta_action_mask)], outputs=[_transforms.AbsoluteActions(delta_action_mask)])
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=self.repack_transforms, data_transforms=data_transforms, model_transforms=model_transforms, action_sequence_keys=self.action_sequence_keys)

@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform({'observation/image': 'image', 'observation/wrist_image': 'wrist_image', 'observation/state': 'state', 'actions': 'actions', 'prompt': 'prompt'})])
        data_transforms = _transforms.Group(inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)], outputs=[libero_policy.LiberoOutputs()])
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(inputs=[_transforms.DeltaActions(delta_action_mask)], outputs=[_transforms.AbsoluteActions(delta_action_mask)])
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack_transform, data_transforms=data_transforms, model_transforms=model_transforms)

@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (droid_rlds_dataset.RLDSDataset(name='droid', version='1.0.1', weight=1.0, filter_dict_path='gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json'),)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform({'observation/exterior_image_1_left': 'observation/image', 'observation/wrist_image_left': 'observation/wrist_image', 'observation/joint_position': 'observation/joint_position', 'observation/gripper_position': 'observation/gripper_position', 'actions': 'actions', 'prompt': 'prompt'})])
        data_transforms = _transforms.Group(inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)], outputs=[droid_policy.DroidOutputs()])
        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(inputs=[_transforms.DeltaActions(delta_action_mask)], outputs=[_transforms.AbsoluteActions(delta_action_mask)])
        model_transforms = ModelTransformFactory()(model_config)
        assert self.rlds_data_dir is not None, 'Need to set rlds data dir for RLDS data loader.'
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack_transform, data_transforms=data_transforms, model_transforms=model_transforms, rlds_data_dir=self.rlds_data_dir, action_space=self.action_space, datasets=self.datasets)

@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform({'observation/exterior_image_1_left': 'exterior_image_1_left', 'observation/exterior_image_2_left': 'exterior_image_2_left', 'observation/wrist_image_left': 'wrist_image_left', 'observation/joint_position': 'joint_position', 'observation/gripper_position': 'gripper_position', 'actions': 'actions', 'prompt': 'prompt'})])
        data_transforms = _transforms.Group(inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)], outputs=[droid_policy.DroidOutputs()])
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack_transform, data_transforms=data_transforms, model_transforms=model_transforms)

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = 'openpi'
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal['bfloat16', 'float32'] = 'bfloat16'
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    assets_base_dir: str = './assets'
    checkpoint_base_dir: str = './checkpoints'
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError('--exp_name must be set')
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError('Cannot resume and overwrite at the same time.')
_CONFIGS = [TrainConfig(name='pi0_aloha', model=pi0_config.Pi0Config(), data=LeRobotAlohaDataConfig(assets=AssetsConfig(asset_id='trossen')), policy_metadata={'reset_pose': [0, -1.5, 1.5, 0, 0, 0]}), TrainConfig(name='pi05_aloha', model=pi0_config.Pi0Config(pi05=True), data=LeRobotAlohaDataConfig(assets=AssetsConfig(asset_id='trossen')), policy_metadata={'reset_pose': [0, -1.5, 1.5, 0, 0, 0]}), TrainConfig(name='pi0_aloha_towel', model=pi0_config.Pi0Config(), data=LeRobotAlohaDataConfig(assets=AssetsConfig(asset_id='trossen'), default_prompt='fold the towel'), policy_metadata={'reset_pose': [0, -1.5, 1.5, 0, 0, 0]}), TrainConfig(name='pi0_aloha_tupperware', model=pi0_config.Pi0Config(), data=LeRobotAlohaDataConfig(assets=AssetsConfig(asset_id='trossen'), default_prompt='open the tupperware and put the food on the plate'), policy_metadata={'reset_pose': [0, -1.5, 1.5, 0, 0, 0]}), TrainConfig(name='pi0_droid', model=pi0_config.Pi0Config(action_horizon=10), data=SimpleDataConfig(assets=AssetsConfig(asset_id='droid'), data_transforms=lambda model: _transforms.Group(inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)], outputs=[droid_policy.DroidOutputs()]), base_config=DataConfig(prompt_from_task=True))), TrainConfig(name='pi0_fast_droid', model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10), data=SimpleDataConfig(assets=AssetsConfig(asset_id='droid'), data_transforms=lambda model: _transforms.Group(inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)], outputs=[droid_policy.DroidOutputs()]), base_config=DataConfig(prompt_from_task=True))), TrainConfig(name='pi05_droid', model=pi0_config.Pi0Config(action_horizon=15, pi05=True), data=SimpleDataConfig(assets=AssetsConfig(asset_id='droid'), data_transforms=lambda model: _transforms.Group(inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)], outputs=[droid_policy.DroidOutputs()]), base_config=DataConfig(prompt_from_task=True))), TrainConfig(name='pi0_libero', model=pi0_config.Pi0Config(), data=LeRobotLiberoDataConfig(repo_id='physical-intelligence/libero', base_config=DataConfig(prompt_from_task=True), extra_delta_transform=True), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi0_base/params'), num_train_steps=30000), TrainConfig(name='pi0_libero_low_mem_finetune', model=pi0_config.Pi0Config(paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora'), data=LeRobotLiberoDataConfig(repo_id='physical-intelligence/libero', base_config=DataConfig(prompt_from_task=True), extra_delta_transform=True), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi0_base/params'), num_train_steps=30000, freeze_filter=pi0_config.Pi0Config(paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora').get_freeze_filter(), ema_decay=None), TrainConfig(name='pi0_fast_libero', model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180), data=LeRobotLiberoDataConfig(repo_id='physical-intelligence/libero', base_config=DataConfig(prompt_from_task=True), extra_delta_transform=True), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi0_fast_base/params'), num_train_steps=30000), TrainConfig(name='pi0_fast_libero_low_mem_finetune', model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant='gemma_2b_lora'), data=LeRobotLiberoDataConfig(repo_id='physical-intelligence/libero', base_config=DataConfig(prompt_from_task=True), extra_delta_transform=True), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi0_fast_base/params'), num_train_steps=30000, freeze_filter=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant='gemma_2b_lora').get_freeze_filter(), ema_decay=None), TrainConfig(name='pi05_libero', model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False), data=LeRobotLiberoDataConfig(repo_id='physical-intelligence/libero', base_config=DataConfig(prompt_from_task=True), extra_delta_transform=False), batch_size=256, lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=10000, peak_lr=5e-05, decay_steps=1000000, decay_lr=5e-05), optimizer=_optimizer.AdamW(clip_gradient_norm=1.0), ema_decay=0.999, weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params'), pytorch_weight_path='/path/to/your/pytorch_weight_path', num_train_steps=30000), TrainConfig(name='pi05_libero_switch', model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, switch_head=True), data=LeRobotLiberoDataConfig(repo_id='physical-intelligence/libero', base_config=DataConfig(prompt_from_task=True), extra_delta_transform=False), batch_size=256, lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=10000, peak_lr=5e-05, decay_steps=1000000, decay_lr=5e-05), optimizer=_optimizer.AdamW(clip_gradient_norm=1.0), ema_decay=0.999, weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params'), pytorch_weight_path='/path/to/your/pytorch_weight_path', num_train_steps=30000), TrainConfig(name='pi0_aloha_pen_uncap', model=pi0_config.Pi0Config(), data=LeRobotAlohaDataConfig(repo_id='physical-intelligence/aloha_pen_uncap_diverse', assets=AssetsConfig(assets_dir='gs://openpi-assets/checkpoints/pi0_base/assets', asset_id='trossen'), default_prompt='uncap the pen', repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform({'images': {'cam_high': 'observation.images.cam_high', 'cam_left_wrist': 'observation.images.cam_left_wrist', 'cam_right_wrist': 'observation.images.cam_right_wrist'}, 'state': 'observation.state', 'actions': 'action'})])), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi0_base/params'), num_train_steps=20000), TrainConfig(name='pi05_aloha_pen_uncap', model=pi0_config.Pi0Config(pi05=True), data=LeRobotAlohaDataConfig(repo_id='physical-intelligence/aloha_pen_uncap_diverse', assets=AssetsConfig(assets_dir='gs://openpi-assets/checkpoints/pi05_base/assets', asset_id='trossen'), default_prompt='uncap the pen', repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform({'images': {'cam_high': 'observation.images.cam_high', 'cam_left_wrist': 'observation.images.cam_left_wrist', 'cam_right_wrist': 'observation.images.cam_right_wrist'}, 'state': 'observation.state', 'actions': 'action'})])), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params'), num_train_steps=20000, batch_size=64), TrainConfig(name='pi0_fast_full_droid_finetune', model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=16, max_token_len=180), data=RLDSDroidDataConfig(repo_id='droid', rlds_data_dir='<path_to_droid_rlds_dataset>', action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi0_fast_base/params'), lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-05, decay_steps=1000000, decay_lr=5e-05), num_train_steps=100000, batch_size=256, log_interval=100, save_interval=5000, keep_period=20000, num_workers=0), TrainConfig(name='pi05_full_droid_finetune', model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16), data=RLDSDroidDataConfig(repo_id='droid', rlds_data_dir='<path_to_droid_rlds_dataset>', action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION, assets=AssetsConfig(assets_dir='gs://openpi-assets/checkpoints/pi05_base/assets/', asset_id='droid')), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params'), lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=5e-05, decay_steps=1000000, decay_lr=5e-05), num_train_steps=100000, batch_size=256, log_interval=100, save_interval=5000, keep_period=10000, num_workers=0), TrainConfig(name='pi05_droid_finetune', model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16), data=LeRobotDROIDDataConfig(repo_id='your_hf_username/my_droid_dataset', base_config=DataConfig(prompt_from_task=True), assets=AssetsConfig(assets_dir='gs://openpi-assets/checkpoints/pi05_droid/assets', asset_id='droid')), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_droid/params'), num_train_steps=20000, batch_size=32), TrainConfig(name='pi0_aloha_sim', model=pi0_config.Pi0Config(), data=LeRobotAlohaDataConfig(repo_id='lerobot/aloha_sim_transfer_cube_human', default_prompt='Transfer cube', use_delta_joint_actions=False), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi0_base/params'), num_train_steps=20000), TrainConfig(name='debug', data=FakeDataConfig(), batch_size=2, model=pi0_config.Pi0Config(paligemma_variant='dummy', action_expert_variant='dummy'), save_interval=100, overwrite=True, exp_name='debug', num_train_steps=10, wandb_enabled=False), TrainConfig(name='debug_restore', data=FakeDataConfig(), batch_size=2, model=pi0_config.Pi0Config(paligemma_variant='dummy', action_expert_variant='dummy'), weight_loader=weight_loaders.CheckpointWeightLoader('./checkpoints/debug/debug/9/params'), overwrite=True, exp_name='debug', num_train_steps=10, wandb_enabled=False), TrainConfig(name='debug_pi05', model=pi0_config.Pi0Config(pi05=True, paligemma_variant='dummy', action_expert_variant='dummy'), data=FakeDataConfig(), batch_size=2, num_train_steps=10, overwrite=True, exp_name='debug_pi05', wandb_enabled=False), TrainConfig(name='pi05_pnp_cup', model=pi0_config.Pi0Config(pi05=True), data=LeRobotAlohaDataConfig(repo_id='pnp_cup', base_config=DataConfig(local_dirs=['data/pnp_cup_0415']), assets=AssetsConfig(assets_dir='gs://openpi-assets/checkpoints/pi05_base/assets', asset_id='trossen'), default_prompt='Pick up the paper cup and put it into the cup sleeve.', repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform({'images': {'cam_high': 'observation.images.cam_high', 'cam_left_wrist': 'observation.images.cam_left_wrist', 'cam_right_wrist': 'observation.images.cam_right_wrist'}, 'state': 'observation.state', 'actions': 'action'})])), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params'), num_train_steps=40000, batch_size=64, num_workers=8), TrainConfig(name='pi05_plug', model=pi0_config.Pi0Config(pi05=True), data=LeRobotAlohaDataConfig(repo_id='plug', base_config=DataConfig(local_dirs=['data/plug_merged']), assets=AssetsConfig(assets_dir='gs://openpi-assets/checkpoints/pi05_base/assets', asset_id='trossen'), default_prompt='Plug the charger into the socket.', repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform({'images': {'cam_high': 'observation.images.cam_high', 'cam_left_wrist': 'observation.images.cam_left_wrist', 'cam_right_wrist': 'observation.images.cam_right_wrist'}, 'state': 'observation.state', 'actions': 'action'})])), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params'), num_train_steps=50000, batch_size=64, num_workers=8), TrainConfig(name='pi05_spray_cap', model=pi0_config.Pi0Config(pi05=True), data=LeRobotAlohaDataConfig(repo_id='spray_cap', base_config=DataConfig(local_dirs=['data/spray_cap_truncated']), assets=AssetsConfig(assets_dir='gs://openpi-assets/checkpoints/pi05_base/assets', asset_id='trossen'), default_prompt='Open the cap of the spray and place it on the table', repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform({'images': {'cam_high': 'observation.images.cam_high', 'cam_left_wrist': 'observation.images.cam_left_wrist', 'cam_right_wrist': 'observation.images.cam_right_wrist'}, 'state': 'observation.state', 'actions': 'action'})])), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params'), num_train_steps=50000, batch_size=64, num_workers=8), TrainConfig(name='pi05_screw', model=pi0_config.Pi0Config(pi05=True), data=LeRobotAlohaDataConfig(repo_id='screw', base_config=DataConfig(local_dirs=['data/screw_0426']), assets=AssetsConfig(assets_dir='gs://openpi-assets/checkpoints/pi05_base/assets', asset_id='trossen'), default_prompt='Insert the screw into the hole in the box.', repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform({'images': {'cam_high': 'observation.images.cam_high', 'cam_left_wrist': 'observation.images.cam_left_wrist', 'cam_right_wrist': 'observation.images.cam_right_wrist'}, 'state': 'observation.state', 'actions': 'action'})])), weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params'), num_train_steps=50000, batch_size=64, num_workers=8), *roboarena_config.get_roboarena_configs(), *polaris_config.get_polaris_configs()]
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError('Config names must be unique.')
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}

def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ''
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
