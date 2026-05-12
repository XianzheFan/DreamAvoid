import dataclasses
import logging
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
logger = logging.getLogger('openpi')

def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

@at.typecheck
def posemb_sincos(pos: at.Real[at.Array, ' b'], embedding_dim: int, min_period: float, max_period: float) -> at.Float[at.Array, 'b {embedding_dim}']:
    if embedding_dim % 2 != 0:
        raise ValueError(f'embedding_dim ({embedding_dim}) must be divisible by 2')
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum('i,j->ij', pos, 1.0 / period * 2 * jnp.pi, precision=jax.lax.Precision.HIGHEST)
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)

@dataclasses.dataclass(frozen=True)
class Pi0SDEConfig(pi0_config.Pi0Config):
    noise_method: str = 'flow_sde'
    noise_level: float = 0.3
    num_steps: int = 5

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05_SDE
        return _model.ModelType.PI0_SDE

    @override
    def inputs_spec(self, *, batch_size: int=1) -> tuple[_model.Observation, _model.Actions]:
        return super().inputs_spec(batch_size=batch_size)

    @override
    def create(self, rng: at.KeyArrayLike) -> 'Pi0SDE':
        return Pi0SDE(self, rngs=nnx.Rngs(rng))

class Pi0SDE(_model.BaseModel):

    def __init__(self, config: Pi0SDEConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(_gemma.Module(configs=[paligemma_config, action_expert_config], embed_dtype=config.dtype, adarms=config.pi05))
        llm.lazy_init(rngs=rngs, method='init', use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(_siglip.Module(num_classes=paligemma_config.width, variant='So400m/14', pool_type='none', scan=True, dtype_mm=config.dtype))
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.deterministic = True

    @at.typecheck
    def embed_prefix(self, obs: _model.Observation) -> tuple[at.Float[at.Array, 'b s emb'], at.Bool[at.Array, 'b s'], at.Bool[at.Array, ' s']]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], 'b -> b s', s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method='embed')
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return (tokens, input_mask, ar_mask)

    @at.typecheck
    def embed_suffix(self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, ' b']) -> tuple[at.Float[at.Array, 'b s emb'], at.Bool[at.Array, 'b s'], at.Bool[at.Array, ' s'], at.Float[at.Array, 'b emb'] | None]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=0.004, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, 'b emb -> b s emb', s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + [False] * (self.action_horizon - 1)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return (tokens, input_mask, ar_mask, adarms_cond)

    @override
    def compute_loss(self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool=False) -> at.Float[at.Array, '*b ah']:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond])
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, *, num_steps: int | at.Int[at.Array, '']=10, noise: at.Float[at.Array, 'b ah ad'] | None=None) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        rng_init, rng_loop = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(rng_init, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step_fn(carry):
            x_t, time, current_rng = carry
            step_rng, next_rng = jax.random.split(current_rng)
            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, (batch_size,)))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar)
            prefix_vis_mask = einops.repeat(prefix_mask, 'b p -> b s p', s=suffix_tokens.shape[1])
            full_mask = jnp.concatenate([prefix_vis_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_mask, positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
            v_t = self.action_out_proj(suffix_out[:, -self.config.action_horizon:])
            delta = jnp.abs(dt)
            denom = jnp.where(time >= 1.0 - 0.0001, delta, 1.0 - time)
            sigma_i = self.config.noise_level * jnp.sqrt(time / denom)
            x0_pred = x_t - v_t * time
            x1_pred = x_t + v_t * (1.0 - time)
            x0_weight = 1.0 - (time - delta)
            x1_weight = time - delta - sigma_i ** 2 * delta / (2.0 * time)
            x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
            x_t_std = jnp.sqrt(delta) * sigma_i
            z = jax.random.normal(step_rng, x_t.shape)
            x_next = x_t_mean + x_t_std * z
            return (x_next, time + dt, next_rng)

        def cond_fn(carry):
            _, time, _ = carry
            return time >= -dt / 2
        x_0, _, _ = jax.lax.while_loop(cond_fn, step_fn, (noise, 1.0, rng_loop))
        return x_0
