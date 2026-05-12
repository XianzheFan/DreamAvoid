import logging
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import optax
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

class Pi0(_model.BaseModel):

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
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
        self.switch_head_enabled = config.switch_head
        if config.switch_head:
            self.switch_head_proj1 = nnx.Linear(paligemma_config.width, 256, rngs=rngs)
            self.switch_head_proj2 = nnx.Linear(256, 1, rngs=rngs)
            self.switch_loss_weight = config.switch_loss_weight
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

    def _compute_switch_logit(self, prefix_out: at.Float[at.Array, 'b s emb'], prefix_mask: at.Bool[at.Array, 'b s']) -> at.Float[at.Array, 'b 1']:
        mask_expanded = prefix_mask[..., None].astype(prefix_out.dtype)
        pooled = jnp.sum(prefix_out * mask_expanded, axis=1) / jnp.maximum(jnp.sum(mask_expanded, axis=1), 1.0)
        h = nnx.relu(self.switch_head_proj1(pooled))
        return self.switch_head_proj2(h)

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
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if self.switch_head_enabled and observation.switch_label is not None:
            switch_logit = self._compute_switch_logit(prefix_out, prefix_mask)
            switch_logit = switch_logit.squeeze(-1)
            switch_bce = optax.sigmoid_binary_cross_entropy(switch_logit, observation.switch_label)
            flow_loss = flow_loss + self.switch_loss_weight * switch_bce[..., None]
        return flow_loss

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, *, num_steps: int | at.Int[at.Array, '']=10, noise: at.Float[at.Array, 'b ah ad'] | None=None, return_switch: bool=False, mode: str='ode', noise_level: float=0.3) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        rng_init, rng_loop = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(rng_init, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out_cached, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def forward_vt(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, 'b p -> b s p', s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1] + suffix_tokens.shape[1])
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon:])

        def ode_step(carry):
            x_t, time, current_rng = carry
            v_t = forward_vt(x_t, time)
            return (x_t + dt * v_t, time + dt, current_rng)

        def sde_step(carry):
            x_t, time, current_rng = carry
            step_rng, next_rng = jax.random.split(current_rng)
            v_t = forward_vt(x_t, time)
            delta = jnp.abs(dt)
            denom = jnp.where(time >= 1.0 - 0.0001, delta, 1.0 - time)
            sigma_i = noise_level * jnp.sqrt(time / denom)
            x0_pred = x_t - v_t * time
            x1_pred = x_t + v_t * (1.0 - time)
            x0_weight = 1.0 - (time - delta)
            x1_weight = time - delta - sigma_i ** 2 * delta / (2.0 * time)
            x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
            x_t_std = jnp.sqrt(delta) * sigma_i
            z = jax.random.normal(step_rng, x_t.shape)
            return (x_t_mean + x_t_std * z, time + dt, next_rng)

        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2
        step_fn = sde_step if mode == 'sde' else ode_step
        x_0, _, _ = jax.lax.while_loop(cond, step_fn, (noise, 1.0, rng_loop))
        if return_switch and self.switch_head_enabled:
            switch_logit = self._compute_switch_logit(prefix_out_cached, prefix_mask)
            switch_prob = jax.nn.sigmoid(switch_logit.squeeze(-1))
            return (x_0, switch_prob)
        return x_0
