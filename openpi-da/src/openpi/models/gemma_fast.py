import dataclasses
from typing import Literal, TypeAlias
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import openpi.models.lora as lora
import openpi.shared.array_typing as at
Variant = Literal['gemma_2b', 'gemma_2b_lora']

def get_config(variant):
    if variant == 'gemma_2b':
        return ml_collections.ConfigDict({'variant': variant, 'width': 2048, 'depth': 18, 'mlp_dim': 16384, 'num_heads': 8, 'num_kv_heads': 1, 'head_dim': 256, 'norm_eps': 1e-06, 'vocab_size': 257152, 'scan': True, 'remat_policy': 'nothing_saveable'})
    if variant == 'gemma_2b_lora':
        return ml_collections.ConfigDict({'variant': variant, 'width': 2048, 'depth': 18, 'mlp_dim': 16384, 'num_heads': 8, 'num_kv_heads': 1, 'head_dim': 256, 'norm_eps': 1e-06, 'vocab_size': 257152, 'scan': True, 'remat_policy': 'nothing_saveable', 'lora_configs': {'attn': lora.LoRAConfig(rank=16, alpha=16.0), 'ffn': lora.LoRAConfig(rank=16, alpha=16.0)}})
    raise ValueError(f'Unknown variant: {variant}')

@at.typecheck
class Einsum(nn.Module):
    shape: tuple[int, ...]

    @nn.compact
    def __call__(self, eqn, x):
        dtype = x.dtype
        w = self.param('w', nn.initializers.zeros_init(), self.shape).astype(dtype)
        return jnp.einsum(eqn, x, w)

@at.typecheck
class RMSNorm(nn.Module):

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        scale = self.param('scale', nn.initializers.zeros_init(), x.shape[-1])
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs.astype(dtype)

@at.typecheck
class Embedder(nn.Module):
    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param('input_embedding', nn.initializers.zeros_init(), (self.vocab_size, self.embed_dim))

    def encode(self, x):
        x = self.input_embedding_table[x,]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)

@at.typecheck
class Attention(nn.Module):
    num_heads: int
    num_kv_heads: int
    features: int
    head_dim: int
    cache_dtype: str | None = None
    lora_config: lora.LoRAConfig | None = None

    def setup(self):
        if self.num_kv_heads == self.num_heads:
            self.qkv_einsum = lora.Einsum(shape=(3, self.num_heads, self.features, self.head_dim), name='qkv_einsum', init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)), lora_config=self.lora_config)
        else:
            self.q_einsum = lora.Einsum(shape=(self.num_heads, self.features, self.head_dim), name='q_einsum', init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)), lora_config=self.lora_config)
            self.kv_einsum = lora.Einsum(shape=(2, self.num_kv_heads, self.features, self.head_dim), name='kv_einsum', init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)), lora_config=self.lora_config)
        self.attn_vec_einsum = lora.Einsum(shape=(self.num_heads, self.head_dim, self.features), name='attn_vec_einsum', init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)), lora_config=self.lora_config)

    def _init_cache(self, k, v, cache_size):
        prefill_len = k.shape[1]
        pad_width = ((0, 0), (0, cache_size - prefill_len), (0, 0), (0, 0))
        cache_dtype = self.cache_dtype or k.dtype
        k_cache = jnp.pad(k.astype(cache_dtype), pad_width)
        v_cache = jnp.pad(v.astype(cache_dtype), pad_width)
        idx = jnp.zeros((k.shape[0],), dtype=jnp.int32) + prefill_len
        return (idx, k_cache, v_cache)

    def _update_cache(self, k, v, idx, k_cache, v_cache):
        assert k.shape[1] == 1, 'Only support kv-cache updates of length 1'
        indices = (0, idx[0], 0, 0)
        cache_dtype = self.cache_dtype or k.dtype
        k_new = jax.lax.dynamic_update_slice(k_cache, k.astype(cache_dtype), indices)
        v_new = jax.lax.dynamic_update_slice(v_cache, v.astype(cache_dtype), indices)
        idx_new = idx + 1
        return (idx_new, k_new, v_new)

    @nn.compact
    def __call__(self, x, positions, attn_mask, kv_cache, decode, deterministic=True):
        dtype = x.dtype
        if self.num_kv_heads == self.num_heads:
            q, k, v = self.qkv_einsum('BSD,3KDH->3BSKH', x)
        else:
            q = self.q_einsum('BTD,NDH->BTNH', x)
            k, v = self.kv_einsum('BSD,2KDH->2BSKH', x)
        q = _apply_rope(q, positions=positions)
        q *= self.head_dim ** (-0.5)
        k = _apply_rope(k, positions=positions)
        if kv_cache is None:
            idx, k_cache, v_cache = self._init_cache(k, v, attn_mask.shape[-1])
        else:
            idx, k_cache, v_cache = kv_cache
            idx, k_cache, v_cache = self._update_cache(k, v, idx, k_cache, v_cache)
        k, v = (k_cache, v_cache)
        kv_cache = (idx, k_cache, v_cache)
        q = einops.rearrange(q, 'B T (K G) H -> B T K G H', K=self.num_kv_heads)
        logits = jnp.einsum('BTKGH,BSKH->BKGTS', q, k, preferred_element_type=jnp.float32)
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(f'Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}')
        big_neg = -2.3819763e+38
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
        encoded = jnp.einsum('BKGTS,BSKH->BTKGH', probs, v)
        encoded = einops.rearrange(encoded, 'B T K G H -> B T (K G) H')
        return (self.attn_vec_einsum('BTNH,NHD->BTD', encoded), kv_cache)

@at.typecheck
class Block(nn.Module):
    num_heads: int
    num_kv_heads: int
    embed_dim: int
    head_dim: int
    hidden_dim: int
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = None
    lora_configs: ml_collections.ConfigDict = dataclasses.field(default_factory=ml_collections.ConfigDict)

    def setup(self):
        self.pre_attention_norm = RMSNorm()
        self.attn = Attention(num_heads=self.num_heads, num_kv_heads=self.num_kv_heads, features=self.embed_dim, head_dim=self.head_dim, cache_dtype=self.cache_dtype, lora_config=self.lora_configs.get('attn'))
        self.pre_ffw_norm = RMSNorm()
        self.mlp = lora.FeedForward(features=self.embed_dim, hidden_dim=self.hidden_dim, name='mlp', lora_config=self.lora_configs.get('ffn'))
        if self.dropout:
            self.drop = nn.Dropout(self.dropout, self.dropout_bdims)
        else:
            self.drop = lambda x, _: x

    def __call__(self, x, kv_cache, positions, attn_mask, decode, deterministic=True):
        x = nn.with_logical_constraint(x, ('act_batch', 'act_len', 'act_emb'))
        inputs_normalized = self.pre_attention_norm(x)
        attn_output, kv_cache = self.attn(inputs_normalized, positions, attn_mask, kv_cache, decode, deterministic)
        attn_output = self.drop(attn_output, deterministic)
        attn_output += x
        residual = attn_output
        attn_output = self.pre_ffw_norm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = self.drop(outputs, deterministic)
        outputs = residual + outputs
        return (outputs, kv_cache)
KVCache: TypeAlias = tuple[at.Int[at.Array, ' b'], at.Float[at.Array, 'b _t _k _h'], at.Float[at.Array, 'b _t _v _h']]

@at.typecheck
class Module(nn.Module):
    variant: str
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    norm_eps: float
    vocab_size: int
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = None
    scan: bool = False
    remat_policy: str = 'none'
    lora_configs: ml_collections.ConfigDict = dataclasses.field(default_factory=ml_collections.ConfigDict)

    @nn.compact
    def __call__(self, tokens=None, embedded_prefix=None, embed_only=False, pre_logits=None, positions=None, mask=None, decode=False, kv_cache=None, deterministic=True, return_prelogits=False):
        out = {}
        embedder = Embedder(vocab_size=self.vocab_size, embed_dim=self.width, name='embedder')
        if pre_logits is not None:
            x = out['pre_logits'] = pre_logits
            logits = out['logits'] = embedder.decode(x)
            return (logits, out)
        x = []
        if embedded_prefix is not None:
            x.append(embedded_prefix)
        if tokens is not None:
            x.append(embedder.encode(tokens))
        x = jnp.concatenate(x, axis=-2)
        x = x.astype(self.embed_dtype)
        batch_size, seq_len, width = x.shape
        if embed_only:
            return x
        if decode:
            assert positions is not None and mask is not None, 'Must explicitly pass positions and mask for decoding.'
        if positions is None:
            positions = jnp.arange(seq_len).astype(jnp.int32)[None, :]
        assert positions.shape[1] == x.shape[1], (positions.shape, x.shape)
        if mask is None:
            mask = nn.attention.make_causal_mask(jnp.ones([batch_size, seq_len]))
        if mask.ndim == 3:
            mask = mask[:, None, :, :]
        cache_size = max(seq_len, mask.shape[-1])
        assert mask.shape == (batch_size, 1, seq_len, cache_size), mask.shape
        if self.remat_policy == 'none':
            block_cls = Block
        else:
            block_cls = nn.remat(Block, prevent_cse=not self.scan, static_argnums=(5, 6), policy=getattr(jax.checkpoint_policies, self.remat_policy))
        block_kw = {'num_heads': self.num_heads, 'head_dim': self.head_dim, 'num_kv_heads': self.num_kv_heads, 'embed_dim': width, 'hidden_dim': self.mlp_dim, 'dropout': self.dropout, 'dropout_bdims': self.dropout_bdims, 'cache_dtype': self.cache_dtype, 'lora_configs': self.lora_configs}
        layers = self.scope.push('layers')
        blocks = [nn.scan(block_cls, variable_axes={'params': 0}, split_rngs={'params': True, 'dropout': True}, in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast), length=self.depth)(parent=layers, **block_kw)]
        for block in blocks:
            x, kv_cache = block(x, kv_cache, positions, mask, decode, deterministic)
        assert x.dtype == jnp.dtype(self.embed_dtype)
        out['encoded'] = x
        x = RMSNorm(name='final_norm')(x)
        out['pre_logits'] = x
        if return_prelogits:
            return (x, kv_cache, out)
        x = embedder.decode(x)
        out['logits'] = x
        return (x, kv_cache, out)

    def init(self):
        self(jnp.zeros((1, 1), dtype=jnp.int32))

def _apply_rope(x, *, positions, max_wavelength=10000):
    freq_exponents = 2.0 / x.shape[-1] * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    sin, cos = (jnp.sin(radians), jnp.cos(radians))
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    return res
