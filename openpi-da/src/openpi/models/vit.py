from collections.abc import Callable
from typing import Any
import flax.linen as nn
import jax
import jax.numpy as jnp
from openpi.models import resnet as models_resnet
Array = Any
PRNGKey = Any
Shape = tuple[int]
Dtype = Any

class IdentityLayer(nn.Module):

    @nn.compact
    def __call__(self, x):
        return x

class AddPositionEmbs(nn.Module):
    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        assert inputs.ndim == 3, f'Number of dimensions should be 3, but it is: {inputs.ndim}'
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape, self.param_dtype)
        return inputs + pe

class MlpBlock(nn.Module):
    mlp_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    out_dim: int | None = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-06)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.mlp_dim, dtype=self.dtype, param_dtype=self.param_dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(features=actual_out_dim, dtype=self.dtype, param_dtype=self.param_dtype, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)

class Encoder1DBlock(nn.Module):
    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, deterministic):
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(dtype=self.dtype, kernel_init=nn.initializers.xavier_uniform(), broadcast_dropout=False, deterministic=deterministic, dropout_rate=self.attention_dropout_rate, num_heads=self.num_heads, force_fp32_for_softmax=True)(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(y, deterministic=deterministic)
        return (x + y, None)

class Encoder(nn.Module):
    dtype: jax.typing.DTypeLike
    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = True

    @nn.compact
    def __call__(self, x, *, train):
        assert x.ndim == 3
        if self.add_position_embedding:
            x = AddPositionEmbs(posemb_init=nn.initializers.normal(stddev=0.02), name='posembed_input')(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = x.astype(self.dtype)
        block = nn.remat(Encoder1DBlock, prevent_cse=False, static_argnums=(2,))
        x, _ = nn.scan(block, variable_axes={'params': 0}, split_rngs={'params': True, 'dropout': True}, in_axes=nn.broadcast, length=self.num_layers)(name='encoderblock', mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate, attention_dropout_rate=self.attention_dropout_rate, dtype=self.dtype, num_heads=self.num_heads)(x, not train)
        return nn.LayerNorm(name='encoder_norm', dtype=self.dtype)(x)

class VisionTransformer(nn.Module):
    dtype: jax.typing.DTypeLike
    num_classes: int
    patches: Any
    transformer: Any
    hidden_size: int
    resnet: Any | None = None
    representation_size: int | None = None
    classifier: str = 'token'
    head_bias_init: float = 0.0
    encoder: type[nn.Module] = Encoder
    model_name: str | None = None

    @nn.compact
    def __call__(self, inputs, *, train):
        x = inputs
        if self.resnet is not None:
            width = int(64 * self.resnet.width_factor)
            x = models_resnet.StdConv(features=width, kernel_size=(7, 7), strides=(2, 2), use_bias=False, name='conv_root')(x)
            x = nn.GroupNorm(name='gn_root')(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
            if self.resnet.num_layers:
                x = models_resnet.ResNetStage(block_size=self.resnet.num_layers[0], nout=width, first_stride=(1, 1), name='block1')(x)
                for i, block_size in enumerate(self.resnet.num_layers[1:], 1):
                    x = models_resnet.ResNetStage(block_size=block_size, nout=width * 2 ** i, first_stride=(2, 2), name=f'block{i + 1}')(x)
        n, h, w, c = x.shape
        x = nn.Conv(features=self.hidden_size, kernel_size=self.patches.size, strides=self.patches.size, padding='VALID', name='embedding')(x)
        if self.transformer is not None:
            n, h, w, c = x.shape
            x = jnp.reshape(x, [n, h * w, c])
            if self.classifier in ['token', 'token_unpooled']:
                cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
                cls = jnp.tile(cls, [n, 1, 1])
                x = jnp.concatenate([cls, x], axis=1)
            x = self.encoder(name='Transformer', **self.transformer, dtype=self.dtype)(x, train=train)
        if self.classifier == 'token':
            x = x[:, 0]
        elif self.classifier == 'gap':
            x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))
        elif self.classifier in ['unpooled', 'token_unpooled']:
            pass
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')
        if self.representation_size is not None:
            x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
            x = nn.tanh(x)
        else:
            x = IdentityLayer(name='pre_logits')(x)
        if self.num_classes:
            x = nn.Dense(features=self.num_classes, name='head', kernel_init=nn.initializers.zeros, bias_init=nn.initializers.constant(self.head_bias_init))(x)
        return x
