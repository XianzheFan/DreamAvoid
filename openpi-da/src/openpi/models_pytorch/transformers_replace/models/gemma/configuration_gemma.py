from typing import Optional
from ...configuration_utils import PretrainedConfig

class GemmaConfig(PretrainedConfig):
    model_type = 'gemma'
    keys_to_ignore_at_inference = ['past_key_values']
    base_model_tp_plan = {'layers.*.self_attn.q_proj': 'colwise', 'layers.*.self_attn.k_proj': 'colwise', 'layers.*.self_attn.v_proj': 'colwise', 'layers.*.self_attn.o_proj': 'rowwise', 'layers.*.mlp.gate_proj': 'colwise', 'layers.*.mlp.up_proj': 'colwise', 'layers.*.mlp.down_proj': 'rowwise'}
    base_model_pp_plan = {'embed_tokens': (['input_ids'], ['inputs_embeds']), 'layers': (['hidden_states', 'attention_mask'], ['hidden_states']), 'norm': (['hidden_states'], ['hidden_states'])}

    def __init__(self, vocab_size=256000, hidden_size=3072, intermediate_size=24576, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=16, head_dim=256, hidden_act='gelu_pytorch_tanh', hidden_activation=None, max_position_embeddings=8192, initializer_range=0.02, rms_norm_eps=1e-06, use_cache=True, pad_token_id=0, eos_token_id=1, bos_token_id=2, tie_word_embeddings=True, rope_theta=10000.0, attention_bias=False, attention_dropout=0.0, use_adarms: bool=False, adarms_cond_dim: Optional[int]=None, **kwargs):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_adarms = use_adarms
        self.adarms_cond_dim = adarms_cond_dim
        if self.use_adarms and self.adarms_cond_dim is None:
            self.adarms_cond_dim = self.hidden_size
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)
__all__ = ['GemmaConfig']
