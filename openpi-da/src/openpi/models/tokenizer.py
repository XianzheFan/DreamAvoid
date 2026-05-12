import logging
import os
import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor
import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download

class PaligemmaTokenizer:

    def __init__(self, max_len: int=48):
        self._max_len = max_len
        path = download.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'})
        with path.open('rb') as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace('_', ' ').replace('\n', ' ')
        if state is not None:
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = ' '.join(map(str, discretized_state))
            full_prompt = f'Task: {cleaned_text}, State: {state_str};\nAction: '
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode('\n')
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. Consider increasing the `max_token_len` in your model config if this happens frequently.')
            tokens = tokens[:self._max_len]
            mask = [True] * self._max_len
        return (np.asarray(tokens), np.asarray(mask))

class FASTTokenizer:

    def __init__(self, max_len: int=256, fast_tokenizer_path: str='physical-intelligence/fast'):
        self._max_len = max_len
        path = download.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'})
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128

    def tokenize(self, prompt: str, state: np.ndarray, actions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace('_', ' ')
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
        if actions is not None:
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
            postfix_tokens = self._paligemma_tokenizer.encode('Action: ') + action_tokens_in_pg.tolist() + self._paligemma_tokenizer.encode('|', add_eos=True)
        else:
            postfix_tokens = []
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. Consider increasing the `max_token_len` in your model config if this happens frequently.')
            tokens = tokens[:self._max_len]
            token_mask = token_mask[:self._max_len]
            ar_mask = ar_mask[:self._max_len]
            loss_mask = loss_mask[:self._max_len]
        return (np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask))

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
        if 'Action: ' not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        raw_action_tokens = np.array(self._paligemma_tokenizer.encode(decoded_tokens.split('Action: ')[1].split('|')[0].strip()))
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode([action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim)[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

class BinningTokenizer:

    def __init__(self, max_len: int=256, n_bins: int=256):
        self._max_len = max_len
        self._n_bins = n_bins
        path = download.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'})
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._fast_skip_tokens = 128

    def tokenize(self, prompt: str, state: np.ndarray, actions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace('_', ' ')
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
        if actions is not None:
            raise NotImplementedError('BinningTokenizer does not support encoding actions atm (only for inference use)')
        postfix_tokens = []
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. Consider increasing the `max_token_len` in your model config if this happens frequently.')
            tokens = tokens[:self._max_len]
            token_mask = token_mask[:self._max_len]
            ar_mask = ar_mask[:self._max_len]
            loss_mask = loss_mask[:self._max_len]
        return (np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask))

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
        if 'Action: ' not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        raw_action_tokens = np.array(self._paligemma_tokenizer.encode(decoded_tokens.split('Action: ')[1].split('|')[0].strip()))
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[:action_horizon * action_dim].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

class FSQTokenizer:

    def __init__(self, max_len: int=256, fsq_tokenizer_path: str | None=None):
        self._max_len = max_len
        assert fsq_tokenizer_path is not None, 'fsq_tokenizer_path must be provided'
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])
        step = int(tok_path.split('/')[-1])
        base_path = tok_path.rsplit('/', 1)[0]
        mgr = ocp.CheckpointManager(base_path, item_handlers={'params': ocp.StandardCheckpointHandler(), 'opt_state': ocp.StandardCheckpointHandler(), 'config': ocp.JsonCheckpointHandler()}, options=ocp.CheckpointManagerOptions(max_to_keep=1))
        try:
            restored = mgr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore()))
            config = restored['config']
            self._params = restored['params']
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(f'Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}') from e
        self._tokenize_fn = jax.jit(lambda params, x: self._fsq_tokenizer.apply({'params': params}, x, method=self._fsq_tokenizer.tokenize))
        self._detokenize_fn = jax.jit(lambda params, x: self._fsq_tokenizer.apply({'params': params}, x, method=self._fsq_tokenizer.detokenize))
        path = download.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'})
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._fast_skip_tokens = 128

    def tokenize(self, prompt: str, state: np.ndarray, actions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace('_', ' ')
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
        if actions is not None:
            raise NotImplementedError('FSQTokenizer does not support encoding actions atm (only for inference use)')
        postfix_tokens = []
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. Consider increasing the `max_token_len` in your model config if this happens frequently.')
            tokens = tokens[:self._max_len]
            token_mask = token_mask[:self._max_len]
            ar_mask = ar_mask[:self._max_len]
            loss_mask = loss_mask[:self._max_len]
        return (np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask))

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
        if 'Action: ' not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        raw_action_tokens = np.array(self._paligemma_tokenizer.encode(decoded_tokens.split('Action: ')[1].split('|')[0].strip()))
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            device = jax.devices('cpu')[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[:action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f'Error decoding FSQ: {e}')
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
