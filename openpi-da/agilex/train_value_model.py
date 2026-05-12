from __future__ import annotations
import argparse
import glob
import json
import logging
import os
import pathlib
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
try:
    import wandb
except ImportError:
    wandb = None
from train_dinov2_value_expert import DEFAULT_SAVE_H, DEFAULT_SAVE_W, REAL_CAMERAS, _cleanup_distributed, _get_video_path, _load_dataset_info, _load_task, _setup_distributed, extract_frames_pyav, resize_stitched, stitch_3view
from train_switch_head_robometer import load_episode_parquet
BUCKET_SUCCESS = 'success'
BUCKET_STEADY = 'steady'
BUCKET_STAGNATION = 'stagnation'
BUCKET_BOUNDARY = 'boundary'
BUCKET_FAILURE = 'failure'
ALL_BUCKETS = (BUCKET_SUCCESS, BUCKET_STEADY, BUCKET_STAGNATION, BUCKET_BOUNDARY, BUCKET_FAILURE)
DEFAULT_BUCKET_TARGET = {BUCKET_SUCCESS: 0.2, BUCKET_STEADY: 0.4, BUCKET_STAGNATION: 0.0, BUCKET_BOUNDARY: 0.2, BUCKET_FAILURE: 0.2}

def _bucket_from_z(z: float, stagnation_eps: float=0.005) -> str:
    if z >= 0.95:
        return BUCKET_SUCCESS
    if z <= -0.95:
        return BUCKET_FAILURE
    if z > stagnation_eps:
        return BUCKET_STEADY
    if z < -stagnation_eps:
        return BUCKET_BOUNDARY
    return BUCKET_STAGNATION

def compute_value_label(eta_start: float, progress_clip: np.ndarray, success_clip: np.ndarray, is_terminal: bool, success_threshold: float=0.5, success_drop_threshold: float=0.3, failure_progress_threshold: float=0.05) -> float:
    progress_clip = np.asarray(progress_clip, dtype=np.float32)
    success_clip = np.asarray(success_clip, dtype=np.float32)
    if len(progress_clip) == 0:
        return 0.0
    eta_end = float(progress_clip[-1])
    final_success = float(success_clip[-1]) if success_clip.size else 0.0
    if is_terminal and final_success >= success_threshold:
        return 1.0
    if success_clip.size > 0:
        peak_success = float(success_clip.max())
        if peak_success > 0.0 and peak_success - final_success >= success_drop_threshold:
            return -1.0
    if is_terminal and eta_end < failure_progress_threshold:
        return -1.0
    return float(np.clip(eta_end - float(eta_start), -1.0, 1.0))

def _stitch_episode(data_dir: pathlib.Path, ep_idx: int, extract_fps: float, max_frames: int, save_h: int, save_w: int, view_layout: str) -> tuple[np.ndarray | None, int]:
    cam_frames = {}
    for cam in REAL_CAMERAS:
        vp = _get_video_path(data_dir, ep_idx, cam)
        if not vp.exists():
            logging.warning(f'  ep {ep_idx}: missing {cam}, skip')
            return (None, 0)
        frames = extract_frames_pyav(str(vp), fps=extract_fps, max_frames=max_frames)
        if frames.size == 0:
            logging.warning(f'  ep {ep_idx}: failed to decode {cam}, skip')
            return (None, 0)
        cam_frames[cam] = frames
    T = min((f.shape[0] for f in cam_frames.values()))
    for cam in cam_frames:
        cam_frames[cam] = cam_frames[cam][:T]
    stitched = stitch_3view(cam_frames['cam_high'], cam_frames['cam_left_wrist'], cam_frames['cam_right_wrist'], layout=view_layout, target_h=save_h, target_w_per_view=save_w // 3)
    return (stitched, T)

def _save_window(out_dir: pathlib.Path, step_i: int, obs_frame: np.ndarray, future_frames: np.ndarray, action_chunk: np.ndarray, state: np.ndarray, z: float, bucket: str, source: str, task: str, episode_idx: int, frame_idx_raw: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f'step_{step_i:05d}.npz', obs_frame=obs_frame.astype(np.uint8), future_frames=future_frames.astype(np.uint8), action_chunk=action_chunk.astype(np.float32), state=state.astype(np.float32), z=np.float32(z), bucket=np.array(bucket), source=np.array(source), task=np.array(task), episode_idx=np.int32(episode_idx), frame_idx=np.int32(frame_idx_raw))

def _detect_rewards_fps(rewards_dir: pathlib.Path, default: float=3.0) -> float:
    summary = rewards_dir / 'summary.json'
    if summary.exists():
        try:
            with open(summary) as f:
                return float(json.load(f).get('fps', default))
        except Exception:
            pass
    return default

def _resample_to_extract_fps(arr: np.ndarray, src_fps: float, dst_fps: float, target_len: int) -> np.ndarray:
    if len(arr) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if abs(src_fps - dst_fps) < 0.001:
        return arr[:target_len].astype(np.float32)
    src_idx = (np.arange(target_len) * (src_fps / dst_fps)).astype(np.int64)
    src_idx = np.clip(src_idx, 0, len(arr) - 1)
    return arr[src_idx].astype(np.float32)

def _pack_real_episodes(args, output_dir: pathlib.Path, stats: dict) -> dict[tuple[int, int], dict]:
    data_dir = pathlib.Path(args.data_dir)
    rewards_dir = pathlib.Path(args.rewards_dir) if args.rewards_dir else data_dir / 'rewards'
    info = _load_dataset_info(data_dir)
    task = _load_task(data_dir)
    dataset_fps = info['fps']
    rewards_fps = float(args.rewards_fps) if args.rewards_fps else _detect_rewards_fps(rewards_dir)
    available_eps = sorted({int(f.stem.split('_')[1]) for f in rewards_dir.glob('episode_*_progress.npy')})
    if args.episodes:
        ep_start, ep_end = map(int, args.episodes.split(':'))
        available_eps = [e for e in available_eps if ep_start <= e < ep_end]
    logging.info(f'[real] Dataset:    {data_dir}')
    logging.info(f'[real] Rewards:    {rewards_dir}  (fps={rewards_fps})')
    logging.info(f'[real] Task:       {task}')
    logging.info(f'[real] Episodes:   {len(available_eps)}')
    logging.info(f'[real] L_future={args.num_future_frames}, K_action={args.action_chunk}, extract_fps={args.extract_fps} (raw_fps={dataset_fps}, rewards_fps={rewards_fps})')
    task_segment = task.replace(' ', '_')[:50]
    raw_per_extracted = max(1, int(round(dataset_fps / args.extract_fps)))
    L_future = int(args.num_future_frames)
    K_action = int(args.action_chunk)
    real_index: dict[tuple[int, int], dict] = {}
    t_start = time.time()
    for ep_pos, ep_idx in enumerate(available_eps):
        rollout_id = f'rollout_{task_segment}_real_ep{ep_idx}_done'
        rollout_dir = output_dir / rollout_id
        already_packed = rollout_dir.exists() and any(rollout_dir.glob('step_*.npz'))
        if already_packed and (not args.overwrite):
            _populate_real_index_from_disk(rollout_dir, real_index)
            logging.info(f'  [real] ep {ep_idx}: already packed, indexed for dreams')
            continue
        progress = np.load(rewards_dir / f'episode_{ep_idx:06d}_progress.npy').astype(np.float32)
        success_path = rewards_dir / f'episode_{ep_idx:06d}_success.npy'
        success = np.load(success_path).astype(np.float32) if success_path.exists() else np.zeros_like(progress)
        stitched, T_cam = _stitch_episode(data_dir, ep_idx, args.extract_fps, args.max_frames, args.save_h, args.save_w, args.view_layout)
        if stitched is None:
            continue
        prog_sub = _resample_to_extract_fps(progress, src_fps=rewards_fps, dst_fps=args.extract_fps, target_len=T_cam)
        succ_sub = _resample_to_extract_fps(success, src_fps=rewards_fps, dst_fps=args.extract_fps, target_len=T_cam)
        T = min(T_cam, len(prog_sub), len(succ_sub))
        stitched = stitched[:T]
        prog_sub = prog_sub[:T]
        succ_sub = succ_sub[:T]
        states, actions, _ = load_episode_parquet(data_dir, ep_idx)
        T_raw = len(actions)
        n_windows = max(0, T - L_future)
        if n_windows < 1:
            logging.warning(f'  [real] ep {ep_idx}: T={T} < L_future+1, skip')
            continue
        ep_buckets = {b: 0 for b in ALL_BUCKETS}
        step_i = 0
        for t_ext in range(0, n_windows, args.window_stride):
            t_raw = t_ext * raw_per_extracted
            if t_raw >= T_raw:
                break
            future_slice_ext = slice(t_ext + 1, t_ext + 1 + L_future)
            prog_clip = prog_sub[future_slice_ext]
            succ_clip = succ_sub[future_slice_ext]
            if len(prog_clip) < L_future:
                break
            is_terminal = t_ext + L_future >= T - 1
            z = compute_value_label(eta_start=float(prog_sub[t_ext]), progress_clip=prog_clip, success_clip=succ_clip, is_terminal=is_terminal, success_threshold=args.success_threshold, success_drop_threshold=args.success_drop_threshold, failure_progress_threshold=args.failure_progress_threshold)
            bucket = _bucket_from_z(z)
            obs_frame = stitched[t_ext]
            future_frames = stitched[future_slice_ext]
            end_raw = min(t_raw + K_action, T_raw)
            chunk_actual = actions[t_raw:end_raw]
            if len(chunk_actual) < K_action:
                pad = np.tile(chunk_actual[-1:], (K_action - len(chunk_actual), 1)) if len(chunk_actual) > 0 else np.zeros((K_action, actions.shape[1]), dtype=np.float32)
                chunk_actual = np.concatenate([chunk_actual, pad], axis=0)
            state = states[t_raw] if t_raw < len(states) else np.zeros(states.shape[1], dtype=np.float32)
            _save_window(rollout_dir, step_i, obs_frame=obs_frame, future_frames=future_frames, action_chunk=chunk_actual, state=state, z=z, bucket=bucket, source='real', task=task, episode_idx=ep_idx, frame_idx_raw=t_raw)
            real_index[ep_idx, t_raw] = {'z': z, 'bucket': bucket, 'task': task}
            ep_buckets[bucket] += 1
            stats['real_clips'] += 1
            stats['bucket_real'][bucket] = stats['bucket_real'].get(bucket, 0) + 1
            step_i += 1
        if step_i > 0:
            stats['real_rollouts'] += 1
        elapsed = time.time() - t_start
        eta = (len(available_eps) - ep_pos - 1) * elapsed / max(ep_pos + 1, 1)
        logging.info(f'  [real] ep {ep_idx}: {step_i} clips | buckets={ep_buckets} | prog=[{prog_sub.min():.2f},{prog_sub.max():.2f}] succ_max={succ_sub.max():.2f} | ETA {eta:.0f}s')
    return real_index

def _populate_real_index_from_disk(rollout_dir: pathlib.Path, real_index: dict[tuple[int, int], dict]):
    for f in sorted(rollout_dir.glob('step_*.npz')):
        try:
            d = np.load(f, allow_pickle=True)
            ep = int(d['episode_idx'])
            fr = int(d['frame_idx'])
            real_index[ep, fr] = {'z': float(d['z']), 'bucket': str(d['bucket'].item()), 'task': str(d['task'].item())}
        except Exception:
            pass

def _pack_dream_entries(args, output_dir: pathlib.Path, stats: dict, real_index: dict[tuple[int, int], dict]):
    if not args.dreamdojo_metadata:
        return
    meta_path = pathlib.Path(args.dreamdojo_metadata)
    with open(meta_path) as f:
        meta_json = json.load(f)
    entries = meta_json['entries'] if isinstance(meta_json, dict) else meta_json
    logging.info(f'[dream] Metadata: {meta_path}  (entries={len(entries)})')
    raw_per_extracted = max(1, int(round(_load_dataset_info(pathlib.Path(args.data_dir))['fps'] / args.extract_fps)))
    L_future = int(args.num_future_frames)
    n_inherited = n_self_labeled = n_skipped = 0
    for ent_i, ent in enumerate(entries):
        video_path = ent['video']
        src_ep = int(ent.get('source_episode', -1))
        src_frame_raw = int(ent.get('source_frame', -1))
        task = ent.get('task', 'dream')
        if not os.path.exists(video_path):
            logging.warning(f'  [dream] missing video: {video_path}, skip')
            n_skipped += 1
            continue
        stitched_raw = extract_frames_pyav(video_path, fps=args.extract_fps, max_frames=args.max_frames)
        if stitched_raw.size == 0 or stitched_raw.shape[0] < L_future:
            n_skipped += 1
            continue
        stitched = resize_stitched(stitched_raw, args.save_h, args.save_w)
        future_frames = stitched[1:1 + L_future]
        if len(future_frames) < L_future:
            n_skipped += 1
            continue
        obs_frame = stitched[0]
        match = real_index.get((src_ep, src_frame_raw))
        if match is None and src_ep >= 0 and (src_frame_raw >= 0):
            for delta in range(1, raw_per_extracted + 1):
                for cand in (src_frame_raw - delta, src_frame_raw + delta):
                    if (src_ep, cand) in real_index:
                        match = real_index[src_ep, cand]
                        src_frame_raw = cand
                        break
                if match is not None:
                    break
        if match is not None:
            z = float(match['z'])
            bucket = match['bucket']
            n_inherited += 1
        elif 'progress' in ent:
            prog_clip = np.asarray(ent['progress'][1:1 + L_future], dtype=np.float32)
            succ_clip = np.asarray(ent.get('success', [0.0] * L_future)[1:1 + L_future], dtype=np.float32)
            eta_start = float(ent['progress'][0]) if len(ent['progress']) > 0 else 0.0
            z = compute_value_label(eta_start=eta_start, progress_clip=prog_clip, success_clip=succ_clip, is_terminal=False, success_threshold=args.success_threshold, success_drop_threshold=args.success_drop_threshold, failure_progress_threshold=args.failure_progress_threshold)
            bucket = _bucket_from_z(z)
            n_self_labeled += 1
        else:
            n_skipped += 1
            continue
        if src_ep >= 0 and src_frame_raw >= 0:
            try:
                states, actions, _ = load_episode_parquet(pathlib.Path(args.data_dir), src_ep)
                T_raw = len(actions)
                end_raw = min(src_frame_raw + args.action_chunk, T_raw)
                chunk = actions[src_frame_raw:end_raw]
                if len(chunk) < args.action_chunk and len(chunk) > 0:
                    pad = np.tile(chunk[-1:], (args.action_chunk - len(chunk), 1))
                    chunk = np.concatenate([chunk, pad], axis=0)
                elif len(chunk) == 0:
                    chunk = np.zeros((args.action_chunk, actions.shape[1]), dtype=np.float32)
                state = states[src_frame_raw] if src_frame_raw < len(states) else np.zeros(states.shape[1], dtype=np.float32)
            except Exception as exc:
                logging.warning(f'  [dream] failed to load actions for ep {src_ep}: {exc}')
                n_skipped += 1
                continue
        else:
            chunk = np.zeros((args.action_chunk, 14), dtype=np.float32)
            state = np.zeros(14, dtype=np.float32)
        task_segment = task.replace(' ', '_')[:50]
        basename = pathlib.Path(video_path).stem
        rollout_id = f'rollout_{task_segment}_dream_ep{src_ep}_fr{src_frame_raw}_{basename}'
        rollout_dir = output_dir / rollout_id
        _save_window(rollout_dir, 0, obs_frame=obs_frame, future_frames=future_frames, action_chunk=chunk, state=state, z=z, bucket=bucket, source='dream', task=task, episode_idx=src_ep, frame_idx_raw=src_frame_raw)
        stats['dream_rollouts'] += 1
        stats['dream_clips'] += 1
        stats['bucket_dream'][bucket] = stats['bucket_dream'].get(bucket, 0) + 1
        if (ent_i + 1) % 50 == 0:
            logging.info(f'  [dream] {ent_i + 1}/{len(entries)} packed | inherited={n_inherited} self={n_self_labeled} skipped={n_skipped}')
    logging.info(f'[dream] Done. inherited={n_inherited} self_labeled={n_self_labeled} skipped={n_skipped}')

def pack(args):
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {'real_rollouts': 0, 'real_clips': 0, 'dream_rollouts': 0, 'dream_clips': 0, 'bucket_real': {}, 'bucket_dream': {}}
    real_index = _pack_real_episodes(args, output_dir, stats)
    _pack_dream_entries(args, output_dir, stats, real_index)
    meta_path = output_dir / 'value_model_clips_meta.json'
    with open(meta_path, 'w') as f:
        json.dump({'extract_fps': args.extract_fps, 'num_future_frames': args.num_future_frames, 'action_chunk': args.action_chunk, 'window_stride': args.window_stride, 'save_h': args.save_h, 'save_w': args.save_w, 'view_layout': args.view_layout, 'success_threshold': args.success_threshold, 'success_drop_threshold': args.success_drop_threshold, 'failure_progress_threshold': args.failure_progress_threshold, 'stats': stats}, f, indent=2)
    logging.info(f'\nPack complete. real_clips={stats['real_clips']} dream_clips={stats['dream_clips']}\n  Buckets (real):  {stats['bucket_real']}\n  Buckets (dream): {stats['bucket_dream']}\n  Saved to:        {output_dir}')

class ValueModelDataset(Dataset):

    def __init__(self, data_dir: str, files: list[str] | None=None, task_to_id: dict[str, int] | None=None):
        if files is not None:
            self.files = sorted(files)
        else:
            self.files = sorted(glob.glob(os.path.join(data_dir, 'rollout_*', 'step_*.npz')))
        if not self.files:
            raise FileNotFoundError(f'No value-model npz files found in {data_dir}')
        first = np.load(self.files[0], allow_pickle=True)
        self.h, w_total, _ = first['obs_frame'].shape
        self.w = w_total
        self.num_future_frames = first['future_frames'].shape[0]
        self.action_chunk = first['action_chunk'].shape[0]
        self.action_dim = first['action_chunk'].shape[1]
        self.state_dim = first['state'].shape[0]
        self._buckets: list[str] = []
        self._zs: list[float] = []
        self._sources: list[str] = []
        self._tasks: list[str] = []
        for f in self.files:
            d = np.load(f, allow_pickle=True)
            self._buckets.append(str(d['bucket'].item()))
            self._zs.append(float(d['z']))
            self._sources.append(str(d['source'].item()))
            self._tasks.append(str(d['task'].item()))
        if task_to_id is None:
            unique = sorted(set(self._tasks))
            task_to_id = {t: i for i, t in enumerate(unique)}
        self.task_to_id = task_to_id
        n_real = sum((1 for s in self._sources if s == 'real'))
        n_dream = len(self._sources) - n_real
        bucket_counts = {b: self._buckets.count(b) for b in ALL_BUCKETS}
        logging.info(f'  ValueModelDataset: {len(self.files)} samples (real={n_real}, dream={n_dream}) | obs=({self.h}x{self.w}) future={self.num_future_frames} action_chunk={self.action_chunk}x{self.action_dim} state={self.state_dim} | bucket={bucket_counts} | tasks={len(self.task_to_id)}')

    def __len__(self):
        return len(self.files)

    @property
    def buckets(self) -> list[str]:
        return self._buckets

    @property
    def zs(self) -> list[float]:
        return self._zs

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        obs = torch.from_numpy(d['obs_frame'].copy()).permute(2, 0, 1).float() / 255.0
        future = torch.from_numpy(d['future_frames'].copy()).permute(0, 3, 1, 2).float() / 255.0
        action = torch.from_numpy(d['action_chunk'].astype(np.float32))
        state = torch.from_numpy(d['state'].astype(np.float32))
        z = torch.tensor(float(d['z']), dtype=torch.float32)
        task = str(d['task'].item())
        task_id = torch.tensor(self.task_to_id.get(task, 0), dtype=torch.long)
        return (obs, future, action, state, task_id, z)

def _collate_value(batch):
    obs, future, action, state, task_id, z = zip(*batch)
    return (torch.stack(obs), torch.stack(future), torch.stack(action), torch.stack(state), torch.stack(task_id), torch.stack(z))

def make_priority_sampler(buckets: list[str], target: dict[str, float] | None=None) -> WeightedRandomSampler:
    if target is None:
        target = DEFAULT_BUCKET_TARGET
    counts = {b: 0 for b in ALL_BUCKETS}
    for b in buckets:
        counts[b] = counts.get(b, 0) + 1
    present = {b: c for b, c in counts.items() if c > 0}
    if not present:
        raise ValueError('All buckets are empty.')
    eff_target = {}
    redistribute = sum((target[b] for b in target if counts.get(b, 0) == 0))
    sum_present_target = sum((target[b] for b in present))
    for b, c in present.items():
        share = target[b] + (redistribute * (target[b] / sum_present_target) if sum_present_target > 0 else 0)
        eff_target[b] = share
    norm = sum(eff_target.values())
    eff_target = {b: v / norm for b, v in eff_target.items()}
    weights = []
    for b in buckets:
        weights.append(eff_target.get(b, 0.0) / max(counts.get(b, 1), 1))
    logging.info(f'  PrioritySampler: bucket_counts={counts}, target_mass={eff_target}')
    return WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.float64), num_samples=len(buckets), replacement=True)

class ValueModel(nn.Module):

    def __init__(self, num_future_frames: int, num_tasks: int, action_chunk: int=50, action_dim: int=14, state_dim: int=14, dinov2_model: str='dinov2_vitb14', attn_dim: int=384, attn_heads: int=6, attn_layers: int=3, hidden_dim: int=256, dropout: float=0.1, freeze_backbone: bool=True, input_h: int=DEFAULT_SAVE_H, input_w: int=DEFAULT_SAVE_W):
        super().__init__()
        self.num_future_frames = num_future_frames
        self.action_chunk = action_chunk
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.input_h = input_h
        self.input_w = input_w
        self.backbone = torch.hub.load('facebookresearch/dinov2', dinov2_model)
        self.feature_dim = self.backbone.embed_dim
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.feat_proj = nn.Linear(self.feature_dim, attn_dim)
        self.frame_type_emb = nn.Parameter(torch.zeros(2, attn_dim))
        self.time_emb = nn.Parameter(torch.zeros(num_future_frames, attn_dim))
        self.value_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
        nn.init.trunc_normal_(self.frame_type_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb, std=0.02)
        nn.init.trunc_normal_(self.value_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=attn_dim, nhead=attn_heads, dim_feedforward=attn_dim * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        self.action_proj = nn.Sequential(nn.Linear(action_chunk * action_dim, attn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(attn_dim, attn_dim))
        self.task_emb = nn.Embedding(max(num_tasks, 1), attn_dim)
        self.state_proj = nn.Sequential(nn.Linear(state_dim, attn_dim), nn.GELU(), nn.Linear(attn_dim, attn_dim))
        self.head = nn.Sequential(nn.LayerNorm(attn_dim * 4), nn.Linear(attn_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def _encode_frames(self, imgs: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(imgs, size=(self.input_h, self.input_w), mode='bilinear', align_corners=False)
        x = (x - self.img_mean) / self.img_std
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        return self.feat_proj(feat)

    def forward(self, obs: torch.Tensor, future: torch.Tensor, action: torch.Tensor, state: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = future.shape
        if L > self.num_future_frames:
            raise ValueError(f'Future length {L} exceeds num_future_frames {self.num_future_frames}')
        obs_tok = self._encode_frames(obs).unsqueeze(1) + self.frame_type_emb[0].view(1, 1, -1)
        fut = self._encode_frames(future.reshape(B * L, C, H, W))
        fut = fut.view(B, L, -1)
        fut = fut + self.frame_type_emb[1].view(1, 1, -1) + self.time_emb[:L].unsqueeze(0)
        cls = self.value_token.expand(B, -1, -1)
        seq = torch.cat([cls, obs_tok, fut], dim=1)
        seq = self.temporal_attn(seq)
        pooled = seq[:, 0]
        action_tok = self.action_proj(action.reshape(B, -1))
        task_tok = self.task_emb(task_id)
        state_tok = self.state_proj(state)
        combined = torch.cat([pooled, action_tok, task_tok, state_tok], dim=-1)
        return self.head(combined).squeeze(-1)

def regression_plus_ranking_loss(q: torch.Tensor, z: torch.Tensor, rank_lambda: float, rank_alpha: float, huber_delta: float=1.0) -> tuple[torch.Tensor, dict]:
    huber = F.huber_loss(q, z, reduction='mean', delta=huber_delta)
    diff_z = z.unsqueeze(1) - z.unsqueeze(0)
    mask = (diff_z > 0).float()
    n_pairs = mask.sum()
    if n_pairs.item() == 0 or rank_lambda <= 0:
        rank = torch.zeros((), device=q.device)
    else:
        diff_q = q.unsqueeze(1) - q.unsqueeze(0)
        margin = F.relu(rank_alpha * diff_z - diff_q)
        rank = (margin * mask).sum() / n_pairs
    total = huber + rank_lambda * rank
    return (total, {'huber': huber.detach(), 'rank': rank.detach()})

def _split_by_rollout(data_dir: str, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    files = sorted(glob.glob(os.path.join(data_dir, 'rollout_*', 'step_*.npz')))
    if not files:
        raise FileNotFoundError(f'No npz files in {data_dir}')
    rollouts: dict[str, list[str]] = {}
    for f in files:
        rid = os.path.basename(os.path.dirname(f))
        rollouts.setdefault(rid, []).append(f)
    rollout_ids = sorted(rollouts.keys())
    rng = random.Random(seed)
    shuffled = rollout_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    if n_val >= len(shuffled):
        raise ValueError(f'val_ratio={val_ratio} leaves no train rollouts')
    val_ids = set(shuffled[:n_val])
    train_files = [f for r in rollout_ids if r not in val_ids for f in rollouts[r]]
    val_files = [f for r in rollout_ids if r in val_ids for f in rollouts[r]]
    logging.info(f'Split by rollout (seed={seed}): {len(rollout_ids) - n_val} train / {n_val} val rollouts ({len(train_files)} / {len(val_files)} clips)')
    return (train_files, val_files)

def _val_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = float((pred - target).pow(2).mean().item())
    mae = float((pred - target).abs().mean().item())
    if pred.numel() < 2:
        return {'mse': mse, 'mae': mae, 'rank_acc': float('nan')}
    pi, pj = (pred.unsqueeze(0), pred.unsqueeze(1))
    ti, tj = (target.unsqueeze(0), target.unsqueeze(1))
    valid = ti != tj
    correct = ((pi > pj) == (ti > tj)) & valid
    rank_acc = float(correct.sum().item()) / max(int(valid.sum().item()), 1)
    return {'mse': mse, 'mae': mae, 'rank_acc': rank_acc}

def train(args):
    rank, local_rank, world_size = _setup_distributed()
    distributed = world_size > 1
    is_main = rank == 0
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if is_main:
        logging.info(f'Using device: {device}  (world_size={world_size})')
    if args.val_ratio > 0:
        train_files, val_files = _split_by_rollout(args.data_dir, args.val_ratio, args.val_split_seed)
    else:
        train_files = sorted(glob.glob(os.path.join(args.data_dir, 'rollout_*', 'step_*.npz')))
        val_files = []
    train_ds = ValueModelDataset(args.data_dir, files=train_files)
    val_ds = ValueModelDataset(args.data_dir, files=val_files, task_to_id=train_ds.task_to_id) if val_files else None
    if args.priority_sample and (not distributed):
        target = None
        if args.bucket_target:
            target = json.loads(args.bucket_target)
        train_sampler = make_priority_sampler(train_ds.buckets, target=target)
        shuffle = False
    elif distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    if args.priority_sample and distributed and is_main:
        logging.warning('priority_sample is mutually exclusive with DDP; falling back to DistributedSampler. Run on 1 GPU to enable priority oversampling.')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_value)
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed and val_ds else None
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_value) if val_ds else None
    model = ValueModel(num_future_frames=train_ds.num_future_frames, num_tasks=max(len(train_ds.task_to_id), 1), action_chunk=train_ds.action_chunk, action_dim=train_ds.action_dim, state_dim=train_ds.state_dim, dinov2_model=args.dinov2_model, attn_dim=args.attn_dim, attn_heads=args.attn_heads, attn_layers=args.attn_layers, hidden_dim=args.hidden_dim, dropout=args.dropout, freeze_backbone=args.freeze_backbone, input_h=train_ds.h, input_w=train_ds.w).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    raw_model = model.module if distributed else model
    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        logging.info(f'Trainable parameters: {sum((p.numel() for p in trainable)):,}')
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_dir = pathlib.Path(args.output_dir)
    if args.wandb_run_name:
        output_dir = output_dir / args.wandb_run_name
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'task_to_id.json', 'w') as f:
            json.dump(train_ds.task_to_id, f, indent=2)
    if distributed:
        dist.barrier()
    use_wandb = is_main and wandb is not None and args.wandb_project
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args) | {'world_size': world_size, 'num_train_clips': len(train_ds), 'num_val_clips': len(val_ds) if val_ds else 0, 'buckets_train': {b: train_ds.buckets.count(b) for b in ALL_BUCKETS}, 'tasks': train_ds.task_to_id, 'input_resolution': [train_ds.h, train_ds.w]})
    best_val_mse = float('inf')
    for epoch in range(args.epochs):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        model.train()
        if args.freeze_backbone:
            raw_model.backbone.eval()
        run = {'loss': 0.0, 'huber': 0.0, 'rank': 0.0, 'n': 0}
        for obs, future, action, state, task_id, z in train_loader:
            obs = obs.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True)
            task_id = task_id.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            q = model(obs, future, action, state, task_id)
            loss, parts = regression_plus_ranking_loss(q, z, args.rank_lambda, args.rank_alpha, args.huber_delta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            run['loss'] += loss.item()
            run['huber'] += parts['huber'].item()
            run['rank'] += parts['rank'].item()
            run['n'] += 1
        n = max(run['n'], 1)
        train_loss = run['loss'] / n
        train_huber = run['huber'] / n
        train_rank = run['rank'] / n
        scheduler.step()
        val_str = 'N/A'
        val_metrics = None
        if val_loader is not None:
            model.eval()
            preds, targets = ([], [])
            with torch.no_grad():
                for obs, future, action, state, task_id, z in val_loader:
                    obs = obs.to(device, non_blocking=True)
                    future = future.to(device, non_blocking=True)
                    action = action.to(device, non_blocking=True)
                    state = state.to(device, non_blocking=True)
                    task_id = task_id.to(device, non_blocking=True)
                    z = z.to(device, non_blocking=True)
                    q = model(obs, future, action, state, task_id)
                    preds.append(q.cpu())
                    targets.append(z.cpu())
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            val_metrics = _val_metrics(preds, targets)
            val_str = f'mse={val_metrics['mse']:.4f} mae={val_metrics['mae']:.4f} rank_acc={val_metrics['rank_acc']:.3f}'
            if is_main and val_metrics['mse'] < best_val_mse:
                best_val_mse = val_metrics['mse']
                torch.save(raw_model.state_dict(), output_dir / 'best_model.pt')
                logging.info(f'  -> Saved best (val_mse={val_metrics['mse']:.4f})')
        if is_main:
            logging.info(f'Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f} huber={train_huber:.4f} rank={train_rank:.4f} | val=[{val_str}] | lr={scheduler.get_last_lr()[0]:.2e}')
        if use_wandb:
            log = {'epoch': epoch + 1, 'train/loss': train_loss, 'train/huber': train_huber, 'train/rank': train_rank, 'lr': scheduler.get_last_lr()[0]}
            if val_metrics is not None:
                log.update({'val/mse': val_metrics['mse'], 'val/mae': val_metrics['mae'], 'val/rank_acc': val_metrics['rank_acc']})
            wandb.log(log, step=epoch + 1)
        if is_main and (epoch + 1) % args.save_every == 0:
            torch.save(raw_model.state_dict(), output_dir / f'model_epoch{epoch + 1}.pt')
    if is_main:
        torch.save(raw_model.state_dict(), output_dir / 'model_final.pt')
        logging.info(f'Training complete. Models saved to {output_dir}')
    if use_wandb:
        wandb.finish()
    _cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Train V_omega(o_t, u_t, y_t) -> z_t value model with continuous progress-delta labels + real-dream joint training.')
    sub = parser.add_subparsers(dest='command', required=True)
    p_pack = sub.add_parser('pack', help='Pack a LeRobot v2.1 dataset into per-window value-model npz')
    p_pack.add_argument('--data_dir', type=str, required=True, help='LeRobot v2.1 dataset (must contain rewards/)')
    p_pack.add_argument('--rewards_dir', type=str, default=None)
    p_pack.add_argument('--rewards_fps', type=float, default=None, help='FPS the Robometer reward arrays were generated at. Auto-detected from rewards/summary.json if absent (default 3.0).')
    p_pack.add_argument('--output_dir', type=str, required=True)
    p_pack.add_argument('--episodes', type=str, default=None, help="e.g. '0:100' to subset")
    p_pack.add_argument('--extract_fps', type=float, default=5.0)
    p_pack.add_argument('--max_frames', type=int, default=2048)
    p_pack.add_argument('--num_future_frames', type=int, default=4, help='L = number of stitched future frames in y_t')
    p_pack.add_argument('--action_chunk', type=int, default=50, help='K = action chunk length (raw FPS)')
    p_pack.add_argument('--window_stride', type=int, default=1, help='Stride (in extracted frames) between window starts')
    p_pack.add_argument('--save_h', type=int, default=DEFAULT_SAVE_H)
    p_pack.add_argument('--save_w', type=int, default=DEFAULT_SAVE_W)
    p_pack.add_argument('--view_layout', type=str, default='top_left_right', choices=['top_left_right', 'top_right_left', 'left_top_right'])
    p_pack.add_argument('--success_threshold', type=float, default=0.5)
    p_pack.add_argument('--success_drop_threshold', type=float, default=0.3, help='Boundary penalty: reset z=-1 when peak success - final success exceeds this within the horizon')
    p_pack.add_argument('--failure_progress_threshold', type=float, default=0.05)
    p_pack.add_argument('--dreamdojo_metadata', type=str, default=None, help='Optional JSON with DreamDojo dream videos to pack alongside the real future frames')
    p_pack.add_argument('--overwrite', action='store_true')
    p_train = sub.add_parser('train', help='Train V_phi on packed clips')
    p_train.add_argument('--data_dir', type=str, required=True)
    p_train.add_argument('--val_ratio', type=float, default=0.1)
    p_train.add_argument('--val_split_seed', type=int, default=42)
    p_train.add_argument('--output_dir', type=str, default='checkpoints/value_model')
    p_train.add_argument('--dinov2_model', type=str, default='dinov2_vitb14', choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])
    p_train.add_argument('--attn_dim', type=int, default=384)
    p_train.add_argument('--attn_heads', type=int, default=6)
    p_train.add_argument('--attn_layers', type=int, default=3)
    p_train.add_argument('--hidden_dim', type=int, default=256)
    p_train.add_argument('--dropout', type=float, default=0.1)
    p_train.add_argument('--freeze_backbone', action='store_true', default=True)
    p_train.add_argument('--no_freeze_backbone', dest='freeze_backbone', action='store_false')
    p_train.add_argument('--batch_size', type=int, default=16)
    p_train.add_argument('--lr', type=float, default=0.0001)
    p_train.add_argument('--weight_decay', type=float, default=0.0001)
    p_train.add_argument('--epochs', type=int, default=30)
    p_train.add_argument('--num_workers', type=int, default=4)
    p_train.add_argument('--save_every', type=int, default=5)
    p_train.add_argument('--rank_lambda', type=float, default=0.5, help='lambda_rank balancing the Huber regression and continuous margin ranking term')
    p_train.add_argument('--rank_alpha', type=float, default=1.0, help='alpha scaling factor in the continuous margin: requires q_i - q_j >= alpha (z_i - z_j) for z_i > z_j')
    p_train.add_argument('--huber_delta', type=float, default=1.0, help='delta parameter of the Huber regression loss')
    p_train.add_argument('--priority_sample', action='store_true', default=True, help='Bucket-balanced sampling (single-GPU only)')
    p_train.add_argument('--no_priority_sample', dest='priority_sample', action='store_false')
    p_train.add_argument('--bucket_target', type=str, default=None, help='JSON dict overriding the bucket target mass')
    p_train.add_argument('--wandb_project', type=str, default=None)
    p_train.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    if args.command == 'pack':
        if args.save_h % 14 != 0 or args.save_w % 14 != 0:
            raise ValueError('save_h and save_w must be divisible by 14')
        if args.save_w % 3 != 0:
            raise ValueError('save_w must be divisible by 3')
        pack(args)
    elif args.command == 'train':
        train(args)
if __name__ == '__main__':
    main()
