import argparse
import glob
import json
import logging
import math
import os
import pathlib
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    import wandb
except ImportError:
    wandb = None
from train_switch_head_robometer import CAMERA_NAMES, CAMERA_TO_KEY, _cleanup_distributed, _collate_switch, _setup_distributed, extract_frames_pyav, get_video_path, load_dataset_info, load_episode_parquet, load_episodes, load_task
from train_switch_head_dual import _split_by_rollout

def soft_label(t: np.ndarray, t_crit: float, beta: float) -> np.ndarray:
    z = (t.astype(np.float32) - float(t_crit)) / max(float(beta), 1e-06)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

def _load_annotations(path: str) -> dict[int, dict]:
    with open(path) as f:
        raw = json.load(f)
    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            ep = int(k)
        except ValueError:
            continue
        if isinstance(v, (int, float)):
            out[ep] = {'t_crit': int(v), 'fps': None}
        else:
            out[ep] = {'t_crit': int(v['t_crit']), 'fps': v.get('fps')}
    return out

def pack(args):
    data_dir = pathlib.Path(args.data_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = _load_annotations(args.annotations)
    logging.info(f'Loaded t_crit annotations for {len(annotations)} episodes')
    info = load_dataset_info(data_dir)
    episodes = load_episodes(data_dir)
    task = load_task(data_dir)
    dataset_fps = info['fps']
    available_eps = sorted(annotations.keys())
    if args.episodes:
        ep_start, ep_end = map(int, args.episodes.split(':'))
        available_eps = [e for e in available_eps if ep_start <= e < ep_end]
    extract_fps = float(args.extract_fps)
    beta_extracted = float(args.beta)
    clip_len = int(args.clip_len)
    replan_interval = int(args.replan_interval)
    logging.info(f'Dataset: {data_dir}')
    logging.info(f'Task   : {task}')
    logging.info(f'Episodes to pack: {len(available_eps)} (annotated)')
    logging.info(f'Extract FPS: {extract_fps} (from {dataset_fps}fps video) | K={clip_len} | replan={replan_interval} | beta={beta_extracted} frames')
    all_episode_meta = []
    stats = {'total_episodes': 0, 'total_pos_steps': 0, 'total_neg_steps': 0}
    t_start = time.time()
    for ep_idx in available_eps:
        ann = annotations[ep_idx]
        ann_fps = ann['fps'] or dataset_fps
        t_crit_extracted = ann['t_crit'] * (extract_fps / float(ann_fps))
        task_segment = task.replace(' ', '_')[:50]
        rollout_dir = output_dir / f'rollout_{task_segment}_ep{ep_idx}_done'
        if rollout_dir.exists() and any(rollout_dir.glob('step_*.npz')):
            logging.info(f'  Skip episode {ep_idx} (already packed)')
            continue
        camera_frames = {}
        ok = True
        for cam in CAMERA_NAMES:
            video_path = get_video_path(data_dir, ep_idx, cam)
            if not video_path.exists():
                logging.warning(f'  Episode {ep_idx}: {cam} video not found')
                ok = False
                break
            frames = extract_frames_pyav(str(video_path), fps=extract_fps, max_frames=args.max_frames)
            if frames.size == 0:
                logging.warning(f'  Episode {ep_idx}: failed to extract {cam}')
                ok = False
                break
            camera_frames[cam] = frames
        if not ok:
            continue
        T = min((f.shape[0] for f in camera_frames.values()))
        for cam in camera_frames:
            camera_frames[cam] = camera_frames[cam][:T]
        states, _actions, _ = load_episode_parquet(data_dir, ep_idx)
        state_interval = max(1, int(round(dataset_fps / extract_fps)))
        sampled_states = states[::state_interval][:T]
        rollout_dir.mkdir(parents=True, exist_ok=True)
        n_pos = n_neg = 0
        for step_i, frame_i in enumerate(range(0, T, replan_interval)):
            y_tilde = float(soft_label(np.array([frame_i]), t_crit_extracted, beta_extracted)[0])
            clips = {}
            for cam in CAMERA_NAMES:
                cam_key = CAMERA_TO_KEY[cam]
                cam_clip = list(camera_frames[cam][max(0, frame_i - clip_len + 1):frame_i + 1])
                if len(cam_clip) < clip_len:
                    cam_clip = [cam_clip[0]] * (clip_len - len(cam_clip)) + cam_clip
                clips[cam_key] = np.stack(cam_clip)
            state = sampled_states[frame_i].copy() if frame_i < len(sampled_states) else np.zeros(states.shape[1], dtype=np.float32)
            save_path = rollout_dir / f'step_{step_i:04d}.npz'
            np.savez_compressed(save_path, top=camera_frames['cam_high'][frame_i].copy(), right=camera_frames['cam_right_wrist'][frame_i].copy(), left=camera_frames['cam_left_wrist'][frame_i].copy(), top_clip=clips['top'], right_clip=clips['right'], left_clip=clips['left'], state=state.astype(np.float32), y_tilde=np.float32(y_tilde), t_crit=np.float32(t_crit_extracted), frame_idx=np.int32(frame_i), episode_idx=np.int32(ep_idx), clip_len=np.int32(clip_len), beta=np.float32(beta_extracted), task=np.array(task), prompt=np.array(task))
            if y_tilde >= 0.5:
                n_pos += 1
            else:
                n_neg += 1
        stats['total_episodes'] += 1
        stats['total_pos_steps'] += n_pos
        stats['total_neg_steps'] += n_neg
        all_episode_meta.append({'episode_idx': ep_idx, 'task': task, 't_crit_raw': ann['t_crit'], 't_crit_extracted': float(t_crit_extracted), 'num_steps': n_pos + n_neg, 'num_pos': n_pos, 'num_neg': n_neg})
        elapsed = time.time() - t_start
        eps_done = len(all_episode_meta)
        eta = (len(available_eps) - eps_done) * elapsed / max(eps_done, 1)
        logging.info(f'  Episode {ep_idx}: steps={n_pos + n_neg} pos={n_pos} neg={n_neg} | t_crit={t_crit_extracted:.1f} | ETA {eta:.0f}s')
    with open(output_dir / 'collection_meta.json', 'w') as f:
        json.dump({'stats': stats, 'episodes': all_episode_meta}, f, indent=2)
    total = stats['total_pos_steps'] + stats['total_neg_steps']
    logging.info(f'Pack complete. {stats['total_episodes']} episodes, {total} steps (pos={stats['total_pos_steps']}, neg={stats['total_neg_steps']})')

class DreamTriggerDataset(Dataset):
    IMAGE_KEYS = ['top', 'right', 'left']

    def __init__(self, data_dir: str, files: list | None=None):
        if files is not None:
            self.files = sorted(files)
        else:
            self.files = sorted(glob.glob(os.path.join(data_dir, 'rollout_*', 'step_*.npz')))
        if not self.files:
            raise FileNotFoundError(f'No .npz files found in {data_dir}')
        first = np.load(self.files[0], allow_pickle=True)
        if 'y_tilde' not in first:
            raise KeyError("Packed .npz must contain 'y_tilde'. Re-pack with train_dream_trigger.py pack ...")
        self.state_dim = first['state'].shape[0]
        self.has_clips = all((f'{k}_clip' in first for k in self.IMAGE_KEYS))
        valid, ys = ([], [])
        for f in self.files:
            try:
                d = np.load(f)
                ys.append(float(d['y_tilde']))
                valid.append(f)
            except Exception:
                logging.warning(f'  Skipping corrupt file: {f}')
        self.files = valid
        self._ys = ys
        n_pos = sum((1 for y in ys if y >= 0.5))
        n_neg = len(ys) - n_pos
        logging.info(f'  DreamTrigger dataset: {len(valid)} samples | pos={n_pos} neg={n_neg} | y_tilde μ={np.mean(ys):.3f} σ={np.std(ys):.3f} | clip={self.has_clips}')

    def __len__(self):
        return len(self.files)

    @property
    def pos_weight(self) -> float:
        n_pos = sum((1 for y in self._ys if y >= 0.5))
        n_neg = len(self._ys) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        images = []
        for key in self.IMAGE_KEYS:
            if self.has_clips and f'{key}_clip' in data:
                clip = torch.from_numpy(data[f'{key}_clip'].copy()).permute(0, 3, 1, 2).float() / 255.0
                images.append(clip)
            else:
                img = torch.from_numpy(data[key].copy()).permute(2, 0, 1).float() / 255.0
                images.append(img)
        state = torch.from_numpy(data['state'].astype(np.float32))
        y_tilde = torch.tensor(float(data['y_tilde']), dtype=torch.float32)
        return (images, state, y_tilde)

class DreamTrigger(nn.Module):

    def __init__(self, dinov2_model: str='dinov2_vitb14', state_dim: int=14, num_cameras: int=3, max_clip_frames: int=20, attn_dim: int=384, attn_heads: int=4, attn_layers: int=2, hidden_dim: int=256, dropout: float=0.1, freeze_backbone: bool=True):
        super().__init__()
        self.num_cameras = num_cameras
        self.max_clip_frames = max_clip_frames
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
        self.cam_emb = nn.Parameter(torch.zeros(num_cameras, attn_dim))
        self.time_emb = nn.Parameter(torch.zeros(max_clip_frames, attn_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
        nn.init.trunc_normal_(self.cam_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=attn_dim, nhead=attn_heads, dim_feedforward=attn_dim * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        trunk_in = attn_dim + state_dim
        self.trunk = nn.Sequential(nn.LayerNorm(trunk_in), nn.Linear(trunk_in, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_dim, 1)

    def _encode_cam_clip(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dim() == 4:
            clip = clip.unsqueeze(1)
        B, T, C, H, W = clip.shape
        flat = clip.reshape(B * T, C, H, W)
        flat = F.interpolate(flat, size=(224, 224), mode='bilinear', align_corners=False)
        flat = (flat - self.img_mean) / self.img_std
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(flat)
        else:
            feat = self.backbone(flat)
        return feat.view(B, T, -1)

    def forward(self, images: list[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        cam_tokens = []
        for c, img in enumerate(images):
            feat = self._encode_cam_clip(img)
            feat = self.feat_proj(feat)
            T = feat.shape[1]
            if T > self.max_clip_frames:
                raise ValueError(f'clip length {T} exceeds max_clip_frames {self.max_clip_frames}')
            feat = feat + self.time_emb[:T].unsqueeze(0)
            feat = feat + self.cam_emb[c].view(1, 1, -1)
            cam_tokens.append(feat)
        tokens = torch.cat(cam_tokens, dim=1)
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.temporal_attn(seq)
        pooled = seq[:, 0]
        combined = torch.cat([pooled, state], dim=-1)
        return self.head(self.trunk(combined)).squeeze(-1)

    @torch.no_grad()
    def predict_prob(self, images, state) -> torch.Tensor:
        return torch.sigmoid(self.forward(images, state))

    @torch.no_grad()
    def trigger(self, images, state, gamma: float=0.5) -> torch.Tensor:
        return (self.predict_prob(images, state) >= gamma).float()

def weighted_soft_bce_with_logits(logits: torch.Tensor, y_tilde: torch.Tensor, w_pos: float) -> torch.Tensor:
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    loss = -(w_pos * y_tilde * log_p + (1.0 - y_tilde) * log_1mp)
    return loss.mean()

def train(args):
    rank, local_rank, world_size = _setup_distributed()
    distributed = world_size > 1
    is_main = rank == 0
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if is_main:
        logging.info(f'Using device: {device}  (world_size={world_size})')
    if args.val_ratio > 0:
        train_files, val_files = _split_by_rollout(args.data_dir, args.val_ratio, args.val_split_seed)
        train_ds = DreamTriggerDataset(args.data_dir, files=train_files)
        val_ds = DreamTriggerDataset(args.data_dir, files=val_files)
    else:
        train_ds = DreamTriggerDataset(args.data_dir)
        val_ds = None
    state_dim = train_ds.state_dim
    model = DreamTrigger(dinov2_model=args.dinov2_model, state_dim=state_dim, num_cameras=3, max_clip_frames=args.max_clip_frames, attn_dim=args.attn_dim, attn_heads=args.attn_heads, attn_layers=args.attn_layers, hidden_dim=args.hidden_dim, dropout=args.dropout, freeze_backbone=args.freeze_backbone).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    raw_model = model.module if distributed else model
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_switch)
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed and val_ds else None
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_switch) if val_ds else None
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        logging.info(f'Trainable parameters: {sum((p.numel() for p in trainable_params)):,}')
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.pos_weight is not None:
        w_pos = float(args.pos_weight)
    else:
        w_pos = float(train_ds.pos_weight)
    if is_main:
        logging.info(f'Class-balanced w_+ = N_-/N_+ = {w_pos:.3f}')
    output_dir = pathlib.Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    use_wandb = is_main and wandb is not None and args.wandb_project
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args) | {'world_size': world_size, 'w_pos': w_pos})
    best_val_f1 = -float('inf')
    gamma = float(args.gamma)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if args.freeze_backbone:
            raw_model.backbone.eval()
        total_loss, nb = (0.0, 0)
        for images, state, y_tilde in train_loader:
            images = [img.to(device) for img in images]
            state = state.to(device)
            y_tilde = y_tilde.to(device)
            logits = model(images, state)
            loss = weighted_soft_bce_with_logits(logits, y_tilde, w_pos)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        avg_loss = total_loss / max(nb, 1)
        scheduler.step()
        val_msg = 'N/A'
        avg_vl = vacc = vp = vr = vf1 = vmae = 0.0
        if val_loader is not None:
            model.eval()
            vl, vn = (0.0, 0)
            tp = fp = fn = tn = 0
            mae_sum = 0.0
            n_samples = 0
            with torch.no_grad():
                for images, state, y_tilde in val_loader:
                    images = [img.to(device) for img in images]
                    state = state.to(device)
                    y_tilde = y_tilde.to(device)
                    logits = model(images, state)
                    vl += weighted_soft_bce_with_logits(logits, y_tilde, w_pos).item()
                    vn += 1
                    p = torch.sigmoid(logits)
                    pred = (p >= gamma).float()
                    hard = (y_tilde >= 0.5).float()
                    tp += int(((pred == 1) & (hard == 1)).sum().item())
                    fp += int(((pred == 1) & (hard == 0)).sum().item())
                    fn += int(((pred == 0) & (hard == 1)).sum().item())
                    tn += int(((pred == 0) & (hard == 0)).sum().item())
                    mae_sum += (p - y_tilde).abs().sum().item()
                    n_samples += y_tilde.numel()
            avg_vl = vl / max(vn, 1)
            vacc = (tp + tn) / max(tp + tn + fp + fn, 1)
            vp = tp / max(tp + fp, 1)
            vr = tp / max(tp + fn, 1)
            vf1 = 2 * vp * vr / max(vp + vr, 1e-08)
            vmae = mae_sum / max(n_samples, 1)
            val_msg = f'loss={avg_vl:.4f} acc={vacc:.3f} P={vp:.3f} R={vr:.3f} F1={vf1:.3f} MAE={vmae:.3f}'
            if is_main and vf1 > best_val_f1:
                best_val_f1 = vf1
                torch.save(raw_model.state_dict(), output_dir / 'best_model.pt')
                logging.info(f'  -> Saved best (F1={vf1:.4f})')
        if is_main:
            logging.info(f'Epoch {epoch + 1}/{args.epochs}  train_loss={avg_loss:.4f}  val=[{val_msg}]  lr={scheduler.get_last_lr()[0]:.2e}')
        if use_wandb:
            log = {'epoch': epoch + 1, 'train/loss': avg_loss, 'lr': scheduler.get_last_lr()[0]}
            if val_loader is not None:
                log.update({'val/loss': avg_vl, 'val/acc': vacc, 'val/precision': vp, 'val/recall': vr, 'val/f1': vf1, 'val/mae': vmae})
            wandb.log(log, step=epoch + 1)
        if is_main and (epoch + 1) % args.save_every == 0:
            torch.save(raw_model.state_dict(), output_dir / f'model_epoch{epoch + 1}.pt')
    if is_main:
        torch.save(raw_model.state_dict(), output_dir / 'model_final.pt')
        logging.info(f'Training complete. Saved to {output_dir}')
    if use_wandb:
        wandb.finish()
    _cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Dream Trigger: pack t_crit annotations and train f_phi.')
    sub = parser.add_subparsers(dest='command', required=True)
    p_pack = sub.add_parser('pack', help='Pack t_crit annotations into .npz files')
    p_pack.add_argument('--data_dir', type=str, required=True, help='Path to LeRobot v2.1 dataset root')
    p_pack.add_argument('--annotations', type=str, required=True, help='Path to t_crit annotation JSON')
    p_pack.add_argument('--output_dir', type=str, required=True)
    p_pack.add_argument('--extract_fps', type=float, default=10.0)
    p_pack.add_argument('--max_frames', type=int, default=2048)
    p_pack.add_argument('--clip_len', type=int, default=8, help='K, number of recent frames per camera')
    p_pack.add_argument('--replan_interval', type=int, default=3)
    p_pack.add_argument('--beta', type=float, default=5.0, help='Soft-label transition width in extracted-clip frames')
    p_pack.add_argument('--episodes', type=str, default=None, help="Episode index range, e.g. '0:50'")
    p_train = sub.add_parser('train', help='Train Dream Trigger')
    p_train.add_argument('--data_dir', type=str, required=True)
    p_train.add_argument('--val_ratio', type=float, default=0.1)
    p_train.add_argument('--val_split_seed', type=int, default=42)
    p_train.add_argument('--output_dir', type=str, default=None)
    p_train.add_argument('--dinov2_model', type=str, default='dinov2_vitb14', choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])
    p_train.add_argument('--max_clip_frames', type=int, default=20)
    p_train.add_argument('--attn_dim', type=int, default=384)
    p_train.add_argument('--attn_heads', type=int, default=4)
    p_train.add_argument('--attn_layers', type=int, default=2)
    p_train.add_argument('--hidden_dim', type=int, default=256)
    p_train.add_argument('--dropout', type=float, default=0.1)
    p_train.add_argument('--freeze_backbone', action='store_true', default=True)
    p_train.add_argument('--no_freeze_backbone', dest='freeze_backbone', action='store_false')
    p_train.add_argument('--batch_size', type=int, default=32)
    p_train.add_argument('--lr', type=float, default=0.0001)
    p_train.add_argument('--weight_decay', type=float, default=0.0001)
    p_train.add_argument('--epochs', type=int, default=20)
    p_train.add_argument('--num_workers', type=int, default=4)
    p_train.add_argument('--save_every', type=int, default=5)
    p_train.add_argument('--gamma', type=float, default=0.5, help='Trigger threshold for val precision/recall/F1')
    p_train.add_argument('--pos_weight', type=float, default=None, help='Override w_+ (default: N_-/N_+ from data)')
    p_train.add_argument('--wandb_project', type=str, default=None)
    p_train.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    if args.command == 'pack':
        pack(args)
    else:
        if args.output_dir is None:
            tag = args.wandb_run_name or pathlib.Path(args.data_dir).name
            args.output_dir = str(pathlib.Path('checkpoints/dream_trigger') / tag)
        train(args)
if __name__ == '__main__':
    main()
