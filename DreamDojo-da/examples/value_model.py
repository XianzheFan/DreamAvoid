from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SAVE_H = 224
DEFAULT_SAVE_W = 672

class ValueModel(nn.Module):
    def __init__(
        self,
        num_future_frames: int,
        num_tasks: int,
        action_chunk: int = 50,
        action_dim: int = 14,
        state_dim: int = 14,
        dinov2_model: str = "dinov2_vitb14",
        attn_dim: int = 384,
        attn_heads: int = 6,
        attn_layers: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        input_h: int = DEFAULT_SAVE_H,
        input_w: int = DEFAULT_SAVE_W,
    ):
        super().__init__()
        self.num_future_frames = num_future_frames
        self.action_chunk = action_chunk
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.input_h = input_h
        self.input_w = input_w
        self.backbone = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.feature_dim = self.backbone.embed_dim
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.feat_proj = nn.Linear(self.feature_dim, attn_dim)
        self.frame_type_emb = nn.Parameter(torch.zeros(2, attn_dim))
        self.time_emb = nn.Parameter(torch.zeros(num_future_frames, attn_dim))
        self.value_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
        nn.init.trunc_normal_(self.frame_type_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb, std=0.02)
        nn.init.trunc_normal_(self.value_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim, nhead=attn_heads, dim_feedforward=attn_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        self.action_proj = nn.Sequential(
            nn.Linear(action_chunk * action_dim, attn_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(attn_dim, attn_dim),
        )
        self.task_emb = nn.Embedding(max(num_tasks, 1), attn_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, attn_dim), nn.GELU(),
            nn.Linear(attn_dim, attn_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(attn_dim * 4),
            nn.Linear(attn_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_frames(self, imgs: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(imgs, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        return self.feat_proj(feat)

    def forward(self, obs, future, action, state, task_id):
        B, L, C, H, W = future.shape
        if L > self.num_future_frames:
            raise ValueError(f"Future length {L} exceeds num_future_frames {self.num_future_frames}")
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

def _infer_args_from_state_dict(sd: dict, dinov2_model: str, action_chunk: int, action_dim: int):
    num_future_frames = sd["time_emb"].shape[0]
    num_tasks = sd["task_emb.weight"].shape[0]
    attn_dim = sd["feat_proj.weight"].shape[0]
    state_dim = sd["state_proj.0.weight"].shape[1]
    hidden_dim = sd["head.1.weight"].shape[0]
    layer_indices = sorted({
        int(k.split(".")[2]) for k in sd
        if k.startswith("temporal_attn.layers.")
    })
    attn_layers = len(layer_indices)
    action_proj_in = sd["action_proj.0.weight"].shape[1]
    if action_proj_in != action_chunk * action_dim:

        for cand_dim in (14, 7, 6, 8, action_dim):
            if action_proj_in % cand_dim == 0:
                action_dim = cand_dim
                action_chunk = action_proj_in // cand_dim
                break
        else:
            raise ValueError(
                f"action_proj_in={action_proj_in} not divisible by any common action_dim. "
                "Pass --value-model-action-chunk and --value-model-action-dim explicitly."
            )
    return dict(
        num_future_frames=num_future_frames, num_tasks=num_tasks,
        action_chunk=action_chunk, action_dim=action_dim, state_dim=state_dim,
        dinov2_model=dinov2_model, attn_dim=attn_dim, attn_layers=attn_layers,
        hidden_dim=hidden_dim,
    )

def _resize_to_value_input(frame_uint8: np.ndarray, h: int = DEFAULT_SAVE_H, w: int = DEFAULT_SAVE_W) -> np.ndarray:
    import cv2
    if frame_uint8.shape[:2] == (h, w):
        return frame_uint8
    return cv2.resize(frame_uint8, (w, h), interpolation=cv2.INTER_LINEAR)

class ServerValueModel:

    def __init__(
        self,
        ckpt_path: str,
        action_chunk: int = 50,
        action_dim: int = 14,
        dinov2_model: str = "dinov2_vitb14",
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        ckpt_dir = Path(ckpt_path).parent
        task_to_id_path = ckpt_dir / "task_to_id.json"
        if not task_to_id_path.exists():
            raise FileNotFoundError(
                f"task_to_id.json not found next to ckpt at {task_to_id_path}. "
                "ValueModel requires this to map task name -> task_id."
            )
        with open(task_to_id_path) as f:
            self.task_to_id: dict[str, int] = json.load(f)
        kwargs = _infer_args_from_state_dict(sd, dinov2_model, action_chunk, action_dim)
        self.action_chunk = kwargs["action_chunk"]
        self.action_dim = kwargs["action_dim"]
        self.state_dim = kwargs["state_dim"]
        self.num_future_frames = kwargs["num_future_frames"]
        self.model = ValueModel(**kwargs).to(self.device).eval()
        missing, unexpected = self.model.load_state_dict(sd, strict=False)

        real_missing = [k for k in missing if k not in {"img_mean", "img_std"}]
        if real_missing or unexpected:
            print(f"[ValueModel] state_dict load: missing={real_missing} unexpected={unexpected}")
        print(
            f"[ValueModel] Loaded {ckpt_path} | "
            f"L={self.num_future_frames} action=({self.action_chunk},{self.action_dim}) "
            f"state={self.state_dim} tasks={list(self.task_to_id.keys())}"
        )

    def _prepare_video(self, video_uint8_thwc: np.ndarray) -> torch.Tensor:
        T = video_uint8_thwc.shape[0]
        L = min(T, self.num_future_frames)
        frames = []
        for t in range(L):
            frames.append(_resize_to_value_input(video_uint8_thwc[t]))

        while len(frames) < self.num_future_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((DEFAULT_SAVE_H, DEFAULT_SAVE_W, 3), dtype=np.uint8))
        arr = np.stack(frames, axis=0)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def _prepare_obs(self, obs_uint8_hwc: np.ndarray) -> torch.Tensor:
        arr = _resize_to_value_input(obs_uint8_hwc)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def _prepare_actions(self, actions: np.ndarray) -> torch.Tensor:
        a = np.asarray(actions, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[1] != self.action_dim:
            raise ValueError(
                f"action dim {a.shape[1]} != expected {self.action_dim}"
            )
        if a.shape[0] >= self.action_chunk:
            a = a[: self.action_chunk]
        else:
            pad = np.tile(a[-1:], (self.action_chunk - a.shape[0], 1)) if a.shape[0] > 0 \
                else np.zeros((self.action_chunk, self.action_dim), dtype=np.float32)
            a = np.concatenate([a, pad], axis=0)
        return torch.from_numpy(a).unsqueeze(0).to(self.device)

    def _prepare_state(self, state: np.ndarray | list[float] | None) -> torch.Tensor:
        if state is None or (hasattr(state, "__len__") and len(state) == 0):
            arr = np.zeros(self.state_dim, dtype=np.float32)
        else:
            arr = np.asarray(state, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != self.state_dim:
                raise ValueError(f"state shape {arr.shape} != ({self.state_dim},)")
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def _resolve_task_id(self, task: str) -> torch.Tensor:
        if task in self.task_to_id:
            tid = self.task_to_id[task]
        elif len(self.task_to_id) == 1:
            tid = next(iter(self.task_to_id.values()))
        else:
            print(f"[ValueModel] task '{task}' not in {list(self.task_to_id.keys())}, defaulting to 0")
            tid = 0
        return torch.tensor([tid], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def score(
        self,
        obs_uint8_hwc: np.ndarray,
        future_uint8_thwc: np.ndarray,
        actions: np.ndarray,
        state: np.ndarray | list[float] | None,
        task: str,
    ) -> float:
        obs = self._prepare_obs(obs_uint8_hwc)
        future = self._prepare_video(future_uint8_thwc)
        action = self._prepare_actions(actions)
        state_t = self._prepare_state(state)
        task_id = self._resolve_task_id(task)
        z = self.model(obs, future, action, state_t, task_id)
        return float(z.squeeze().item())
