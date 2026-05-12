import argparse
import base64
import json
import os
import threading
from pathlib import Path

import mediapy
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cosmos_oss.init import init_environment
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.config import DEFAULT_NEGATIVE_PROMPT

from value_model import ServerValueModel

_server_value_model: ServerValueModel | None = None

class GenerateRequest(BaseModel):
    frame: str
    frame_height: int
    frame_width: int
    actions: list[list[float]]
    save_name: str = "output"
    prompt: str = ""
    seed: int = 0
    guidance: float = 0.0
    num_latent_conditional_frames: int = 1

    frame_left: str | None = None
    frame_right: str | None = None

    state: list[float] = []
    task: str = ""

    source_episode: int = -1
    source_frame: int = -1
    progress: list[float] = []
    success: list[float] = []

class GenerateResponse(BaseModel):
    save_path: str
    score: float | None = None

app = FastAPI(title="DreamDojo API")
_model: Video2WorldInference | None = None
_save_dir: Path | None = None
_server_id: str | None = None
_cp_size: int = 1
_fps: int = 8

_action_slot_start  = 169
_action_slot_end    = 176
_model_action_dim   = 384
_model_action_chunk = 12

_action_min: np.ndarray | None = None
_action_max: np.ndarray | None = None

_SIGNAL_EXIT     = torch.tensor([0], dtype=torch.int64)
_SIGNAL_GENERATE = torch.tensor([1], dtype=torch.int64)

_inference_lock = threading.Lock()

def _normalize_actions(actions: np.ndarray) -> np.ndarray:
    if _action_min is None or _action_max is None:
        return actions
    denom = _action_max - _action_min
    denom = np.where(denom < 1e-8, 1.0, denom)
    normalized = 2.0 * (actions - _action_min) / denom - 1.0
    return np.clip(normalized, -1.0, 1.0)

def _preprocess_actions(actions: np.ndarray) -> torch.Tensor:
    T, raw_dim = actions.shape
    expected_raw_dim = _action_slot_end - _action_slot_start
    assert raw_dim == expected_raw_dim, (
        f"Expected raw action dim {expected_raw_dim} "
        f"(slot {_action_slot_start}:{_action_slot_end}), got {raw_dim}"
    )

    norm_actions = _normalize_actions(actions.astype(np.float32))

    num_entries = _model_action_chunk + 1
    if T < num_entries:
        pad = np.tile(norm_actions[-1:], (num_entries - T, 1))
        padded = np.concatenate([norm_actions, pad], axis=0)
    else:
        padded = norm_actions[:num_entries]

    delta_list = []
    for t in range(1, len(padded) - 1, 4):
        delta_list.append(padded[t:t + 4] - padded[t - 1])
    delta_actions = np.concatenate(delta_list, axis=0).astype(np.float32)

    action_seq = np.zeros((_model_action_chunk, _model_action_dim), dtype=np.float32)
    n = min(len(delta_actions), _model_action_chunk)
    action_seq[:n, _action_slot_start:_action_slot_end] = delta_actions[:n]
    return torch.from_numpy(action_seq)

def _build_vid_input(frame_np: np.ndarray, num_frames: int) -> torch.Tensor:
    img_tensor = torchvision.transforms.functional.to_tensor(frame_np).unsqueeze(0) * 255.0
    img_tensor = img_tensor.to(torch.uint8)
    vid_input = torch.cat(
        [img_tensor, torch.zeros_like(img_tensor).repeat(num_frames - 1, 1, 1, 1)], dim=0
    ).unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    return vid_input

def _tile_3view(top_np: np.ndarray, left_np: np.ndarray, right_np: np.ndarray) -> np.ndarray:
    import cv2
    H, W = 480, 640
    h, w = H // 2, W // 2
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h, :w] = cv2.resize(top_np,   (w, h), interpolation=cv2.INTER_LINEAR)
    canvas[:h, w:] = cv2.resize(left_np,  (w, h), interpolation=cv2.INTER_LINEAR)
    canvas[h:, :w] = cv2.resize(right_np, (w, h), interpolation=cv2.INTER_LINEAR)
    return canvas

def _broadcast_inputs(vid_input, action, seed, guidance, num_latent_conditional_frames, prompt=""):
    device = torch.device("cuda")

    prompt_bytes = prompt.encode("utf-8") if dist.get_rank() == 0 else b""
    if dist.get_rank() == 0:
        meta = torch.tensor(
            [vid_input.shape[2], action.shape[0], action.shape[1], seed, num_latent_conditional_frames,
             int(guidance * 1e6), len(prompt_bytes)],
            dtype=torch.int64, device=device,
        )
    else:
        meta = torch.zeros(7, dtype=torch.int64, device=device)
    dist.broadcast(meta, src=0)

    T            = int(meta[0].item())
    action_steps = int(meta[1].item())
    action_dim   = int(meta[2].item())
    seed         = int(meta[3].item())
    num_lcf      = int(meta[4].item())
    guidance     = float(meta[5].item()) / 1e6
    prompt_len   = int(meta[6].item())

    if dist.get_rank() != 0:
        vid_input = torch.zeros(1, 3, T, 480, 640, dtype=torch.uint8, device=device)
        action    = torch.zeros(action_steps, action_dim, dtype=torch.float32, device=device)
    else:
        vid_input = vid_input.to(device)
        action    = action.to(device)

    dist.broadcast(vid_input, src=0)
    dist.broadcast(action,    src=0)

    if prompt_len > 0:
        if dist.get_rank() == 0:
            prompt_tensor = torch.frombuffer(
                prompt_bytes.ljust(prompt_len, b"\x00"), dtype=torch.uint8
            ).to(device)
        else:
            prompt_tensor = torch.zeros(prompt_len, dtype=torch.uint8, device=device)
        dist.broadcast(prompt_tensor, src=0)
        prompt = prompt_tensor.cpu().numpy().tobytes().rstrip(b"\x00").decode("utf-8")
    else:
        prompt = ""

    return vid_input.cpu(), action.cpu(), seed, guidance, num_lcf, prompt

def _run_inference(vid_input, action, seed, guidance, num_latent_conditional_frames, prompt=""):
    with torch.no_grad():
        video = _model.generate_vid2world(
            prompt=prompt,
            input_path=vid_input,
            action=action.float(),
            guidance=guidance,
            num_video_frames=vid_input.shape[2],
            num_latent_conditional_frames=num_latent_conditional_frames,
            resolution="480,640",
            seed=seed,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            lam_video=None,
        )
    return video

def worker_loop():
    device = torch.device("cuda")
    while True:
        signal = _SIGNAL_EXIT.clone().to(device)
        dist.broadcast(signal, src=0)

        if signal.item() == _SIGNAL_EXIT.item():
            break

        vid_input, action, seed, guidance, num_lcf, prompt = _broadcast_inputs(
            None, None, None, None, None
        )
        _run_inference(vid_input, action, seed, guidance, num_lcf, prompt)

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    def _decode(b64: str) -> np.ndarray:
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.uint8).reshape(req.frame_height, req.frame_width, 3)

    frame_np = _decode(req.frame)
    if frame_np.shape[:2] != (480, 640):
        import cv2
        frame_np = cv2.resize(frame_np, (640, 480))

    if req.frame_left is not None and req.frame_right is not None:
        left_np  = _decode(req.frame_left)
        right_np = _decode(req.frame_right)
        if left_np.shape[:2]  != (480, 640):
            import cv2
            left_np  = cv2.resize(left_np,  (640, 480))
        if right_np.shape[:2] != (480, 640):
            import cv2
            right_np = cv2.resize(right_np, (640, 480))
        cond_np = _tile_3view(frame_np, left_np, right_np)
    else:

        cond_np = frame_np

    actions   = np.array(req.actions, dtype=np.float32)

    print(f"[DEBUG] Received actions shape={actions.shape}, "
          f"abs_mean={np.abs(actions).mean():.4f}, "
          f"min={actions.min():.4f}, max={actions.max():.4f}", flush=True)
    print(f"[DEBUG] Actions[0]: {actions[0]}", flush=True)
    model_required_frames = _model.model.tokenizer.get_pixel_num_frames(_model.model.config.state_t)
    vid_input  = _build_vid_input(cond_np, model_required_frames)
    action_t   = _preprocess_actions(actions)

    _deltas = action_t[:, _action_slot_start:_action_slot_end].numpy()
    print(f"[DEBUG] Deltas abs_mean={np.abs(_deltas).mean():.6f}, "
          f"abs_max={np.abs(_deltas).max():.6f}", flush=True)

    with _inference_lock:
        if _cp_size > 1:

            signal = _SIGNAL_GENERATE.clone().to("cuda")
            dist.broadcast(signal, src=0)
            vid_input, action_t, seed, guidance, num_lcf, prompt = _broadcast_inputs(
                vid_input, action_t, req.seed, req.guidance, req.num_latent_conditional_frames,
                req.prompt,
            )
        else:
            seed, guidance, num_lcf, prompt = (
                req.seed, req.guidance, req.num_latent_conditional_frames, req.prompt
            )

        video = _run_inference(vid_input, action_t, seed, guidance, num_lcf, prompt)

    video_np = (
        torch.clamp((video[0] + 1) / 2, 0, 1) * 255
    ).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()

    rel = Path(req.save_name)
    parts = [p for p in rel.parts if p not in ("..", "/", "\\", "")]
    rel = Path(*parts) if parts else Path("output")

    if _server_id is not None:
        rel = rel.with_name(f"{rel.name}_s{_server_id}")
    base = (_save_dir / rel).with_suffix("")

    i = 0
    while base.with_name(f"{base.name}_ep{i}.mp4").exists():
        i += 1
    save_path = base.with_name(f"{base.name}_ep{i}.mp4")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(save_path), video_np, fps=_fps)

    final_score = None
    if _server_value_model is not None:
        final_score = _server_value_model.score(
            obs_uint8_hwc=cond_np,
            future_uint8_thwc=video_np[1:],
            actions=actions,
            state=req.state,
            task=req.task,
        )
        print(f"[ValueModel] In-memory score computed: {final_score:.4f}", flush=True)

    sidecar = {
        "video": str(save_path.resolve()),
        "task": req.task,
    }
    if final_score is not None:
        sidecar["score"] = float(final_score)
    if req.source_episode >= 0:
        sidecar["source_episode"] = int(req.source_episode)
    if req.source_frame >= 0:
        sidecar["source_frame"] = int(req.source_frame)
    if req.progress:
        sidecar["progress"] = list(req.progress)
    if req.success:
        sidecar["success"] = list(req.success)
    score_path = save_path.with_suffix(".json")
    with open(score_path, "w") as f:
        json.dump(sidecar, f)

    return GenerateResponse(save_path=str(save_path), score=final_score)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--experiment",  required=True)
    parser.add_argument("--save-dir",    required=True)
    parser.add_argument("--config-file", default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py")
    parser.add_argument("--port",        type=int, default=8000)
    parser.add_argument("--host",        default="0.0.0.0")
    parser.add_argument("--server-id",   default=None,
                        help="Tag appended to saved filenames as _s<id> to disambiguate parallel servers "
                             "writing to a shared --save-dir. Omit for no tag.")
    parser.add_argument("--context-parallel-size", type=int, default=1,
                        help="Number of GPUs per server for context parallelism")
    parser.add_argument("--action-slot-start", type=int, default=169,
                        help="Start index in 384-dim embodiment vector for raw actions (default: 169 = LIBERO)")
    parser.add_argument("--action-slot-end",   type=int, default=176,
                        help="End index (exclusive) in 384-dim embodiment vector (default: 176 = LIBERO 7-DoF)")
    parser.add_argument("--model-action-dim",  type=int, default=384,
                        help="Total action dim expected by the model (default: 384)")
    parser.add_argument("--model-action-chunk", type=int, default=12,
                        help="num_action_per_chunk expected by the model (default: 12)")
    parser.add_argument("--fps", type=int, default=8,
                        help="FPS for saved mp4 videos (should match training data fps, e.g. 3=Fractal, 5=Bridge, 10=LIBERO, 8≈new_agilex_3view 7.5fps)")
    parser.add_argument("--value-model-ckpt", type=str, default=None,
                        help="Path to DINOv2 value model checkpoint for in-memory scoring. "
                             "Training code: openpi-da/agilex/train_value_model.py")
    parser.add_argument("--stats-json", type=str, default=None,
                        help="Path to dataset stats.json for action min-max normalization. "
                             "If provided, raw actions are normalized to [-1,1] before inference.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    _save_dir  = Path(args.save_dir)
    _server_id = args.server_id
    _cp_size   = args.context_parallel_size
    _fps       = args.fps

    _action_slot_start  = args.action_slot_start
    _action_slot_end    = args.action_slot_end
    _model_action_dim   = args.model_action_dim
    _model_action_chunk = args.model_action_chunk

    if args.stats_json:
        with open(args.stats_json) as f:
            stats = json.load(f)
        _action_min = np.array(stats["action"]["min"], dtype=np.float32)
        _action_max = np.array(stats["action"]["max"], dtype=np.float32)
        print(f"[DreamDojo] Loaded action stats from {args.stats_json}")
        print(f"  action min: {_action_min}")
        print(f"  action max: {_action_max}")
    else:
        print("[DreamDojo] WARNING: No --stats-json provided, actions will NOT be normalized.")

    init_environment()

    if args.value_model_ckpt:
        print(f"[DreamDojo] Loading value model from {args.value_model_ckpt}...")
        _server_value_model = ServerValueModel(checkpoint_path=args.value_model_ckpt)
        print("[DreamDojo] Value model loaded.")

    _model = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=Path(args.checkpoint),
        s3_credential_path="",
        context_parallel_size=_cp_size,
        config_file=args.config_file,
    )
    state_t = _model.model.config.state_t
    if _cp_size > 1 and state_t % _cp_size != 0:
        raise ValueError(
            f"context_parallel_size={_cp_size} must divide state_t={state_t} "
            f"(T_latent={state_t}). Use --context-parallel-size 1, 2, or 4."
        )
    print(f"[rank {dist.get_rank() if dist.is_initialized() else 0}] "
          f"Model loaded from {args.checkpoint}", flush=True)

    if _cp_size > 1 and dist.is_initialized() and dist.get_rank() != 0:

        worker_loop()
    else:

        uvicorn.run(app, host=args.host, port=args.port)
