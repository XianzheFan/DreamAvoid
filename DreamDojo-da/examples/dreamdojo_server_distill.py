import argparse
import base64
import json
import os
import threading
import time
from pathlib import Path

import mediapy
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

try:
    from megatron.core import parallel_state
except Exception:
    class _DummyParallelState:
        def is_initialized(self): return False
        def get_context_parallel_group(self): return None
        def initialize_model_parallel(self, **kwargs): return None
        def destroy_model_parallel(self): return None
    parallel_state = _DummyParallelState()

from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.distill.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2.interactive.datasets.utils import extract_cr1_embedding
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
)

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

app = FastAPI(title="DreamDojo Distill API")
_model = None
_config = None
_t5_emb_gpu: torch.Tensor | None = None
_t5_mask_gpu: torch.Tensor | None = None
_save_dir: Path | None = None
_server_id: str | None = None
_cp_size: int = 1
_fps: int = 8
_num_steps: int = 4
_cache_frame_size: int = 3

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

def _build_cond_frame(frame_np: np.ndarray) -> torch.Tensor:
    img_tensor = torchvision.transforms.functional.to_tensor(frame_np).unsqueeze(0) * 255.0
    img_tensor = img_tensor.to(torch.uint8)
    return img_tensor.unsqueeze(2)

def _tile_3view(top_np: np.ndarray, left_np: np.ndarray, right_np: np.ndarray) -> np.ndarray:
    import cv2
    H, W = 480, 640
    h, w = H // 2, W // 2
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h, :w] = cv2.resize(top_np,   (w, h), interpolation=cv2.INTER_LINEAR)
    canvas[:h, w:] = cv2.resize(left_np,  (w, h), interpolation=cv2.INTER_LINEAR)
    canvas[h:, :w] = cv2.resize(right_np, (w, h), interpolation=cv2.INTER_LINEAR)
    return canvas

def _broadcast_inputs(cond_frame, action, seed, guidance, prompt=""):
    device = torch.device("cuda")
    prompt_bytes = prompt.encode("utf-8") if dist.get_rank() == 0 else b""
    if dist.get_rank() == 0:
        meta = torch.tensor(
            [cond_frame.shape[3], cond_frame.shape[4],
             action.shape[0], action.shape[1], seed,
             int(guidance * 1e6), len(prompt_bytes)],
            dtype=torch.int64, device=device,
        )
    else:
        meta = torch.zeros(7, dtype=torch.int64, device=device)
    dist.broadcast(meta, src=0)

    H            = int(meta[0].item())
    W            = int(meta[1].item())
    action_steps = int(meta[2].item())
    action_dim   = int(meta[3].item())
    seed         = int(meta[4].item())
    guidance     = float(meta[5].item()) / 1e6
    prompt_len   = int(meta[6].item())

    if dist.get_rank() != 0:
        cond_frame = torch.zeros(1, 3, 1, H, W, dtype=torch.uint8, device=device)
        action     = torch.zeros(action_steps, action_dim, dtype=torch.float32, device=device)
    else:
        cond_frame = cond_frame.to(device)
        action     = action.to(device)

    dist.broadcast(cond_frame, src=0)
    dist.broadcast(action,     src=0)

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

    return cond_frame.cpu(), action.cpu(), seed, guidance, prompt

def _prepare_data_batch(video_b_c_t_h_w: torch.Tensor, action_chunk: torch.Tensor, fps: float) -> dict:
    _, _, _, H, W = video_b_c_t_h_w.shape
    data_batch = {
        "dataset_name": "video_data",
        "video": video_b_c_t_h_w,
        "action": action_chunk.float(),
        "fps": torch.tensor([fps], dtype=torch.float32),
        "padding_mask": torch.zeros(1, 1, H, W, dtype=torch.float32),
        NUM_CONDITIONAL_FRAMES_KEY: 0,
    }
    for k, v in list(data_batch.items()):
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
        elif isinstance(v, torch.Tensor):
            data_batch[k] = v.cuda()
    return data_batch

@torch.inference_mode()
def _run_inference(cond_frame_uint8: torch.Tensor, action_t: torch.Tensor, seed: int) -> torch.Tensor:
    chunk_size = _cache_frame_size * 4

    cond_frames = cond_frame_uint8

    first_stack = cond_frames.permute(0, 2, 1, 3, 4)
    zeros_tail = torch.zeros_like(first_stack[:, :1]).repeat(
        1, chunk_size - first_stack.shape[1] + 1, 1, 1, 1
    )
    vid_chw_t = torch.cat([first_stack, zeros_tail], dim=1)
    video_b_c_t_h_w = vid_chw_t.permute(0, 2, 1, 3, 4).contiguous()

    action_chunk = action_t.unsqueeze(0)

    data_batch = _prepare_data_batch(video_b_c_t_h_w, action_chunk, fps=4)
    _model._normalize_video_databatch_inplace(data_batch)
    _model._augment_image_dim_inplace(data_batch)

    data_batch["t5_text_embeddings"] = _t5_emb_gpu
    data_batch["t5_text_mask"] = _t5_mask_gpu

    bf16 = _model.tensor_kwargs["dtype"]
    for k, v in list(data_batch.items()):
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            data_batch[k] = v.to(dtype=bf16)

    _, x0, condition, _ = _model.get_data_and_condition(data_batch)
    x0 = x0.to(dtype=bf16)
    condition = condition.edit_data_type(DataType.VIDEO)
    condition = condition.set_video_condition(
        gt_frames=x0,
        random_min_num_conditional_frames=None,
        random_max_num_conditional_frames=None,
        num_conditional_frames=data_batch[NUM_CONDITIONAL_FRAMES_KEY],
    )

    _T, _H, _W = data_batch[_model.input_data_key].shape[-3:]
    state_shape = (
        _model.config.state_ch,
        int(_model.tokenizer.get_latent_num_frames(_T)),
        _H // _model.tokenizer.spatial_compression_factor,
        _W // _model.tokenizer.spatial_compression_factor,
    )
    noise = misc.arch_invariant_rand(
        (1, *state_shape),
        torch.float32,
        _model.tensor_kwargs["device"],
        seed,
    )

    K = len(_model.config.selected_sampling_time)
    n_steps = max(1, min(int(_num_steps), K))
    start_idx = (cond_frames.shape[2] - 1) // 4 + 1

    latents = _model.generate_streaming_video(
        condition, noise, n_steps=n_steps, cache_frame_size=_cache_frame_size,
        start_idx=start_idx, stateless_kv=False,
    )
    video = _model.decode(latents).clip(min=-1, max=1)
    return video

def worker_loop():
    device = torch.device("cuda")
    while True:
        signal = _SIGNAL_EXIT.clone().to(device)
        dist.broadcast(signal, src=0)
        if signal.item() == _SIGNAL_EXIT.item():
            break
        cond_frame, action_t, seed, guidance, prompt = _broadcast_inputs(
            None, None, None, None
        )
        _run_inference(cond_frame, action_t, seed)

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

    actions = np.array(req.actions, dtype=np.float32)
    print(f"[DEBUG] Received actions shape={actions.shape}, "
          f"abs_mean={np.abs(actions).mean():.4f}, "
          f"min={actions.min():.4f}, max={actions.max():.4f}", flush=True)

    cond_frame_uint8 = _build_cond_frame(cond_np)
    action_t = _preprocess_actions(actions)
    _deltas = action_t[:, _action_slot_start:_action_slot_end].numpy()
    print(f"[DEBUG] Deltas abs_mean={np.abs(_deltas).mean():.6f}, "
          f"abs_max={np.abs(_deltas).max():.6f}", flush=True)

    with _inference_lock:
        if _cp_size > 1:
            signal = _SIGNAL_GENERATE.clone().to("cuda")
            dist.broadcast(signal, src=0)
            cond_frame_uint8, action_t, seed, _g, _p = _broadcast_inputs(
                cond_frame_uint8, action_t, req.seed, req.guidance, req.prompt
            )
        else:
            seed = req.seed

        t0 = time.time()
        video = _run_inference(cond_frame_uint8, action_t, seed)
        torch.cuda.synchronize()
        print(f"[DEBUG] Distill inference took {(time.time()-t0)*1000:.0f}ms", flush=True)

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
        print(f"[ValueModel] In-memory score: {final_score:.4f}", flush=True)

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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",  required=True,
                   help="DCP checkpoint directory (e.g. logs/.../iter_000003000)")
    p.add_argument("--experiment",  required=True,
                   help="Self-forcing/DMD2 experiment name from config_distill.py")
    p.add_argument("--save-dir",    required=True)
    p.add_argument("--config-file", default="cosmos_predict2/_src/predict2/interactive/configs/config_distill.py")
    p.add_argument("--cr1-embeddings-path", default="datasets/cr1_empty_string_text_embeddings.pt")
    p.add_argument("--port",        type=int, default=8000)
    p.add_argument("--host",        default="0.0.0.0")
    p.add_argument("--server-id",   default=None)
    p.add_argument("--context-parallel-size", type=int, default=1)
    p.add_argument("--num-steps",   type=int, default=4,
                   help="Student denoising steps per latent frame (clamped to len(selected_sampling_time))")
    p.add_argument("--cache-frame-size", type=int, default=-1,
                   help="KV cache frame size (-1 reads from config)")
    p.add_argument("--enable-fsdp", action="store_true")
    p.add_argument("--action-slot-start", type=int, default=169)
    p.add_argument("--action-slot-end",   type=int, default=176)
    p.add_argument("--model-action-dim",  type=int, default=384)
    p.add_argument("--model-action-chunk", type=int, default=12)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--value-model-ckpt", type=str, default=None,
                   help="DINOv2 value model checkpoint path. "
                        "Training: openpi-da/agilex/train_value_model.py")
    p.add_argument("--stats-json", type=str, default=None)
    return p.parse_args()

def _init_distributed_for_cp(cp_size: int):
    process_group = None
    if cp_size > 1:
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=cp_size)
        process_group = parallel_state.get_context_parallel_group()
        logger.info(f"Initialized CP with size {cp_size}, "
                    f"rank {distributed.get_rank()}/{distributed.get_world_size()}")
    return process_group

if __name__ == "__main__":
    args = parse_args()

    _save_dir  = Path(args.save_dir)
    _server_id = args.server_id
    _cp_size   = args.context_parallel_size
    _fps       = args.fps
    _num_steps = args.num_steps

    _action_slot_start  = args.action_slot_start
    _action_slot_end    = args.action_slot_end
    _model_action_dim   = args.model_action_dim
    _model_action_chunk = args.model_action_chunk

    if args.stats_json:
        with open(args.stats_json) as f:
            stats = json.load(f)
        _action_min = np.array(stats["action"]["min"], dtype=np.float32)
        _action_max = np.array(stats["action"]["max"], dtype=np.float32)
        print(f"[DreamDojo-distill] Loaded action stats from {args.stats_json}")
        print(f"  action min: {_action_min}")
        print(f"  action max: {_action_max}")
    else:
        print("[DreamDojo-distill] WARNING: No --stats-json, actions will NOT be normalized.")

    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())

    process_group = _init_distributed_for_cp(_cp_size)

    if args.value_model_ckpt:
        rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if rank0:
            print(f"[DreamDojo-distill] Loading value model from {args.value_model_ckpt}...")
            _server_value_model = ServerValueModel(checkpoint_path=args.value_model_ckpt)
            print("[DreamDojo-distill] Value model loaded.")

    _model, _config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=args.checkpoint,
        config_file=args.config_file,
        load_ema_to_reg=True,
        experiment_opts=["ckpt_type=dcp"],
        skip_teacher_init=True,
        enable_fsdp=args.enable_fsdp,
    )
    if _cp_size > 1:
        _model.net.enable_context_parallel(process_group)

    _cache_frame_size = (
        args.cache_frame_size if args.cache_frame_size > 0
        else _config.model.config.cache_frame_size
    )
    print(f"[DreamDojo-distill] cache_frame_size={_cache_frame_size}, num_steps={_num_steps}")

    extract_cr1_embedding(args.cr1_embeddings_path)
    _emb = torch.load(args.cr1_embeddings_path, map_location="cpu")
    if isinstance(_emb, (list, tuple)):
        _emb = _emb[0]
    if not torch.is_tensor(_emb):
        raise ValueError("CR1 embeddings file did not load to a tensor")
    if _emb.dim() == 2:
        _emb = _emb.unsqueeze(0)
    elif _emb.dim() != 3:
        raise ValueError(f"Unexpected CR1 embeddings dim: {_emb.dim()}")
    _t5_emb_gpu = _emb.to(device=_model.tensor_kwargs["device"], dtype=torch.bfloat16)
    _t5_mask_gpu = torch.ones(
        (_t5_emb_gpu.shape[0], _t5_emb_gpu.shape[1]),
        device=_model.tensor_kwargs["device"], dtype=torch.bfloat16,
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank {rank}] Distill model loaded from {args.checkpoint}", flush=True)

    if _cp_size > 1 and dist.is_initialized() and rank != 0:
        worker_loop()
    else:
        uvicorn.run(app, host=args.host, port=args.port)
