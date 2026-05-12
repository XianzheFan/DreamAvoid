import argparse
import collections
import concurrent.futures
import json
import logging
import os
import pathlib
import shutil
import signal
import sys
import termios
import threading
import time
import tty
import cv2
import imageio
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from agilex_utils import drain_keyboard_events, get_config, handle_interactive_mode, process_action
from ros_operator import RosOperator, get_ros_observation
from agilex_infer_dinov2_value_switch import ROLLOUT_FPS, SwitchClipBuffer, _dreamdojo_generate
observation_window = None
observation_window_lock = threading.Lock()
shutdown_event = threading.Event()


class StandaloneDreamTrigger:

    def __init__(self, checkpoint_path: str, dinov2_model: str = 'dinov2_vitb14', state_dim: int = 14, num_cameras: int = 3, max_clip_frames: int = 20, attn_dim: int = 384, attn_heads: int = 4, attn_layers: int = 2, hidden_dim: int = 256, dropout: float = 0.0, device: str | None = None):
        from train_dream_trigger import DreamTrigger
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = DreamTrigger(dinov2_model=dinov2_model, state_dim=state_dim, num_cameras=num_cameras, max_clip_frames=max_clip_frames, attn_dim=attn_dim, attn_heads=attn_heads, attn_layers=attn_layers, hidden_dim=hidden_dim, dropout=dropout, freeze_backbone=True)
        sd = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

    def _frame_to_tensor(self, frame_hwc: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(frame_hwc)).permute(2, 0, 1).float() / 255.0

    @torch.no_grad()
    def predict(self, top, right, left, state_np: np.ndarray) -> float:
        images = []
        for cam in (top, right, left):
            if isinstance(cam, list):
                t = torch.stack([self._frame_to_tensor(f) for f in cam]).unsqueeze(0)
            else:
                t = self._frame_to_tensor(cam).unsqueeze(0)
            images.append(t.to(self.device))
        state_t = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        prob = self.model.predict_prob(images, state_t)
        return float(prob[0].item())


class DreamTriggerChecker:

    def __init__(self, gamma: float = 0.5, warmup_tf: float = 0.0):
        self.gamma = float(gamma)
        self.warmup_tf = float(warmup_tf)
        self.history: list[tuple[float, float]] = []

    def reset(self):
        self.history.clear()

    def add_and_check(self, p_t: float, time_fraction: float) -> tuple[bool, dict]:
        p_t = float(p_t)
        self.history.append((p_t, float(time_fraction)))
        triggered = p_t >= self.gamma and time_fraction >= self.warmup_tf
        info = {'p_t': p_t, 'time_fraction': float(time_fraction), 'gamma': self.gamma, 'warmup_tf': self.warmup_tf}
        return (bool(triggered), info)


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        import rospy
        rospy.signal_shutdown('SIGINT')
    except Exception:
        pass


def reset_observation_window():
    global observation_window
    with observation_window_lock:
        observation_window = None


def update_observation_window(args, config, ros_operator):
    global observation_window
    with observation_window_lock:
        if observation_window is None:
            observation_window = collections.deque(maxlen=2)
            observation_window.append({'qpos': None, 'images': {config['camera_names'][0]: None, config['camera_names'][1]: None, config['camera_names'][2]: None}, 'eef_pose': None})
    img_front, img_left, img_right, follower_arm_left, follower_arm_right, follower_arm_left_pose, follower_arm_right_pose = get_ros_observation(args, ros_operator)
    qpos = np.concatenate((np.array(follower_arm_left.position), np.array(follower_arm_right.position)), axis=0)
    eef_pose = ros_operator.build_follower_arm_pose(follower_arm_left_pose, follower_arm_right_pose, follower_arm_left, follower_arm_right)
    with observation_window_lock:
        observation_window.append({'qpos': qpos, 'images': {config['camera_names'][0]: img_front, config['camera_names'][1]: img_right, config['camera_names'][2]: img_left}, 'eef_pose': eef_pose})


def _get_obs_snapshot(args, config):
    with observation_window_lock:
        image_arrs = [observation_window[-1]['images'][config['camera_names'][0]], observation_window[-1]['images'][config['camera_names'][1]], observation_window[-1]['images'][config['camera_names'][2]]]
        if args.ctrl_type == 'joint':
            state = observation_window[-1]['qpos']
        elif args.ctrl_type == 'eef':
            state = observation_window[-1]['eef_pose']
        else:
            raise ValueError(f'Unknown ctrl_type: {args.ctrl_type}')
    return {'top': image_arrs[0], 'right': image_arrs[1], 'left': image_arrs[2], 'instruction': config['language_instruction'], 'state': state, 'action_prefix': None, 'delay': None}


def inference_fn_sync(args, config, policy, ros_operator):
    update_observation_window(args, config, ros_operator)
    start_time = time.perf_counter()
    payload = _get_obs_snapshot(args, config)
    actions = policy.predict_action(payload)
    elapsed = (time.perf_counter() - start_time) * 1000
    print(f'[Sync] Model inference: {elapsed:.1f}ms')
    return np.asarray(actions)


def _select_best_action_with_prefix(obs_snapshot: dict, sde_policy: 'OpenpiClient', exec_horizon: int, task_description: str, step_save_dir: pathlib.Path, dd_host: str, dd_base_port: int, num_samples: int, prefix_action: 'np.ndarray | None', skip_steps: int, dd_action_stride: int = 4, dd_action_chunk_in: int = 13) -> tuple[np.ndarray, dict]:
    step_save_dir.mkdir(parents=True, exist_ok=True)
    raw_chunks = [np.asarray(sde_policy.predict_action(obs_snapshot)) for _ in range(num_samples)]
    skip = int(max(0, skip_steps))
    action_chunks: list[np.ndarray] = []
    for ch in raw_chunks:
        ch = np.array(ch, copy=True)
        if prefix_action is not None and skip > 0:
            k = min(skip, ch.shape[0])
            ch[:k] = np.asarray(prefix_action, dtype=ch.dtype)
        action_chunks.append(ch)
    frame_img = obs_snapshot['top']
    frame_left_img = obs_snapshot.get('left')
    frame_right_img = obs_snapshot.get('right')
    tail_parts = step_save_dir.parts[-3:]
    save_prefix = '/'.join(tail_parts) if tail_parts else step_save_dir.name
    s = max(1, int(dd_action_stride))
    n_in = max(1, int(dd_action_chunk_in))
    tasks = [{'host': dd_host, 'port': dd_base_port + i, 'actions': np.asarray(action_chunks[i][:exec_horizon][::s][:n_in], dtype=np.float32), 'save_name': f'{save_prefix}/chunk_{i}'} for i in range(num_samples)]
    logging.info(f'[DreamTrigger] Launching {num_samples} DreamDojo gens+scores (prefix_skip={skip}, horizon={exec_horizon}, dd_stride={s}, dd_n_in={n_in}, sent_actions={(tasks[0]['actions'].shape[0] if tasks else 0)})...')
    for i, ch in enumerate(action_chunks):
        raw = np.asarray(ch[:exec_horizon], dtype=np.float32)
        sub = np.asarray(tasks[i]['actions'], dtype=np.float32)
        if raw.shape[0] >= 2:
            raw_d = np.linalg.norm(np.diff(raw, axis=0), axis=1)
            raw_stats = f'mean={raw_d.mean():.4f} max={raw_d.max():.4f} min={raw_d.min():.4f} sum={raw_d.sum():.4f}'
        else:
            raw_stats = 'n<2'
        if sub.shape[0] >= 2:
            sub_d = np.linalg.norm(np.diff(sub, axis=0), axis=1)
            sub_stats = f'mean={sub_d.mean():.4f} max={sub_d.max():.4f} min={sub_d.min():.4f} sum={sub_d.sum():.4f}'
        else:
            sub_stats = 'n<2'
        logging.info(f'[DreamTrigger][cand={i}] raw_chunk(N={raw.shape[0]}) Δ-norm: {raw_stats} | sent_to_dd(N={sub.shape[0]}, stride={s}) Δ-norm: {sub_stats}')

    def _submit(task):
        return _dreamdojo_generate(task['host'], task['port'], frame_img, task['actions'], task['save_name'], task_description, frame_left_np=frame_left_img, frame_right_np=frame_right_img)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_samples)) as ex:
        futures = {ex.submit(_submit, t): i for i, t in enumerate(tasks)}
        results_dict: dict[int, dict] = {}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results_dict[idx] = fut.result()
            except Exception as e:
                logging.error(f'[DreamTrigger] Request {idx} failed: {e}')
                results_dict[idx] = None
    candidate_scores: dict[int, float] = {}
    local_valid: list[tuple[int, str]] = []
    for orig_i in range(num_samples):
        res = results_dict.get(orig_i)
        if res and res.get('score') is not None:
            score = float(res['score'])
            candidate_scores[orig_i] = score
            logging.info(f'[ServerScore] Candidate {orig_i} evaluated by server: score={score:.4f}')
            orig_path = res.get('save_path')
            if orig_path and os.path.exists(orig_path):
                dst = step_save_dir / f'output_{orig_i}.mp4'
                try:
                    shutil.copy2(orig_path, dst)
                    local_valid.append((orig_i, str(dst)))
                except Exception as e:
                    logging.warning(f'[DreamTrigger] Could not copy {orig_path} -> {dst}: {e}')
                    local_valid.append((orig_i, orig_path))
        else:
            logging.warning(f'[DreamTrigger] Candidate {orig_i} failed or returned no score.')
    if not candidate_scores:
        logging.warning('[DreamTrigger] All video generations/scoring failed; using chunk 0.')
        best_idx = 0
    else:
        best_idx = max(candidate_scores, key=candidate_scores.get)
    selection_record = {'num_candidates': len(candidate_scores), 'candidate_scores': {str(k): v for k, v in candidate_scores.items()}, 'best_idx': int(best_idx), 'best_score': candidate_scores.get(best_idx, -float('inf')), 'prefix_skip_steps': skip}
    selection_record['all_candidates'] = [np.asarray(ch[:exec_horizon], dtype=np.float32) for ch in action_chunks]
    selection_record['all_candidate_scores'] = [candidate_scores.get(i, None) for i in range(num_samples)]
    logging.info(f'[DreamTrigger] Selected candidate {best_idx} (score={candidate_scores.get(best_idx, 'N/A')}, prefix_skip={skip})')
    return (np.asarray(action_chunks[best_idx][:exec_horizon]), selection_record)


def model_inference(args, config, ros_operator):
    import rospy
    if args.combined:
        sde_host = args.sde_host or args.host
        sde_port = args.sde_port if args.sde_host else args.port
        policy = OpenpiClient(host=args.host, port=args.port, mode='ode')
        sde_policy = OpenpiClient(host=sde_host, port=sde_port, mode='sde')
        logging.info(f'Combined policy server: {args.host}:{args.port} (ODE+SDE share weights)')
    else:
        policy = OpenpiClient(host=args.host, port=args.port)
        sde_policy = None
        if args.sde_host:
            sde_policy = OpenpiClient(host=args.sde_host, port=args.sde_port)
            logging.info(f'SDE policy connected to {args.sde_host}:{args.sde_port}')
    if not args.dream_trigger_ckpt or not os.path.exists(args.dream_trigger_ckpt):
        raise FileNotFoundError(f'--dream_trigger_ckpt is required and must exist (got: {args.dream_trigger_ckpt})')
    dream_trigger = StandaloneDreamTrigger(checkpoint_path=args.dream_trigger_ckpt, dinov2_model=args.dt_dinov2_model, state_dim=14, num_cameras=3, max_clip_frames=args.dt_max_clip_frames, attn_dim=args.dt_attn_dim, attn_heads=args.dt_attn_heads, attn_layers=args.dt_attn_layers, hidden_dim=args.dt_hidden_dim, dropout=0.0)
    logging.info(f'Dream Trigger loaded from {args.dream_trigger_ckpt} (K={args.dt_clip_len}, gamma={args.gamma})')
    can_switch = sde_policy is not None
    if not can_switch:
        logging.warning('SDE policy missing; Dream Trigger will still report scores but rescue actions will not be executed.')
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']
    left0 = config['left0']
    right0 = config['right0']
    task_description = config['language_instruction']
    print(config)
    rescue_check_stride = max(1, int(round(args.publish_rate * args.rescue_check_interval_sec)))
    rescue_cooldown_steps = max(rescue_check_stride, int(round(chunk_size * args.rescue_cooldown_frac)))
    logging.info(f'[DreamTrigger] check every {rescue_check_stride} publish steps (~{args.rescue_check_interval_sec:.2f}s @ {args.publish_rate} Hz), cooldown {rescue_cooldown_steps} steps.')
    ros_operator.follower_arm_publish_continuous(left0, right0)
    print('Warmup servers...')
    policy.warmup()
    if sde_policy:
        sde_policy.warmup(rtc=False, streaming=False)
    print('Servers warmed up')
    run_stamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = pathlib.Path(args.video_out_path) / f'run_{run_stamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Output dir for this run: {output_dir}')
    try:
        episode_idx = 0
        while not rospy.is_shutdown():
            rate = rospy.Rate(args.publish_rate)
            reset_observation_window()
            input('Press enter to start episode')
            task_time = time.time()
            ros_operator.follower_arm_publish_continuous(left0, right0)
            rescue_log: list = []
            trigger_log: list = []
            value_selections: list = []
            collected_frames: list = []
            frame_counter = 0
            clip_buffer = SwitchClipBuffer(args.dt_clip_len)
            checker = DreamTriggerChecker(gamma=args.gamma, warmup_tf=args.warmup_tf)
            task_segment = task_description.replace(' ', '_')
            rollout_dir = output_dir / f'rollout_{task_segment}_ep{episode_idx}_running'
            rollout_dir.mkdir(parents=True, exist_ok=True)
            action_buffer = np.zeros([chunk_size, config['state_dim']])
            action_buffer_base_t = -chunk_size
            action_buffer_source = 'ode'
            last_rescue_check_t = -rescue_check_stride
            last_rescue_trigger_t = -10 ** 9
            last_published_action: np.ndarray | None = None
            latest_p: float = 0.0
            latest_tf: float = 0.0
            latest_valid: bool = False
            latest_manual: bool = False
            rescue_active_until_frame: int = -1
            user_stopped = False
            t = 0
            sde_chunk_log: list = []
            pending_rescue: dict | None = None
            manual_rescue_remaining: int = 0
            manual_chain_gap_steps = max(0, int(round(args.manual_rescue_chain_pause_sec * args.publish_rate)))
            while t < max_publish_step and (not rospy.is_shutdown()) and (not shutdown_event.is_set()):
                manual_rescue_pressed = False
                events = drain_keyboard_events()
                for key in events:
                    if key == ' ':
                        result = handle_interactive_mode(task_time)
                        if result == 'reset':
                            ros_operator.follower_arm_publish_continuous(left0, right0)
                            user_stopped = True
                            break
                        elif result == 'quit':
                            user_stopped = True
                            return
                    elif key == 's':
                        manual_rescue_pressed = True
                        manual_rescue_remaining = max(0, int(args.manual_rescue_repeat) - 1)
                        if manual_rescue_remaining > 0:
                            print(f"\n\x1b[93m>>> [System] Key 's' detected! Manually triggering SDE + DreamDojo rescue mechanism! (will chain {manual_rescue_remaining} more rescue(s) after this one)\x1b[0m", flush=True)
                        else:
                            print("\n\x1b[93m>>> [System] Key 's' detected! Manually triggering SDE + DreamDojo rescue mechanism!\x1b[0m", flush=True)
                if user_stopped:
                    break
                if not manual_rescue_pressed and manual_rescue_remaining > 0 and (pending_rescue is None) and (frame_counter > rescue_active_until_frame + manual_chain_gap_steps):
                    manual_rescue_pressed = True
                    manual_rescue_remaining -= 1
                    print(f'\n\x1b[93m>>> [System] Auto-chaining manual rescue ({manual_rescue_remaining} more after this)\x1b[0m', flush=True)
                front_msg = ros_operator.front_image_queue[-1] if len(ros_operator.front_image_queue) > 0 else None
                if front_msg is not None:
                    front_img = ros_operator.bridge.imgmsg_to_cv2(front_msg, 'passthrough')
                    frame_counter += 1
                    rescue_active_now = frame_counter <= rescue_active_until_frame
                    hud_frame = np.ascontiguousarray(np.asarray(front_img).copy())
                    if latest_valid:
                        _draw_dream_trigger_hud(hud_frame, latest_p, latest_tf, args.gamma, rescue_active_now, latest_manual)
                    if args.show_live_window:
                        try:
                            cv2.imshow('dream_trigger', hud_frame[..., ::-1])
                            cv2.waitKey(1)
                        except Exception as e:
                            logging.warning(f'[HUD] cv2.imshow failed: {e}')
                    collected_frames.append(hud_frame)
                    right_msg = ros_operator.right_image_queue[-1] if len(ros_operator.right_image_queue) > 0 else None
                    left_msg = ros_operator.left_image_queue[-1] if len(ros_operator.left_image_queue) > 0 else None
                    if right_msg is not None and left_msg is not None:
                        right_img = ros_operator.bridge.imgmsg_to_cv2(right_msg, 'passthrough')
                        left_img = ros_operator.bridge.imgmsg_to_cv2(left_msg, 'passthrough')
                        clip_buffer.update(front_img, right_img, left_img)
                rescue_triggered = False
                rescue_in_flight = pending_rescue is not None and (not pending_rescue['future'].done())
                should_check_auto = can_switch and (not rescue_in_flight) and (frame_counter > 0) and (t - last_rescue_check_t >= rescue_check_stride) and (t - last_rescue_trigger_t >= rescue_cooldown_steps)
                if rescue_in_flight and manual_rescue_pressed:
                    print("\n\x1b[93m>>> [Rescue] DreamDojo still running from a previous trigger — 's' ignored.\x1b[0m", flush=True)
                    manual_rescue_pressed = False
                if can_switch and (should_check_auto or manual_rescue_pressed):
                    if should_check_auto:
                        last_rescue_check_t = t
                    update_observation_window(args, config, ros_operator)
                    obs_snapshot = _get_obs_snapshot(args, config)
                    with observation_window_lock:
                        cur_imgs = observation_window[-1]['images']
                        front_img_now = cur_imgs.get(config['camera_names'][0])
                        right_img_now = cur_imgs.get(config['camera_names'][1])
                        left_img_now = cur_imgs.get(config['camera_names'][2])
                        switch_state = observation_window[-1]['qpos']
                    if front_img_now is not None:
                        clip_buffer.update(front_img_now, right_img_now, left_img_now)
                    if clip_buffer.ready():
                        top_c, right_c, left_c = clip_buffer.clips()
                        p_t = dream_trigger.predict(top_c, right_c, left_c, switch_state)
                    else:
                        p_t = dream_trigger.predict(obs_snapshot['top'], obs_snapshot['right'], obs_snapshot['left'], switch_state)
                    time_fraction = t / max(max_publish_step - 1, 1)
                    auto_triggered, info = checker.add_and_check(p_t, time_fraction)
                    triggered = auto_triggered or manual_rescue_pressed
                    reasons: list = []
                    if auto_triggered:
                        reasons.append(f'p_ge_gamma({p_t:.3f}>={args.gamma:.2f})')
                    if manual_rescue_pressed:
                        reasons.append('manual_keyboard_override')
                    trigger_log.append({'t': t, 'frame': frame_counter, 'p_t': p_t, 'time_fraction': time_fraction, 'gamma': args.gamma, 'triggered': triggered, 'reasons': reasons})
                    latest_p = float(p_t)
                    latest_tf = float(time_fraction)
                    latest_valid = True
                    latest_manual = bool(manual_rescue_pressed)
                    logging.info(f'[DreamTrigger t={t:4d} f={frame_counter:4d}] p={p_t:.3f} gamma={args.gamma:.2f} tf={time_fraction:.2f} {('TRIGGER' if triggered else 'ok')}{(' ' + ','.join(reasons) if triggered else '')}')
                    if triggered:
                        step_save_dir = rollout_dir / 'rescue_steps' / f't{t}_f{frame_counter}'
                        reasons_str = ','.join(reasons)
                        print(f'\n\x1b[93m>>> [System] Dream Trigger fired (p={p_t:.3f} >= gamma={args.gamma:.2f})!\x1b[0m', flush=True)
                        print(f'\x1b[93m>>> [System] Reasons: {reasons_str}\x1b[0m', flush=True)
                        print(f'\x1b[93m>>> [System] DreamDojo running SYNC ({args.num_sde_samples} candidates). Arm pausing until candidates scored...\x1b[0m\n', flush=True)
                        future = concurrent.futures.Future()
                        try:
                            _sync_result = _select_best_action_with_prefix(obs_snapshot=obs_snapshot, sde_policy=sde_policy, exec_horizon=chunk_size, task_description=task_description, step_save_dir=step_save_dir, dd_host=args.dd_host, dd_base_port=args.dd_base_port, num_samples=args.num_sde_samples, prefix_action=last_published_action, skip_steps=args.rescue_skip_sde_steps, dd_action_stride=args.dd_action_stride, dd_action_chunk_in=args.dd_action_chunk_in)
                            future.set_result(_sync_result)
                        except Exception as _sync_err:
                            future.set_exception(_sync_err)
                        pending_rescue = {'future': future, 't_submit': t, 'frame_submit': frame_counter, 'p_t': float(p_t), 'reasons': list(reasons)}
                        last_rescue_trigger_t = t
                if pending_rescue is not None and pending_rescue['future'].done():
                    pr = pending_rescue
                    pending_rescue = None
                    try:
                        best_actions, sel_record = pr['future'].result()
                    except Exception as e:
                        logging.error(f'[Rescue] async DreamDojo failed: {e}')
                        best_actions, sel_record = (None, None)
                    if best_actions is not None and sel_record is not None:
                        best_score = sel_record.get('best_score', 'N/A')
                        if isinstance(best_score, float):
                            best_score_str = f'{best_score:.4f}'
                        else:
                            best_score_str = str(best_score)
                        wait_steps = t - pr['t_submit']
                        print(f'\n\x1b[92m>>> [System] DreamDojo done after {wait_steps} publish steps (best_idx={sel_record.get('best_idx')}, score={best_score_str}). Injecting now.\x1b[0m\n', flush=True)
                        sel_record['t'] = int(t)
                        sel_record['frame'] = int(frame_counter)
                        sel_record['t_submit'] = int(pr['t_submit'])
                        sel_record['frame_submit'] = int(pr['frame_submit'])
                        sel_record['p_t'] = pr['p_t']
                        sel_record['reasons'] = pr['reasons']
                        json_safe_sel = {k: v for k, v in sel_record.items() if k not in ('all_candidates',)}
                        value_selections.append(json_safe_sel)
                        rescue_log.append({'t': int(t), 'frame': int(frame_counter), 't_submit': int(pr['t_submit']), 'p_t': pr['p_t'], 'reasons': pr['reasons']})
                        best_actions = np.asarray(best_actions)
                        L = min(best_actions.shape[0], chunk_size)
                        action_buffer = np.zeros_like(action_buffer)
                        best_actions = best_actions.astype(action_buffer.dtype, copy=False)
                        K = int(max(0, args.rescue_blend_steps))
                        can_blend = K > 0 and last_published_action is not None and (L > 0) and (last_published_action.shape == best_actions[0].shape)
                        if can_blend:
                            offset = last_published_action.astype(action_buffer.dtype) - best_actions[0]
                            ramp = np.maximum(0.0, 1.0 - np.arange(L, dtype=np.float64) / float(K))
                            action_buffer[:L] = best_actions[:L] + offset[None, :] * ramp[:, None]
                            max_jump = float(np.max(np.abs(action_buffer[0] - last_published_action)))
                            logging.info(f'[Rescue] Blended first {min(K, L)} steps (max |Δ| step0 -> last = {max_jump:.4f})')
                        else:
                            action_buffer[:L] = best_actions[:L]
                        action_buffer_base_t = t
                        action_buffer_source = 'sde'
                        rescue_active_until_frame = frame_counter + chunk_size - 1
                        sde_chunk_log.append({'t_start': int(t), 'frame_start': int(frame_counter), 'actions': np.asarray(action_buffer[:L]).copy(), 'best_idx': int(sel_record.get('best_idx', -1)), 'best_score': sel_record.get('best_score', None), 'all_candidates': [np.asarray(c, dtype=np.float32) for c in sel_record.get('all_candidates', [])], 'all_candidate_scores': list(sel_record.get('all_candidate_scores', [])), 'reasons': list(pr['reasons']), 'p_t': float(pr['p_t'])})
                        rescue_triggered = True
                        logging.info(f'[Rescue] Injected {L} SDE actions at t={t} (submitted at t={pr['t_submit']})')
                if not rescue_triggered and (t - action_buffer_base_t >= chunk_size):
                    actions = inference_fn_sync(args, config, policy, ros_operator)
                    assert actions is not None, 'Sync inference returned None'
                    assert actions.shape[0] >= chunk_size, f'Action chunk length {actions.shape[0]} is smaller than {chunk_size}'
                    action_buffer = actions[:chunk_size]
                    action_buffer_base_t = t
                    action_buffer_source = 'ode'
                idx_in_chunk = t - action_buffer_base_t
                idx_in_chunk = max(0, min(idx_in_chunk, chunk_size - 1))
                act = action_buffer[idx_in_chunk]
                if args.ctrl_type == 'joint':
                    left_action, right_action = process_action(config['task'], act)
                    ros_operator.follower_arm_publish(left_action, right_action)
                    last_published_action = np.asarray(act).copy()
                elif args.ctrl_type == 'eef':
                    left_action, right_action = process_action(config['task'], act)
                    ros_operator.follower_arm_pose_publish(left_action, right_action)
                    last_published_action = np.asarray(act).copy()
                t += 1
                tag = action_buffer_source.upper()
                print(f'[Step {t:4d}] Published [{tag}] (buf_idx={idx_in_chunk}/{chunk_size})')
                rate.sleep()
            if pending_rescue is not None:
                pending_rescue['future'].cancel()
                pending_rescue = None
            suffix = 'stopped' if user_stopped else 'done'
            base_name = f'rollout_{task_segment}_ep{episode_idx}_{suffix}'
            final_dir = output_dir / base_name
            dedup_idx = 1
            while final_dir.exists():
                final_dir = output_dir / f'{base_name}_{dedup_idx}'
                dedup_idx += 1
            shutil.move(str(rollout_dir), str(final_dir))
            _write_results(final_dir, task_description, suffix, rescue_log, trigger_log, value_selections)
            if collected_frames:
                annotated = _annotate_rescue_frames(collected_frames, rescue_log, window=chunk_size)
                imageio.mimwrite(str(final_dir / 'complete_video.mp4'), annotated, fps=ROLLOUT_FPS)
            logging.info(f'Episode {episode_idx} finished: {suffix}')
            logging.info(f'Trigger activations: {len(rescue_log)}')
            episode_idx += 1
            ros_operator.follower_arm_publish_continuous(left0, right0)
    finally:
        ros_operator.follower_arm_publish_continuous(left0, right0)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _draw_dream_trigger_hud(img: np.ndarray, p_t: float, time_fraction: float, gamma: float, rescue_active: bool, is_manual: bool) -> np.ndarray:
    h, _w = img.shape[:2]
    if is_manual and rescue_active:
        main_color = (0, 165, 255)
    elif rescue_active:
        main_color = (0, 0, 255)
    elif p_t >= gamma:
        main_color = (0, 0, 255)
    else:
        main_color = (0, 220, 0)
    main_txt = f'p={p_t:.2f}  gamma={gamma:.2f}  tf={time_fraction:.2f}'
    y = h - 16
    cv2.putText(img, main_txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, main_txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, main_color, 2, cv2.LINE_AA)
    sub = ''
    if rescue_active:
        sub = 'MANUAL OVERRIDE' if is_manual else 'DREAM TRIGGER'
    elif p_t >= gamma:
        sub = 'ABOVE THRESHOLD'
    if sub:
        cv2.putText(img, sub, (12, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, sub, (12, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, main_color, 1, cv2.LINE_AA)
    return img


def _annotate_rescue_frames(frames: list, rescue_log: list, window: int) -> list:
    if not rescue_log:
        return [np.asarray(f) for f in frames]
    events = sorted(rescue_log, key=lambda e: int(e['frame']))
    out = []
    ev_idx = 0
    active_until = -1
    active_p = 0.0
    active_reasons: list = []
    for i, frm in enumerate(frames):
        frame_no = i + 1
        while ev_idx < len(events) and int(events[ev_idx]['frame']) <= frame_no:
            active_until = int(events[ev_idx]['frame']) + window - 1
            active_p = float(events[ev_idx].get('p_t', 0.0))
            active_reasons = list(events[ev_idx].get('reasons', []))
            ev_idx += 1
        img = np.ascontiguousarray(np.asarray(frm))
        if frame_no <= active_until:
            h, w = img.shape[:2]
            is_manual = 'manual_keyboard_override' in active_reasons
            color = (0, 165, 255) if is_manual else (0, 0, 255)
            prefix = 'MANUAL OVERRIDE' if is_manual else 'DREAM TRIGGER'
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 4)
            txt = f'{prefix} p={active_p:.2f}'
            cv2.putText(img, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            if active_reasons:
                display_reasons = [r for r in active_reasons if r != 'manual_keyboard_override']
                sub = ','.join(display_reasons) if display_reasons else 'Triggered by Keyboard'
                cv2.putText(img, sub, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, sub, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
        out.append(img)
    return out


def _write_results(rollout_dir: pathlib.Path, task_description: str, suffix: str, rescue_log: list, trigger_log: list, value_selections: list):
    json_path = rollout_dir / 'dream_trigger_results.json'
    with open(json_path, 'w') as f:
        json.dump({'task': task_description, 'outcome': suffix, 'rescue_activations': rescue_log, 'dream_trigger_trace': trigger_log, 'value_expert_selections': value_selections}, f, indent=2)
    logging.info(f'Results written to {json_path}')
    txt_path = rollout_dir / 'dream_trigger_results.txt'
    with open(txt_path, 'w') as f:
        f.write(f'Task: {task_description}\nOutcome: {suffix}\n')
        f.write(f'Trigger activations: {len(rescue_log)}\n{'=' * 60}\n\n')
        for e in rescue_log:
            f.write(f'  t={e['t']} frame={e['frame']}  p={e['p_t']:.3f}  reasons={','.join(e['reasons'])}\n')
        f.write(f'\nDream Trigger samples: {len(trigger_log)}\n{'=' * 60}\n\n')
        for e in trigger_log:
            f.write(f'  t={e['t']:4d} f={e['frame']:4d}  p={e['p_t']:.3f} gamma={e['gamma']:.2f} tf={e['time_fraction']:.2f}  {('TRIGGER' if e['triggered'] else 'ok')}{(' ' + ','.join(e['reasons']) if e['triggered'] else '')}\n')
        f.write(f'\nValue expert selections: {len(value_selections)}\n{'=' * 60}\n\n')
        for s in value_selections:
            f.write(f'  t={s.get('t')} frame={s.get('frame')}  best_idx={s['best_idx']}  best_score={s.get('best_score', 'N/A')}\n    scores={s.get('candidate_scores', {})}\n')


def get_arguments():
    parser = argparse.ArgumentParser(description='Agilex SYNC inference: Dream Trigger + DreamDojo')
    parser.add_argument('--max_publish_step', type=int, default=10000)
    parser.add_argument('--img_front_topic', type=str, default='/camera_f/color/image_raw')
    parser.add_argument('--img_left_topic', type=str, default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic', type=str, default='/camera_r/color/image_raw')
    parser.add_argument('--img_front_depth_topic', type=str, default='/camera_f/depth/image_raw')
    parser.add_argument('--img_left_depth_topic', type=str, default='/camera_l/depth/image_raw')
    parser.add_argument('--img_right_depth_topic', type=str, default='/camera_r/depth/image_raw')
    parser.add_argument('--leader_arm_left_topic', type=str, default='/leader/joint_left')
    parser.add_argument('--leader_arm_right_topic', type=str, default='/leader/joint_right')
    parser.add_argument('--follower_arm_left_topic', type=str, default='/follower/joint_left')
    parser.add_argument('--follower_arm_right_topic', type=str, default='/follower/joint_right')
    parser.add_argument('--pos_cmd_left_topic', type=str, default='/follower/pos_cmd_left')
    parser.add_argument('--pos_cmd_right_topic', type=str, default='/follower/pos_cmd_right')
    parser.add_argument('--follower_arm_left_pose_topic', type=str, default='/follower/end_pose_euler_left')
    parser.add_argument('--follower_arm_right_pose_topic', type=str, default='/follower/end_pose_euler_right')
    parser.add_argument('--publish_rate', type=int, default=30)
    parser.add_argument('--chunk_size', type=int, default=50)
    parser.add_argument('--arm_steps_length', type=float, nargs=7, default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2])
    parser.add_argument('--use_depth_image', action='store_true', default=False)
    parser.add_argument('--ctrl_type', type=str, choices=['joint', 'eef'], default='joint')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--video_out_path', type=str, default='data/agilex/output')
    parser.add_argument('--sde_host', type=str, default=None)
    parser.add_argument('--sde_port', type=int, default=8001)
    parser.add_argument('--combined', action='store_true', default=False)
    parser.add_argument('--dream_trigger_ckpt', type=str, required=True, help='DreamTrigger checkpoint (.pt) from train_dream_trigger.py')
    parser.add_argument('--dt_dinov2_model', type=str, default='dinov2_vitb14', choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])
    parser.add_argument('--dt_clip_len', type=int, default=8, help='K, number of recent frames per camera fed to f_phi')
    parser.add_argument('--dt_max_clip_frames', type=int, default=20, help='Architectural max (must match training)')
    parser.add_argument('--dt_attn_dim', type=int, default=384)
    parser.add_argument('--dt_attn_heads', type=int, default=4)
    parser.add_argument('--dt_attn_layers', type=int, default=2)
    parser.add_argument('--dt_hidden_dim', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.5, help='Trigger threshold: c_t^test = 1 if p_t >= gamma')
    parser.add_argument('--warmup_tf', type=float, default=0.0, help='Suppress auto trigger when time_fraction < this')
    parser.add_argument('--rescue_check_interval_sec', type=float, default=1.0)
    parser.add_argument('--rescue_cooldown_frac', type=float, default=1.0)
    parser.add_argument('--rescue_blend_steps', type=int, default=5)
    parser.add_argument('--rescue_skip_sde_steps', type=int, default=0)
    parser.add_argument('--dd_action_stride', type=int, default=4)
    parser.add_argument('--dd_action_chunk_in', type=int, default=13)
    parser.add_argument('--show_live_window', action='store_true', default=False)
    parser.add_argument('--dd_host', type=str, default='127.0.0.1')
    parser.add_argument('--dd_base_port', type=int, default=8020)
    parser.add_argument('--num_sde_samples', type=int, default=4)
    parser.add_argument('--manual_rescue_repeat', type=int, default=2)
    parser.add_argument('--manual_rescue_chain_pause_sec', type=float, default=0.0)
    return parser.parse_args()


def main():
    args = get_arguments()
    import rospy
    ros_operator = RosOperator(args, mode='inference')
    config = get_config(args)
    signal.signal(signal.SIGINT, _on_sigint)
    logging.basicConfig(level=logging.INFO)
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == '__main__':
    main()
