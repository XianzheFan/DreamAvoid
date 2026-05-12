import argparse
import json
import os
import pathlib
import sys
import cv2

def list_episodes(data_dir: pathlib.Path, camera: str, episodes_spec: str | None):
    cam_dir = data_dir / 'videos' / 'chunk-000' / f'observation.images.{camera}'
    if not cam_dir.exists():
        raise FileNotFoundError(f'{cam_dir} not found')
    files = sorted(cam_dir.glob('episode_*.mp4'))
    eps = []
    for f in files:
        try:
            idx = int(f.stem.split('_')[1])
        except Exception:
            continue
        eps.append((idx, str(f)))
    eps.sort()
    if episodes_spec:
        a, b = map(int, episodes_spec.split(':'))
        eps = [(i, p) for i, p in eps if a <= i < b]
    return eps

def load_existing(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

def save_annotations(path: pathlib.Path, ann: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(ann.items(), key=lambda kv: int(kv[0]))
    ordered = {k: v for k, v in items}
    with open(path, 'w') as f:
        json.dump(ordered, f, indent=2)

def render_overlay(frame, ep_idx: int, f_idx: int, n_frames: int, fps: float, hint: str, current_save: int | None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    line1 = f'ep{ep_idx:03d}   f={f_idx}/{n_frames - 1}   t={f_idx / max(fps, 1e-06):.2f}s   fps={fps:.1f}'
    cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    saved_str = f'saved={current_save}' if current_save is not None else 'saved=none'
    cv2.putText(frame, f'{saved_str}    {hint}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    if current_save is not None:
        x = int(w * (current_save / max(n_frames - 1, 1)))
        cv2.line(frame, (x, h - 24), (x, h - 4), (0, 255, 255), 3)
    bar_x = int(w * (f_idx / max(n_frames - 1, 1)))
    cv2.line(frame, (bar_x, h - 12), (bar_x, h - 2), (0, 255, 0), 2)
    return frame

def annotate_one(video_path: str, ep_idx: int, existing: int | None) -> tuple[str, int | None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  [warn] cannot open {video_path}')
        return ('skip', None)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f_idx = existing if existing is not None and 0 <= existing < n else n // 2
    saved = existing
    playing = False
    win = 'annotate_t_crit'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    hint = ',/. j/l u/o 0-9 jump  s save  n skip  b back  r reset  q quit'
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            f_idx = max(0, n - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                break
        view = render_overlay(frame, ep_idx, f_idx, n, fps, hint, saved)
        cv2.imshow(win, view)
        wait_ms = max(1, int(1000.0 / max(fps, 1.0))) if playing else 0
        k = cv2.waitKey(wait_ms) & 255
        if playing and k == 255:
            f_idx = min(n - 1, f_idx + 1)
            if f_idx >= n - 1:
                playing = False
            continue
        if k == ord('q'):
            return ('quit', saved)
        elif k == ord('s'):
            saved = f_idx
            return ('save', saved)
        elif k == ord('n'):
            return ('skip', None)
        elif k == ord('b'):
            return ('back', None)
        elif k == ord('r'):
            f_idx = 0
            playing = False
        elif k == ord(' '):
            playing = not playing
        elif k == ord(','):
            f_idx = max(0, f_idx - 1)
            playing = False
        elif k == ord('.'):
            f_idx = min(n - 1, f_idx + 1)
            playing = False
        elif k == ord('j'):
            f_idx = max(0, f_idx - 10)
            playing = False
        elif k == ord('l'):
            f_idx = min(n - 1, f_idx + 10)
            playing = False
        elif k == ord('u'):
            f_idx = max(0, f_idx - 30)
            playing = False
        elif k == ord('o'):
            f_idx = min(n - 1, f_idx + 30)
            playing = False
        elif k == ord('h'):
            f_idx = max(0, f_idx - 100)
            playing = False
        elif k == ord(';'):
            f_idx = min(n - 1, f_idx + 100)
            playing = False
        elif ord('0') <= k <= ord('9'):
            frac = (k - ord('0')) / 10.0
            f_idx = int(round((n - 1) * frac))
            playing = False
    cap.release()
    return ('skip', None)

def main():
    p = argparse.ArgumentParser(description='Annotate t_crit frame indices for the Dream Trigger.')
    p.add_argument('--data_dir', required=True, help='LeRobot v2.1 dataset root (contains videos/chunk-000/...)')
    p.add_argument('--camera', default='cam_high', choices=['cam_high', 'cam_left_wrist', 'cam_right_wrist'])
    p.add_argument('--episodes', default=None, help="Episode range, e.g. '0:50'. Default: all.")
    p.add_argument('--output', required=True, help='Path to t_crit JSON (created/updated)')
    p.add_argument('--redo', action='store_true', help='Re-annotate episodes already present in --output')
    args = p.parse_args()
    data_dir = pathlib.Path(args.data_dir)
    out_path = pathlib.Path(args.output)
    eps = list_episodes(data_dir, args.camera, args.episodes)
    if not eps:
        print(f'No episodes found under {data_dir} (camera={args.camera}, range={args.episodes})')
        sys.exit(1)
    annotations = load_existing(out_path)
    print(f'Loaded {len(annotations)} existing annotations from {out_path}')
    print(f'Episodes to consider: {len(eps)} (range={args.episodes or 'all'})')
    i = 0
    while i < len(eps):
        ep_idx, vpath = eps[i]
        key = str(ep_idx)
        if not args.redo and key in annotations:
            print(f'  ep{ep_idx:03d}: already annotated -> {annotations[key]}, skip')
            i += 1
            continue
        existing = None
        if key in annotations:
            v = annotations[key]
            existing = int(v) if isinstance(v, (int, float)) else int(v.get('t_crit'))
        print(f'\n--- ep{ep_idx:03d} ({i + 1}/{len(eps)}) {os.path.basename(vpath)} ---')
        action, frame_idx = annotate_one(vpath, ep_idx, existing)
        if action == 'save' and frame_idx is not None:
            annotations[key] = int(frame_idx)
            save_annotations(out_path, annotations)
            print(f'  ep{ep_idx:03d}: t_crit={frame_idx}  -> {out_path}')
            i += 1
        elif action == 'skip':
            print(f'  ep{ep_idx:03d}: skipped')
            i += 1
        elif action == 'back':
            i = max(0, i - 1)
            if str(eps[i][0]) in annotations and (not args.redo):
                annotations.pop(str(eps[i][0]), None)
                save_annotations(out_path, annotations)
                print(f'  going back to ep{eps[i][0]:03d} (cleared its save)')
        elif action == 'quit':
            if frame_idx is not None:
                annotations[key] = int(frame_idx)
            save_annotations(out_path, annotations)
            print(f'\nQuit. {len(annotations)} annotations saved to {out_path}')
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    save_annotations(out_path, annotations)
    print(f'\nDone. {len(annotations)} annotations saved to {out_path}')
if __name__ == '__main__':
    main()
