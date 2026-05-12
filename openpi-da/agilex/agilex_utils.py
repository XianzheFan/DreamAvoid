import argparse
import os
from pathlib import Path
import select
import sys
import time
import cv2
import dm_env
import numpy as np
import rospy
import yaml
TASK_CONFIG_PATH = Path(__file__).with_name('task_configs.yaml')

def _load_task_configs():
    with TASK_CONFIG_PATH.open('r', encoding='utf-8') as file:
        task_configs = yaml.safe_load(file)
    if not isinstance(task_configs, dict):
        raise ValueError(f'Invalid task config format in {TASK_CONFIG_PATH}')
    return task_configs
TASK_CONFIGS = _load_task_configs()

def get_config(args):
    task_config = TASK_CONFIGS.get(args.task)
    if task_config is None:
        raise ValueError(f'Invalid task name: {args.task}')
    language_instruction = task_config.get('language_instruction')
    left0 = task_config.get('left0')
    right0 = task_config.get('right0')
    if language_instruction is None or left0 is None or right0 is None:
        raise ValueError(f'Task config for {args.task} is missing required fields')
    state_dim = 14
    config = {'episode_len': args.max_publish_step, 'state_dim': state_dim, 'left0': left0, 'right0': right0, 'action_postprocess': task_config.get('action_postprocess', {}), 'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'], 'task': args.task, 'language_instruction': language_instruction, 'ctrl_type': args.ctrl_type, 'chunk_size': args.chunk_size, 'delay': getattr(args, 'delay', None), 'exec_horizon': getattr(args, 'exec_horizon', None), 'mode': getattr(args, 'mode', None), 'model': getattr(args, 'model', None)}
    return config

def _apply_gripper_rules(gripper_value, rules):
    for rule in rules:
        condition = rule.get('when')
        threshold = rule.get('threshold')
        if condition == 'below' and (not gripper_value < threshold):
            continue
        if condition == 'above' and (not gripper_value > threshold):
            continue
        if 'set' in rule:
            gripper_value = rule['set']
        if 'add' in rule:
            gripper_value += rule['add']
    return max(0, gripper_value)

def process_action(task, action):
    action = action.copy()
    left_action = action[:7]
    right_action = action[7:14]
    task_config = TASK_CONFIGS.get(task)
    if task_config is None:
        raise ValueError(f'Invalid task name: {task}')
    action_postprocess = task_config.get('action_postprocess', {})
    left_action[6] = _apply_gripper_rules(left_action[6], action_postprocess.get('left_gripper', []))
    right_action[6] = _apply_gripper_rules(right_action[6], action_postprocess.get('right_gripper', []))
    return (left_action, right_action)

def check_keyboard_input():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None

def read_keyboard_event(esc_timeout: float=0.02):
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch != '\x1b':
        return ch
    if not select.select([sys.stdin], [], [], esc_timeout)[0]:
        return 'ESC'
    ch2 = sys.stdin.read(1)
    if ch2 not in ('[', 'O'):
        return 'ESC'
    if not select.select([sys.stdin], [], [], esc_timeout)[0]:
        return 'ESC'
    ch3 = sys.stdin.read(1)
    return {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}.get(ch3, 'ESC')

def drain_keyboard_events(max_events: int=32):
    events = []
    for _ in range(max_events):
        ev = read_keyboard_event()
        if ev is None:
            break
        events.append(ev)
    return events

def handle_interactive_mode(task_time):
    print('\n' + '=' * 50)
    print(f'Task time: {time.time() - task_time:.1f} s')
    print('INTERACTIVE MODE')
    print("  'c' - Continue running")
    print("  'r' - Reset to starting point and restart")
    print("  'q' - Quit/Stop")
    print('=' * 50)
    while True:
        key = sys.stdin.read(1).lower()
        if key == 'c':
            print('Continuing...')
            return 'continue'
        elif key == 'r':
            print('Restarting...')
            return 'reset'
        elif key == 'q':
            print('Stopping...')
            return 'quit'

def build_observation(observation, config, ros_operator):
    if observation is None:
        return None
    img_front, img_left, img_right, follower_arm_left, follower_arm_right, follower_arm_left_pose, follower_arm_right_pose = observation
    qpos = np.concatenate((np.array(follower_arm_left.position), np.array(follower_arm_right.position)), axis=0)
    eef_pose = ros_operator.build_follower_arm_pose(follower_arm_left_pose, follower_arm_right_pose, follower_arm_left, follower_arm_right)
    return {'qpos': qpos, 'eef_pose': eef_pose, 'images': {config['camera_names'][0]: img_front, config['camera_names'][1]: img_left, config['camera_names'][2]: img_right}}

def get_inference_observation(args, config, ros_operator):
    from ros_operator import get_ros_observation
    return build_observation(get_ros_observation(args, ros_operator), config, ros_operator)

def get_rollout_observation(args, config, ros_operator):
    from ros_operator import get_latest_ros_observation
    return build_observation(get_latest_ros_observation(args, ros_operator), config, ros_operator)

def save_inference_data(args, timesteps, actions, dataset_path, success: bool=True):
    import h5py
    data_size = len(actions)
    first_observation = timesteps[0].observation if timesteps else None
    if first_observation is None:
        raise ValueError('No timesteps available for saving')
    data_dict = {'/observations/qpos': [], '/observations/eef_pose': [], '/action': []}
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/eef_pose'].append(ts.observation['eef_pose'])
        data_dict['/action'].append(action)
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['rollout'] = True
        root.attrs['success'] = bool(success)
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('eef_pose', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        for name, array in data_dict.items():
            root[name][...] = array
    tag = 'SUCCESS' if success else 'FAIL'
    print(f'\x1b[32m\nSaving[{tag}]: {time.time() - t0:.1f} secs. %s \x1b[0m\n' % dataset_path)

class InferenceDataRecorder:

    def __init__(self, args, config, shutdown_event=None):
        self.enabled = args.save_rollout
        self.shutdown_event = shutdown_event
        self.save_args = argparse.Namespace(camera_names=config['camera_names'])
        self.save_dir = os.path.expanduser(args.save_dir)
        self.episode_idx = 0
        if self.enabled:
            from collect_data.collect_data import get_next_episode_idx
            os.makedirs(self.save_dir, exist_ok=True)
            self.episode_idx = get_next_episode_idx(self.save_dir)
            print(f'Rollout recording enabled: {self.save_dir}, next episode {self.episode_idx}')
        self.reset()

    def reset(self):
        self.timesteps = []
        self.actions = []

    def add_step(self, observation, action):
        if not self.enabled or observation is None:
            return
        step_type = dm_env.StepType.FIRST if len(self.timesteps) == 0 else dm_env.StepType.MID
        self.timesteps.append(dm_env.TimeStep(step_type=step_type, reward=None, discount=None, observation=observation))
        self.actions.append(np.asarray(action).copy())
        if len(self.actions) % 50 == 0:
            print(f'Recorded inference frames: {len(self.actions)}')

    def should_stop_waiting(self):
        return rospy.is_shutdown() or (self.shutdown_event is not None and self.shutdown_event.is_set())

    def wait_save_choice(self):
        print("\n\x1b[33m\nRollout paused. Press 'y' = SUCCESS, 'n' = FAIL, 'q' = DISCARD: \x1b[0m", end='', flush=True)
        while not self.should_stop_waiting():
            key = sys.stdin.read(1).lower()
            if key in {'y', 'n', 'q'}:
                print(key)
                return key
        return 'q'

    def save_episode(self):
        if not self.enabled:
            return
        if len(self.actions) == 0:
            print('\x1b[31m\nNo inference data to save (0 frames recorded).\x1b[0m')
            self.reset()
            return
        print('len(timesteps): ', len(self.timesteps))
        print('len(actions)  : ', len(self.actions))
        choice = self.wait_save_choice()
        if choice == 'q':
            print(f'\x1b[31m\nEpisode discarded. {len(self.actions)} frames thrown away.\x1b[0m')
            self.reset()
            return
        success = choice == 'y'
        dataset_path = os.path.join(self.save_dir, f'episode_{self.episode_idx}')
        save_inference_data(self.save_args, self.timesteps.copy(), self.actions.copy(), dataset_path, success=success)
        tag = 'SUCCESS' if success else 'FAIL'
        print(f'\x1b[32mEpisode {self.episode_idx} saved as {tag}!\x1b[0m')
        self.episode_idx += 1
        self.reset()

def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img

def resize_with_pad(im: np.ndarray, height: int, width: int, interpolation: int=cv2.INTER_LINEAR) -> np.ndarray:
    cur_height, cur_width = (im.shape[0], im.shape[1])
    if cur_width == width and cur_height == height:
        return np.ascontiguousarray(im)
    ratio = max(cur_width / width, cur_height / height)
    resized_width = int(cur_width / ratio)
    resized_height = int(cur_height / ratio)
    resized = cv2.resize(im, (resized_width, resized_height), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    out = np.zeros((height, width, resized.shape[2]), dtype=im.dtype)
    pad_top = (height - resized_height) // 2
    pad_left = (width - resized_width) // 2
    out[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = resized
    return out
