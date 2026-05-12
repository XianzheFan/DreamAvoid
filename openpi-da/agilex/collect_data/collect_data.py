import argparse
import os
import select
import sys
import termios
import time
import tty
import h5py
import rospy
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from ros_operator import RosOperator

class KeyboardHandler:

    def __init__(self):
        self.old_settings = None

    def setup(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def cleanup(self):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def wait_for_key(self, valid_keys):
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key in valid_keys:
                    return key

def save_data(args, timesteps, actions, dataset_path):
    data_size = len(actions)
    data_dict = {'/observations/qpos': [], '/observations/qvel': [], '/observations/effort': [], '/observations/eef_pose': [], '/action': [], '/collect': []}
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/observations/eef_pose'].append(ts.observation['eef_pose'])
        data_dict['/action'].append(action)
        data_dict['/collect'].append('teleop')
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16', chunks=(1, 480, 640))
        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = obs.create_dataset('eef_pose', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('collect', (data_size,), dtype=h5py.string_dtype(encoding='utf-8'))
        for name, array in data_dict.items():
            root[name][...] = array
    print(f'\x1b[32m\nSaving: {time.time() - t0:.1f} secs. %s \x1b[0m\n' % dataset_path)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.', default='', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', default='test', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Starting episode index (auto-increments after each save).', default=0, required=False)
    parser.add_argument('--camera_names', nargs='+', help='camera_names', default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic', default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic', default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic', default='/camera_r/color/image_raw', required=False)
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic', default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic', default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic', default='/camera_r/depth/image_raw', required=False)
    parser.add_argument('--leader_arm_left_topic', action='store', type=str, help='leader_arm_left_topic', default='/leader/joint_left', required=False)
    parser.add_argument('--leader_arm_right_topic', action='store', type=str, help='leader_arm_right_topic', default='/leader/joint_right', required=False)
    parser.add_argument('--follower_arm_left_topic', action='store', type=str, help='follower_arm_left_topic', default='/follower/joint_left', required=False)
    parser.add_argument('--follower_arm_right_topic', action='store', type=str, help='follower_arm_right_topic', default='/follower/joint_right', required=False)
    parser.add_argument('--follower_arm_left_pose_topic', action='store', type=str, help='follower_arm_left_pose_topic', default='/follower/end_pose_euler_left', required=False)
    parser.add_argument('--follower_arm_right_pose_topic', action='store', type=str, help='follower_arm_right_pose_topic', default='/follower/end_pose_euler_right', required=False)
    parser.add_argument('--use_depth_image', action='store_true', help='use_depth_image', required=False)
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate', default=30, required=False)
    args = parser.parse_args()
    return args

def get_next_episode_idx(dataset_dir):
    if not os.path.exists(dataset_dir):
        return 0
    existing_episodes = [f for f in os.listdir(dataset_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
    if not existing_episodes:
        return 0
    indices = []
    for ep in existing_episodes:
        try:
            idx = int(ep.replace('episode_', '').replace('.hdf5', ''))
            indices.append(idx)
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0

def main():
    args = get_arguments()
    if len(args.camera_names) != 3:
        raise ValueError('--camera_names must contain exactly 3 names for front/left/right cameras.')
    ros_operator = RosOperator(args, mode='collection')
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    keyboard_handler = KeyboardHandler()
    keyboard_handler.setup()
    if args.episode_idx == 0:
        current_episode_idx = get_next_episode_idx(dataset_dir)
    else:
        current_episode_idx = args.episode_idx
    try:
        print('\n\x1b[32m========================================\x1b[0m')
        print('\x1b[32m   Continuous Data Collection Started   \x1b[0m')
        print('\x1b[32m========================================\x1b[0m')
        print('\x1b[33mControls:\x1b[0m')
        print('  - Press ENTER to start recording')
        print('  - Press SPACE to stop current recording')
        print("  - Then press 's' to SAVE or 'q' to DISCARD")
        print('  - Press Ctrl+C to exit the program')
        print('\x1b[32m========================================\x1b[0m\n')
        while not rospy.is_shutdown():
            print(f'\n\x1b[36m>>> Episode {current_episode_idx} ready <<<\x1b[0m')
            print('\x1b[33mPress ENTER to start recording...\x1b[0m', end='', flush=True)
            keyboard_handler.wait_for_key(['\n', '\r'])
            print()
            ros_operator.reset()
            timesteps, actions, choice = ros_operator.process(keyboard_handler)
            if choice == 's':
                if len(actions) == 0:
                    print('\x1b[31m\nNo data to save (0 frames recorded).\x1b[0m')
                else:
                    dataset_path = os.path.join(dataset_dir, 'episode_' + str(current_episode_idx))
                    save_data(args, timesteps, actions, dataset_path)
                    print(f'\x1b[32mEpisode {current_episode_idx} saved successfully!\x1b[0m')
                    current_episode_idx += 1
            else:
                print(f'\x1b[31m\nEpisode discarded. {len(actions)} frames thrown away.\x1b[0m')
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('\n\x1b[33m\nExiting data collection...\x1b[0m')
    finally:
        keyboard_handler.cleanup()
        print('\x1b[32mData collection ended.\x1b[0m')
if __name__ == '__main__':
    main()
