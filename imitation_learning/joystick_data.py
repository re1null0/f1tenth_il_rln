import argparse
import os
import time
from datetime import datetime

import gym
import numpy as np
import pyglet


def parse_args():
    parser = argparse.ArgumentParser(description="Record joystick demonstrations in the F1TENTH gym")
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to store log files')
    parser.add_argument('--map', type=str, default='levine_map/levine', help='Map path without extension')
    parser.add_argument('--map-ext', type=str, default='.png', help='Extension of the map image')
    parser.add_argument('--render', type=str, default='human_fast', choices=['human', 'human_fast'], help='Render mode')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum number of environment steps')
    parser.add_argument('--steer-limit', type=float, default=0.4, help='Absolute steering limit in radians')
    parser.add_argument('--speed-limit', type=float, default=7.0, help='Maximum speed in m/s')
    parser.add_argument('--dt', type=float, default=0.01, help='Simulation step duration')
    return parser.parse_args()


def main():
    args = parse_args()

    joysticks = pyglet.input.get_joysticks()
    if not joysticks:
        raise RuntimeError('No joystick device found')
    js = joysticks[0]
    js.open()

    env = gym.make('f110_gym:f110-v0', map=args.map, map_ext=args.map_ext, num_agents=1)
    obs, _, done, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    env.render(mode=args.render)

    lidar = []
    actions = []
    step = 0
    try:
        while not done and step < args.max_steps:
            steer = float(js.x) * args.steer_limit
            speed = max(0.0, -float(js.y)) * args.speed_limit
            action = np.array([[steer, speed]], dtype=np.float32)
            obs, _, done, _ = env.step(action)
            env.render(mode=args.render)

            lidar.append(obs['scans'][0].astype(np.float32))
            actions.append(action[0])

            step += 1
            time.sleep(args.dt)
    finally:
        js.close()

    lidar = np.asarray(lidar, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
    np.savez_compressed(out_file, lidar=lidar, actions=actions)
    print(f'Saved {len(actions)} samples to {out_file}')


if __name__ == '__main__':
    main()
