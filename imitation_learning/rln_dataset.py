import os
import argparse
import numpy as np
from typing import Tuple, List


def load_log_directory(directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expected log format::
        lidar: (T, N) array of lidar scans
        actions: (T, 2) array of [steering, speed]
    """
    lidar_list: List[np.ndarray] = []
    action_list: List[np.ndarray] = []
    if not directory or not os.path.isdir(directory):
        return np.empty((0, 1080)), np.empty((0, 2))

    for name in sorted(os.listdir(directory)):
        if not name.endswith('.npz'):
            continue
        path = os.path.join(directory, name)
        data = np.load(path)
        if 'lidar' not in data or 'actions' not in data:
            continue
        lidar_list.append(data['lidar'])
        action_list.append(data['actions'])
    if not lidar_list:
        return np.empty((0, 1080)), np.empty((0, 2))
    lidar = np.concatenate(lidar_list)
    actions = np.concatenate(action_list)
    return lidar, actions


def build_sequences(lidar: np.ndarray, actions: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if lidar.size == 0:
        return np.empty((0, seq_len * lidar.shape[1])), np.empty((0, actions.shape[1]))

    seqs = []
    acts = []
    for i in range(len(lidar)):
        start = max(0, i - seq_len + 1)
        seq = lidar[start:i + 1]
        if len(seq) < seq_len:
            pad = np.repeat(seq[:1], seq_len - len(seq), axis=0)
            seq = np.vstack((pad, seq))
        seqs.append(seq.flatten())
        acts.append(actions[i])
    return np.asarray(seqs, dtype=np.float32), np.asarray(acts, dtype=np.float32)


def normalize_actions(actions: np.ndarray, steer_limit: float, speed_limit: float) -> np.ndarray:
    if actions.size == 0:
        return actions
    norm = np.empty_like(actions, dtype=np.float32)
    norm[:, 0] = np.clip(actions[:, 0] / steer_limit, -1.0, 1.0)
    norm[:, 1] = np.clip(actions[:, 1] / speed_limit, -1.0, 1.0)
    return norm


def balance_datasets(h_obs: np.ndarray, h_act: np.ndarray, p_obs: np.ndarray, p_act: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError('ratio must be in [0,1]')

    total = min(len(h_obs), len(p_obs))
    h_num = int(total * ratio)
    p_num = total - h_num

    rng = np.random.default_rng()
    h_idx = rng.permutation(len(h_obs))[:h_num]
    p_idx = rng.permutation(len(p_obs))[:p_num]

    obs = np.concatenate([h_obs[h_idx], p_obs[p_idx]])
    act = np.concatenate([h_act[h_idx], p_act[p_idx]])
    shuffle = rng.permutation(len(obs))
    return obs[shuffle], act[shuffle]


def main():
    parser = argparse.ArgumentParser(description='Prepare RLN dataset from logs.')
    parser.add_argument('--human-dir', type=str, default=None, help='Directory with joystick logs (*.npz)')
    parser.add_argument('--pp-dir', type=str, default=None, help='Directory with pure pursuit logs (*.npz)')
    parser.add_argument('--output', type=str, required=True, help='Output dataset file (.npz)')
    parser.add_argument('--seq-len', type=int, default=3, help='Length of lidar sequence')
    parser.add_argument('--human-ratio', type=float, default=0.5, help='Fraction of human samples in the final dataset')
    parser.add_argument('--steer-limit', type=float, default=0.4, help='Maximum absolute steering angle (rad)')
    parser.add_argument('--speed-limit', type=float, default=7.0, help='Maximum speed (m/s) used for normalization')
    args = parser.parse_args()

    human_lidar, human_actions = load_log_directory(args.human_dir)
    pp_lidar, pp_actions = load_log_directory(args.pp_dir)

    human_obs, human_act = build_sequences(human_lidar, human_actions, args.seq_len)
    pp_obs, pp_act = build_sequences(pp_lidar, pp_actions, args.seq_len)

    human_act = normalize_actions(human_act, args.steer_limit, args.speed_limit)
    pp_act = normalize_actions(pp_act, args.steer_limit, args.speed_limit)

    if len(human_obs) and len(pp_obs):
        observs, actions = balance_datasets(human_obs, human_act, pp_obs, pp_act, args.human_ratio)
    else:
        observs = np.concatenate([human_obs, pp_obs])
        actions = np.concatenate([human_act, pp_act])

    np.savez_compressed(args.output, observs=observs, actions=actions)
    print(f'Saved dataset with {len(observs)} samples to {args.output}')


if __name__ == '__main__':
    main()
