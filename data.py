import math

import gym
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
from torch.nn.utils.rnn import pad_sequence


def filter_inplace_transitions(dataset):
    # remove transitions that correspond to small (normalized) distance traveled in obs space
    obs = dataset['observations']
    norm_obs = (obs - obs.mean()) / obs.std()
    obs_diffs = np.linalg.norm(np.diff(norm_obs, axis=0), axis=1)
    mask = (obs_diffs / obs_diffs.mean() < 1.2)
    mask = np.concatenate([mask, [False]], axis=0)  # add 1 back
    mask_start_ixs = np.where(mask)[0]
    # split trajectories when transition missing
    dataset['terminals'][mask_start_ixs-1] = True
    dataset = {
        k: v[~mask] for k, v in dataset.items()
    }
    return dataset


def split_d4rl_dataset_to_trajectories(dataset, exclude_terminal_states=False):
    timeouts, terminals = dataset['timeouts'], dataset['terminals']
    # split on both terminals and timeouts
    timeouts = np.maximum(timeouts, terminals).astype(np.int32)
    # chop trajectories where timeouts are non-consecutive and env resets
    # so timeouts switches from 1 to 0
    timeout_ixs = np.where(np.diff(timeouts) == -1)[0] + 1
    timeout_ixs = [0, *timeout_ixs.tolist()]
    boundaries = [(timeout_ixs[i], timeout_ixs[i+1]) for i in range(len(timeout_ixs) - 1)]
    traj_dataset = {
        k: [v[b[0]:b[1]] for b in boundaries] for k, v in dataset.items()
    }

    if exclude_terminal_states:
        new_dataset = {k: [] for k in traj_dataset.keys()}
        for i in range(len(traj_dataset['observations'])):
            timeouts = np.maximum(traj_dataset['timeouts'][i], traj_dataset['terminals'][i])
            end_ix = int(np.where(np.diff(timeouts) == 1)[0] + 1)
            for k in traj_dataset.keys():
                new_dataset[k].append(traj_dataset[k][i][:end_ix])
        traj_dataset = new_dataset
    return traj_dataset


def subsample_d4rl_traj_dataset(traj_dataset, fraction, seed=0):
    assert fraction > 0.0 and fraction <= 1.0
    # random permutation of indices of length
    rng = np.random.default_rng(seed)
    num_trajs = len(traj_dataset['observations'])
    num_transitions = sum(len(l) for l in traj_dataset['observations'])
    perm = rng.permutation(num_trajs)
    traj_ixs = []
    total_transitions = 0
    for ix in perm:
        traj_ixs.append(ix)
        total_transitions += len(traj_dataset['observations'][ix])
        if total_transitions > fraction * num_transitions:
            break
    new_dataset = {
        k: [v[ix] for ix in traj_ixs] for k, v in traj_dataset.items()
    }
    return new_dataset


def discretize_actions(actions, num_clusters, normalize=False):
    # standardize before pushing to kmeans to account for different ranges
    if normalize:
        mean = actions.mean(axis=0, keepdims=True)
        std = actions.std(axis=0, keepdims=True)
        actions = (actions - mean) / (std + 1e-8)
    # note even with n_init=1, with different cpus the results of below are
    # nondeterministic
    # https://stackoverflow.com/questions/54984058/does-scipy-stats-produce-different-random-numbers-for-different-computer-hardwar
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10, max_iter=1000).fit(actions)
    primitives = kmeans.cluster_centers_
    actions = kmeans.predict(actions)
    if normalize:
        primitives = (std + 1e-8) * primitives + mean
    return actions, primitives


def normalize_observations(observations):
    mean = observations.mean(axis=0, keepdims=True)
    std = observations.std(axis=0, keepdims=True)
    return (observations - mean) / (std + 1e-8)


# following https://github.com/openai/gym/tree/master/gym/wrappers
# add random gaussian noise to actions with some chance
class StochasticActionWrapper(gym.ActionWrapper):
    def __init__(self, env, prob, noise_scale):
        super().__init__(env)
        self.prob = prob
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(12345)

    def action(self, act):
        if self.rng.random() < self.prob:
            # add random gaussian perturbation to actions
            pert = self.rng.normal(size=act.shape)
            pert = np.linalg.norm(act) * pert / np.linalg.norm(pert)
            act = act + self.noise_scale * pert
            act = np.clip(act, self.action_space.low, self.action_space.high)
        return act


# following https://github.com/openai/gym/tree/master/gym/wrappers
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, discrete_actions):
        super().__init__(env)
        self.discrete_actions = discrete_actions
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

    def action(self, act_ix):
        act = self.discrete_actions[act_ix]
        return act


# following https://github.com/openai/gym/tree/master/gym/wrappers
class TemporallyExtendedDiscreteActionWrapper(gym.Wrapper):
    def __init__(self, env, primitives, vocab):
        super().__init__(env)
        # np.array of shape [n, d_action] for original env
        self.primitives = primitives
        # list of list of primitives for "subword" actions
        self.vocab = vocab
        self.action_space = gym.spaces.Discrete(len(vocab))
        self._primitive_steps = 0

    def step(self, act_ix):
        sequence = self.vocab[act_ix]
        # open-loop iteration
        total_reward = 0
        primitive_steps = 0
        for primitive_ix in sequence:
            act = self.primitives[primitive_ix]
            obs, reward, done, info = self.env.step(act)
            total_reward += reward
            primitive_steps += 1
            if done:
                break
        self._primitive_steps = primitive_steps
        reward = total_reward
        return obs, reward, done, info


class TemporallyExtendedActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TemporallyExtendedActionWrapper, self).__init__(env)
        # self.action_space = gym.spaces.
        self.action_fn = np.sin

    def step(self, action):
        # turn parameters of action into parametrized action
        # discounted sum of rewards
        obs, reward, done, info = self.env.step(action)
        return obs, total_reward, done, info


class ProcgenWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    # somehow original in procgen causes errors
    def seed(self, i):
        pass
