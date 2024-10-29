from datetime import timedelta
import os
import time
from typing import Dict, Any

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import wandb

# same as evaluate_policy but computes entropy
from evaluate import evaluate_policy_entropy_value, render_extended_action_episode
from sac_discrete import SACDiscrete


class Sb3TrainCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self):
        rewards = self.locals['rewards']
        rew_mean, rew_std, rew_max = rewards.mean(), rewards.std(), rewards.max()
        results = {
            'train_online/reward_mean': rew_mean,
            'train_online/reward_std': rew_std,
            'train_online/reward_max': rew_max,
        }
        if isinstance(self.model, MultiDQN):
            with torch.no_grad():
                q_vals = self.model.policy.q_net(torch.tensor(self.locals['new_obs'], device=self.model.device))
            q_vals = tuple(qv.mean().detach().cpu().numpy() for qv in q_vals)
            for i in range(len(q_vals)):
                results[f'train_online/q_val_{i}'] = float(q_vals[i])
        if 'values' in self.locals:
            values = self.locals['values'].detach().flatten().cpu().numpy()
            value_target = self.locals['rollout_buffer'].returns
            val_mean, val_std = values.mean(), values.std()
            tgt_mean, tgt_std = value_target.mean(), value_target.std()
            results['train_online/value_mean'] = val_mean
            results['train_online/value_std'] = val_std
            results['train_online/valtgt_mean'] = tgt_mean
            results['train_online/valtgt_std'] = tgt_std
        if isinstance(self.model, SACDiscrete) or isinstance(self.model, REDQDiscrete) or isinstance(self.model, SACDiscreteR):
            with torch.no_grad():
                obs = torch.from_numpy(self.locals['new_obs']).to(self.model.device)
                log_probs = self.model.policy.actor.get_action_distrib(obs)
                log_probs = log_probs.cpu()
                probs = log_probs.exp()
                entropy = (probs * -log_probs).sum(dim=1).mean()
                qvals = self.model.critic(obs)
                results['train_online/pi_ent'] = float(entropy)
                for i in range(len(qvals)):
                    results[f'train_online/q_val_{i}'] = float(qvals[i].mean())
                    results[f'train_online/q_val_{i}_std'] = float(qvals[i].std(dim=1).mean())
                results['train_online/alpha'] = float(self.model.log_ent_coef.exp())
        wandb.log(results, step=self.model.num_timesteps)
        return True


class Sb3EvalCallback(BaseCallback):
    """
    Custom callback for plotting additional values in wandb.
    """

    def __init__(self, eval_env: gym.Env, eval_freq: int, n_eval_episodes: int = 1, deterministic: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self.last_logging_timestep = 0

    def _get_scalars(self):
        if isinstance(self.model, PPO):
            returns, lengths, entropies, values = evaluate_policy_entropy_value(self.model, self._eval_env, n_eval_episodes=self._n_eval_episodes, deterministic=self._deterministic, return_episode_rewards=True)
        else:
            returns, lengths = evaluate_policy(self.model, self._eval_env, n_eval_episodes=self._n_eval_episodes, deterministic=self._deterministic, return_episode_rewards=True)
        ret_mean, ret_std, ret_min, ret_max = np.mean(returns), np.std(returns), np.min(returns), np.max(returns)
        len_mean, len_std = np.mean(lengths), np.std(lengths)

        results = {
            'online/return_mean': ret_mean,
            'online/return_std': ret_std,
            'online/return_min': ret_min,
            'online/return_max': ret_max,
            'online/length_mean': len_mean,
            'online/length_std': len_std,
        }

        if isinstance(self.model, PPO):
            entropies = [e / l for e, l in zip(entropies, lengths)]
            ent_mean, ent_std = np.mean(entropies), np.std(entropies)
            values = [v / l for v, l in zip(values, lengths)]
            val_mean, val_std = np.mean(values), np.std(values)
            additional_results = {
                'online/entropy_mean': ent_mean,
                'online/entropy_std': ent_std,
                'online/value_mean': val_mean,
                'online/value_std': val_std,
            }
            results = {**results, **additional_results}

        return results

    def _on_step(self):
        # log when > crossing multiple of log
        timesteps_since_logging = self.model.num_timesteps - self.last_logging_timestep
        if timesteps_since_logging // self._eval_freq > 0:
        # if self.n_calls % self._eval_freq == 0:
            results = self._get_scalars()
            wandb.log(results, step=self.model.num_timesteps)
            self.last_logging_timestep = self.model.num_timesteps
        return True


class Sb3VisitationCallback(BaseCallback):
    """
    Custom callback for visualizing state visitation in wandb.
    """

    def __init__(self, env_id, eval_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self._eval_freq = eval_freq
        self.last_logging_timestep = 0
        assert 'antmaze' in env_id

    @torch.no_grad()
    def _log_visitation_heatmaps(self):
        results = {}
        replay_buffer = self.model.replay_buffer
        observations = torch.from_numpy(replay_buffer.observations[:replay_buffer.pos]).to(self.model.device)
        observations = observations.reshape(-1, observations.size(-1))
        actions = torch.from_numpy(replay_buffer.actions[:replay_buffer.pos]).to(self.model.device)
        actions = actions.reshape(-1, actions.size(-1))
        x = observations[:, 0].cpu().numpy()
        y = observations[:, 1].cpu().numpy()

        # log visitation
        visitation = observations[:, :2].cpu().numpy()
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=100, density=True)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        normalize = matplotlib.colors.PowerNorm(gamma=0.2)
        density_fig, ax = plt.subplots(tight_layout=True)
        hm = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='Reds', norm=normalize, vmax=None)
        density_fig.colorbar(hm, ax=ax)
        plt.axis('off')
        results['visit/density'] = density_fig

        # log q values
        if hasattr(self.model.policy, 'q_net'):
            qvals = self.model.policy.q_net(observations)
        elif hasattr(self.model.policy, 'critic'):
            qvals = self.model.policy.critic(observations)
        else:
            raise ValueError
        # for cases like DQN with a single q-function
        if not isinstance(qvals, tuple):
            qvals = (qvals,)
        qvals_ent = tuple(qv.log_softmax(dim=-1) for qv in qvals)
        qvals_ent = tuple(-(qv * qv.exp()).sum(dim=-1) for qv in qvals_ent)
        # mean qval at each observation
        qvals_mean = tuple(qv.mean(dim=1) for qv in qvals)
        # qval of selected action
        qvals_act = tuple(torch.gather(qv, dim=1, index=actions.long()).squeeze() for qv in qvals)

        qvals_mean = sum(qv.cpu().numpy() for qv in qvals_mean) / len(qvals_mean)
        qvals_ent = sum(qv.cpu().numpy() for qv in qvals_ent) / len(qvals_ent)
        qvals_act = sum(qv.cpu().numpy() for qv in qvals_act) / len(qvals_act)

        z_grid, extent, vmin, vmax = get_xyz_heatmap(x, y, qvals_mean)
        q_fig, ax = plt.subplots(tight_layout=True)
        hm = ax.imshow(z_grid, extent=extent, origin='lower', cmap='Reds', vmin=vmin, vmax=vmax)
        q_fig.colorbar(hm, ax=ax)
        plt.axis('off')
        results['visit/q'] = q_fig

        z_grid, extent, vmin, vmax = get_xyz_heatmap(x, y, qvals_ent)
        q_ent_fig, ax = plt.subplots(tight_layout=True)
        hm = ax.imshow(z_grid, extent=extent, origin='lower', cmap='Reds', vmin=vmin, vmax=vmax)
        q_ent_fig.colorbar(hm, ax=ax)
        plt.axis('off')
        results['visit/q_ent'] = q_ent_fig

        # policy entropy heatmap
        if hasattr(self.model, 'actor'):
            log_probs = self.model.actor.get_action_distrib(observations.float())
            entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
            entropies = entropies.cpu().numpy()

            z_grid, extent, vmin, vmax = get_xyz_heatmap(x, y, entropies)
            act_ent_fig, ax = plt.subplots(tight_layout=True)
            hm = ax.imshow(z_grid, extent=extent, origin='lower', cmap='Reds', vmin=vmin, vmax=vmax)
            act_ent_fig.colorbar(hm, ax=ax)
            plt.axis('off')
            results['visit/act_ent'] = act_ent_fig

        results = {
            k: wandb.Image(v) for k, v in results.items()
        }
        wandb.log(results, step=self.model.num_timesteps)
        plt.close(density_fig)
        plt.close(q_fig)
        plt.close(q_ent_fig)
        if 'visit/act_ent' in results:
            plt.close(act_ent_fig)
        if 'visit/rm' in results:
            plt.close(rm_fig)
        if 'visit/rnd' in results:
            plt.close(rnd_fig)

    def _on_step(self):
        # log when > crossing multiple of log
        timesteps_since_logging = self.model.num_timesteps - self.last_logging_timestep
        if timesteps_since_logging // self._eval_freq > 0:
        # if self.n_calls % self._eval_freq == 0:
            self._log_visitation_heatmaps()
            self.last_logging_timestep = self.model.num_timesteps
        return True


def get_xyz_heatmap(x, y, z):
    # https://stackoverflow.com/questions/45777934/creating-a-heatmap-by-sampling-and-bucketing-from-a-3d-array
    # bucket and visualize
    X = np.arange(x.min() - 1, x.max() + 1, 0.2)
    Y = np.arange(y.min() - 1, y.max() + 1, 0.2)
    x_mask = ((x >= X[:-1, None]) & (x < X[1:, None]))
    y_mask = ((y >= Y[:-1, None]) & (y < Y[1:, None]))
    z_grid = np.dot(y_mask * z[None].astype(np.float32), x_mask.T)
    counts = y_mask.dot(x_mask.T.astype(np.float32))
    z_grid[counts > 0] /= counts[counts > 0]

    # first 2 dims are x, y of body
    # put single visitation at each corner of the maze to make frequency look good?
    extent = [X[0], X[-1], Y[0], Y[-1]]
    unique_vals = np.unique(z_grid)
    if len(unique_vals) > 1:
        vmin = unique_vals[1] if unique_vals[0] == 0.0 else unique_vals[0]
        vmax = unique_vals[-2] if unique_vals[-1] == 0.0 else unique_vals[-1]
    else:
        vmin = unique_vals[0]
        vmax = unique_vals[0] + 1.0

    # nans for empty spaces
    z_grid[counts == 0] = np.nan
    return z_grid, extent, vmin, vmax


# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#bonus-make-a-gif-of-a-trained-agent
class Sb3VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, vocab=None, n_eval_episodes: int = 1, deterministic: bool = False):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to WandB

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._render_freq = render_freq
        self._eval_env = eval_env
        self._vocab = vocab
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self.last_logging_timestep = 0

    def _on_step(self) -> bool:
        timesteps_since_logging = self.model.num_timesteps - self.last_logging_timestep
        if timesteps_since_logging // self._render_freq > 0:
        # if self.n_calls % self._render_freq == 0:
            total_screens = []
            for episode in range(self._n_eval_episodes):
                screens, _ = render_extended_action_episode(self.model, self._eval_env, self._vocab, deterministic=self._deterministic)
                total_screens.append(screens)
            screens = np.concatenate(total_screens, axis=0)

            wandb.log({"online/video": wandb.Video(screens, fps=40, format="mp4")}, step=self.model.num_timesteps)
            screens = [] # deallocate screens?
            self.last_logging_timestep = self.model.num_timesteps
        return True


# only save latest checkpoint instead of collecting all of them
# see https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py
# modified from https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/callbacks.html#CheckpointCallback
class Sb3LatestCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        self.last_save_timestep = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = '', extension: str = '') -> str:
        # return os.path.join(self.save_path, f'{self.name_prefix}_{self.checkpoint_type}{self.num_timesteps}_steps.{extension}')
        return os.path.join(self.save_path, f'{self.name_prefix}_{checkpoint_type}_last.{extension}')

    def _on_step(self) -> bool:
        timesteps_since_save = self.model.num_timesteps - self.last_save_timestep
        if timesteps_since_save // self.save_freq > 0:
            model_path = self._checkpoint_path('model', extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")
            self.last_save_timestep = self.model.num_timesteps
        return True


# parsing similar to https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/callbacks/timer.html#Timer
class Sb3SlurmTimer(BaseCallback):
    def __init__(self, duration):
        super().__init__()
        try:
            dhms = duration.strip().split(':')
            dhms = [int(i) for i in dhms]
            duration = timedelta(days=dhms[0], hours=dhms[1], minutes=dhms[2], seconds=dhms[3])
        except:
            raise ValueError(f'Duration {duration} must be specified in the format "DD:HH:MM:SS"')
        self.duration = duration.total_seconds()
        self.start_time = time.time()

    def _on_step(self):
        time_elapsed = time.time() - self.start_time
        if time_elapsed > self.duration:
            raise TimeoutError(f'Duration for training {self.duration} elapsed')
        return True


class Sb3Seeder(BaseCallback):
    def __init__(self, init_seed, seed_freq):
        super().__init__()
        self.init_seed = init_seed
        self.seed_freq = seed_freq

    def _on_step(self):
        # set seed based on num model updates (env steps differ with subwords)
        # need to load model and set seed based on loaded model, but how to
        # partition seeding when rollouts are single step?
        self.model.set_random_seed(self.init_seed + self._n_updates)
        if self.model._n_updates % self.seed_freq == 0:
            self.model.set_random_seed(self.init_seed + self.model._n_updates // self.seed_freq) 
        return True
