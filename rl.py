from datetime import timedelta
from glob import glob
import os
import random
import pickle
import time

import d4rl
import gym
import numpy as np
import pytorch_lightning as pl
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

import data
from evaluate import log_action_videos, log_demonstration_videos, time_evaluation
import ppo
from sac_discrete import SACDiscrete, TemporallyExtendedSACDiscrete
import subwords


def main(config, savedir):
    start_time = time.time()
    if config.continuous_actions and config.k_actions is not None:
        print(f'Continuous actions {config.continuous_actions} is incompatible with k actions {config.k_actions}, returning')
        return 
    # seed for deterministic subwords
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # basically all the seeds affect the starting positions, so more random
    # seeds makes the learning problem easier than a single one
    dataset = get_dataset(config.env_id)
    primitives = None
    if config.normalize_observations:
        dataset['observations'] = data.normalize_observations(dataset['observations'])
    if not config.continuous_actions:
        action_path = os.path.join(savedir, 'discrete_actions.pkl')
        if os.path.exists(action_path):
            with open(action_path, 'rb') as f:
                actions, primitives = pickle.load(f)
        else:
            if 'procgen' in config.env_id:
                actions = dataset['actions']
                primitives = list(range(15))
            elif 'CartPole' in config.env_id:
                actions = dataset['actions']
                primitives = list(range(2))
            else:
                actions, primitives = data.discretize_actions(dataset['actions'], config.num_clusters, normalize=config.normalize_actions)
            with open(action_path, 'wb') as f:
                pickle.dump((actions, primitives), f)
        dataset['actions'] = actions

    if config.filter_inplace_transitions:
        dataset = data.filter_inplace_transitions(dataset)

    traj_dataset = data.split_d4rl_dataset_to_trajectories(dataset, config.exclude_terminal_states)

    if config.trajectory_fraction is not None:
        traj_dataset = data.subsample_d4rl_traj_dataset(traj_dataset, config.trajectory_fraction, config.subset_seed)

    vocab, tokenizer = None, None
    if config.k_actions is not None and not config.continuous_actions:
        vocab_path = os.path.join(savedir, 'vocab.pkl')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
                if type(vocab) == tuple:
                    vocab = vocab[0]
        else:
            vocab, tokenizer = subwords.get_vocab(config, traj_dataset)
            with open(vocab_path, 'wb') as f:
                pickle.dump(vocab, f)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_env = vectorize_env(config.num_train_envs, config, primitives, vocab, seed=config.seed)
    # seed differently than training envs
    eval_env = vectorize_env(config.num_eval_envs, config, primitives, vocab, seed=config.seed + config.num_train_envs + 1, train=False)
    render_env = init_render_env(config, primitives, seed=config.seed + config.num_train_envs + 1)

    if config.use_wandb and config.log_demonstrations:
        print('Logging demonstration videos...')
        demo_env = gym.make(config.env_id)
        demo_dataset = get_dataset(config.env_id)
        # demo_dataset = demo_env.get_dataset()
        demo_traj_dataset = data.split_d4rl_dataset_to_trajectories(demo_dataset, config.exclude_terminal_states)
        log_demonstration_videos(demo_env, demo_traj_dataset)
        print('Logging demonstration videos... DONE')
        return

    if config.use_wandb and not config.continuous_actions and config.log_action_videos:
        pl_checkpoints = list(glob(os.path.join(savedir, '**/*.ckpt'), recursive=True))
        sb3_checkpoints = list(glob(os.path.join(savedir, '*_last.zip')))
        not_logged = len(pl_checkpoints) == 0 and len(sb3_checkpoints) == 0
        if not_logged:
            print('Logging discrete action videos...')
            action_env = init_render_env(config, primitives)
            if config.k_actions is None:
                action_vocab = [[i] for i in range(len(primitives))]
            else:
                action_vocab = vocab
            log_action_videos(action_vocab, action_env)
            del action_vocab, action_env
            print('Logging discrete action videos... DONE')

    if config.online_algorithm == 'sac':
        policy_kwargs = dict(
            activation_fn=torch.nn.LeakyReLU,
            net_arch=[256, 256, 256, 256],
        )
        OnlineAlg = SACDiscrete
        if not config.continuous_actions and config.k_actions is not None:
            OnlineAlg = TemporallyExtendedSACDiscrete
        try:
            # if 'auto' not in config.sac_ent_coef try casting to float
            config.sac_ent_coef = float(config.sac_ent_coef)
        except:
            pass
        model = OnlineAlg(
            'MlpPolicy',
            train_env,
            verbose=1,
            train_freq=config.sac_train_freq,  # collect_rollout only collects single transition
            replay_buffer_class=None,
            # replay_buffer_kwargs=dict(alpha=0.7, beta=0.4) if config.use_per else None,
            buffer_size=1_000_000 if 'procgen' not in config.env_id else 100_000,
            learning_starts=5_000,
            learning_rate=config.online_lr,
            batch_size=config.online_batch_size,
            gradient_steps=config.sac_gradient_steps,  # 1 means 1 gradient step for every collect_rollout, -1 means num_train_env steps per collect_rollout
            ent_coef=config.sac_ent_coef, # auto_1.0 is default
            # ent_coef=f'auto_{config.sac_init_ent_coef}',  # 1 causes critic divergence early on, so does 0.1
            reward_scale=config.sac_reward_scale,
            target_update_interval=config.sac_train_freq,
            target_entropy='auto',
            target_entropy_mult=config.sac_tgt_ent_mult,
            policy_kwargs=policy_kwargs,
            seed=config.seed,
            device=device,
        )
    elif config.online_algorithm == 'ppo':
        # for details https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
        # separate policy + value branches is better
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256]),
        )
        OnlineAlg = PPO
        if not config.continuous_actions and config.k_actions is not None:
            OnlineAlg = ppo.TemporallyExtendedPPO
        model = OnlineAlg(
            'MlpPolicy',
            train_env,
            verbose=1,
            n_steps=1000,
            n_epochs=config.ppo_epochs,
            batch_size=config.online_batch_size,
            # batch_size=config.num_train_envs,
            gae_lambda=config.ppo_gae_lambda,
            ent_coef=config.ppo_ent_coef,
            normalize_advantage=(not config.ppo_unnormalized_advantage),
            policy_kwargs=policy_kwargs,
            seed=config.seed,
            device=device
        )
    else:
        raise NotImplementedError(f'RL algorithm {config.online_algorithm=} is not implemented')

    setup_duration = time.time() - start_time

    # train RL
    from callbacks import Sb3LatestCheckpointCallback, Sb3SlurmTimer
    # save_interval = config.online_save_interval // config.num_train_envs
    save_interval = config.online_save_interval
    saver = Sb3LatestCheckpointCallback(
        save_freq=save_interval,
        save_path=savedir,
        name_prefix='online_rl_sb3_model',
        save_replay_buffer=True,  # for resuming training
    )

    time_left = config.time_limit
    time_left -= setup_duration
    if time_left < 0:
        raise TimeoutError('Time limit {config.time_limit} has elapsed')
    time_str = timedelta_to_str(timedelta(seconds=time_left))
    timer = Sb3SlurmTimer(time_str)

    callback = [saver, timer]

    if config.use_wandb:
        from callbacks import Sb3TrainCallback, Sb3EvalCallback, Sb3VisitationCallback, Sb3VideoRecorderCallback
        # when parallelizing, the evals are delayed
        # eval_interval = config.online_eval_interval // config.num_train_envs
        # render_interval = config.online_render_interval // config.num_train_envs
        eval_interval = config.online_eval_interval
        render_interval = config.online_render_interval
        if config.online_log_training:
            train_evaluator = Sb3TrainCallback()
            callback.append(train_evaluator)
        evaluator = Sb3EvalCallback(
            eval_env,
            eval_interval,
            config.num_eval_envs * config.num_evals_per_env,
            deterministic=True,
        )
        callback.append(evaluator)
        if 'antmaze' in config.env_id:
            heatmap_logger = Sb3VisitationCallback(
                config.env_id,
                eval_interval
            )
            callback.append(heatmap_logger)
        if 'antmaze' in config.env_id:
            renderer = Sb3VideoRecorderCallback(
                render_env,
                render_interval,
                vocab=vocab,
                n_eval_episodes=4,
                deterministic=True,
            )
            callback.append(renderer)

    model.policy.to(device)  # model should be on device, need to redo after pl
    model_path = saver._checkpoint_path('model', extension='zip')
    if os.path.exists(model_path):
        model = OnlineAlg.load(
            model_path,
            env=train_env,
            device='auto',
            reset_num_timesteps=False,
            verbose=1,
        )
        replay_buffer_path = saver._checkpoint_path('replay_buffer_', extension='pkl')
        if hasattr(model, 'replay_buffer') and os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
    timesteps_left = config.num_steps - model.num_timesteps
    # torch.use_deterministic_algorithms(False)  # scatter2d in sb3's train loop, is this important?
    if config.time_rollouts:
        time_evaluation(model, train_env)
    model.learn(
        total_timesteps=timesteps_left,
        log_interval=config.log_interval,
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=config.progress_bar,
    )


# DummyVecEnv calls each environment in sequence within single process
# SubprocVecEnv uses multiprocess, but if env is not IO bound, shouldn't exceed
# number of cores, which will be 2 on slurm
def vectorize_env(num_envs, config, primitives, vocab, seed=0, train=True):
    if config.multiprocess:
        assert num_envs <= 2, f"SubprocVecEnv uses multiprocess, but if env is not IO bound, num_envs ({num_envs}) shouldn't exceed number of cores, which will be 2 on slurm"
        return SubprocVecEnv([make_env(config, primitives, vocab, i, seed, train=train) for i in range(num_envs)])
    else:
        return DummyVecEnv([make_env(config, primitives, vocab, i, seed, train=train) for i in range(num_envs)])


def get_dataset(env_id, discrete_actions=False):
    if 'procgen' in env_id or 'CartPole' in env_id:
        dataset = dict(np.load('demo_easy_merged.npz'))
        # downsample visual observations
        obs = torch.from_numpy(dataset['observations']).permute(0, 3, 1, 2)
        obs = torch.nn.functional.interpolate(obs, scale_factor=(0.5, 0.5))
        dataset['observations'] = obs.permute(0, 2, 3, 1).numpy()
        # reshape to vectors for subwords
        dataset['observations'] = dataset['observations'].reshape(len(dataset['observations']), -1)
    elif 'antmaze-umaze' in env_id:
        # transfer skills from medium to umaze
        # env = gym.make('antmaze-medium-diverse-v1')
        env = gym.make(env_id)
        dataset = env.get_dataset()
    else:
        env = gym.make(env_id)
        dataset = env.get_dataset()
    return dataset


def init_env(config, primitives, vocab, train=True):
    if 'procgen' in config.env_id:
        env = gym.make('procgen:procgen-coinrun-v0', start_level=0, num_levels=10, paint_vel_info=True, distribution_mode='hard', use_sequential_levels=True, debug_mode=1)
        env = data.ProcgenWrapper(env)
    else:
        env = gym.make(config.env_id)
    if config.use_goals:
        env = data.GoalWrapper(env)

    if config.stochastic_action_prob > 0:
        env = data.StochasticActionWrapper(env, config.stochastic_action_prob, config.stochastic_action_noise)

    if config.continuous_actions:
        env = env
    else:
        if config.k_actions is not None:
            env = data.TemporallyExtendedDiscreteActionWrapper(env, primitives, vocab)
        else:
            env = data.DiscreteActionWrapper(env, primitives)
    return env


def init_render_env(config, primitives, seed=0):
    if 'procgen' in config.env_id:
        env = gym.make('procgen:procgen-coinrun-v0', start_level=0, num_levels=10, paint_vel_info=True, distribution_mode='hard', use_sequential_levels=True)
        env = data.ProcgenWrapper(env)
    else:
        env = gym.make(config.env_id)
    if config.use_goals:
        env = data.GoalWrapper(env)
    if not config.continuous_actions:
        env = data.DiscreteActionWrapper(env, primitives)
    env.seed(seed)
    return env


# adaped from https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-unleashing-the-power-of-vectorized-environments
def make_env(config, primitives, vocab, rank, seed=0, train=True):
    def _init():
        env = init_env(config, primitives, vocab, train=train)
        env.seed(config.seed + seed + rank)
        return env
    set_random_seed(config.seed, using_cuda=torch.cuda.is_available())
    return _init


def timedelta_to_str(td):
    days, hours, minutes = td.days, td.seconds // 3600, (td.seconds // 60) % 60
    assert days < 100
    seconds = td.seconds - hours * 3600 - minutes * 60
    day_str = f'{days:0>2}'
    hour_str = f'{hours:0>2}'
    min_str = f'{minutes:0>2}'
    sec_str = f'{seconds:0>2}'
    string = ':'.join([day_str, hour_str, min_str, sec_str])
    return string


if __name__=='__main__':
    from config import config, setup_wandb
    if config.use_wandb:
        import wandb
        config, wandb_dir, wandb_name, wandb_id, savedir = setup_wandb(config)
    else:
        savedir = config.savedir
        os.makedirs(savedir, exist_ok=True)
    try:
        main(config, savedir)
    except TimeoutError as e:
        print(str(e))  # otherwise clogs emails with slurm timeouts
    finally:
        if config.use_wandb:
            wandb.finish()
