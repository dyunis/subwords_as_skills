from collections import defaultdict
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
import torch

import data


def time_evaluation(model, env, deterministic=True):
    for i in range(100):
        import time
        observations = env.reset()
        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        done = False
        leng = 0
        while not done:
            start = time.time()
            actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
            observations, rewards, dones, infos = env.step(actions)
            done = dones[0]
            leng += 1
        print('episode len:', time.time() - start)
        print('len:', leng)


# modified from https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html#evaluate_policy
# need policy entropy, value
def evaluate_policy_entropy_value(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_entropies = []
    episode_values = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_entropies = np.zeros(n_envs)
    current_values = np.zeros(n_envs)
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)

        with torch.no_grad():
            value, log_likelihood, entropy = model.policy.evaluate_actions(torch.from_numpy(observations).to(model.device), torch.from_numpy(actions).to(model.device))
        
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        current_entropies += entropy.detach().cpu().numpy()
        current_values += value.detach().cpu().flatten().numpy()
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            episode_entropies.append(info["episode"]["e"])
                            episode_values.append(info["episode"]["v"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_entropies.append(current_entropies[i])
                        episode_values.append(current_values[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    current_entropies[i] = 0
                    current_values[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_entropies, episode_values
    return mean_reward, std_reward

def evaluate_and_render_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    screens = []
    import cv2

    def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        """
        Renders the environment in its current state, recording the screen in the captured `screens` list

        :param _locals: A dictionary containing all local variables of the callback's scope
        :param _globals: A dictionary containing all global variables of the callback's scope
        """
        screen = env.render(mode="rgb_array")
        screen = cv2.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2))
        screens.append(screen.transpose(2, 0, 1))

    episode_rewards, episode_lengths, episode_entropies, episode_values =  evaluate_policy_entropy_value(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        callback=grab_screens,
        reward_threshold=reward_threshold,
        return_episode_rewards=True,
        warn=warn
    )
    screens = np.stack(screens, axis=0)
    return episode_rewards, episode_lengths, episode_entropies, episode_values, screens


# basically each action assumes a pose, and repeated actions assume a pose more strongly
# do their existence come from the end of training when you need to stop?
def log_action_videos(vocab, env, num_steps=200):
    import wandb
    assert isinstance(env, data.DiscreteActionWrapper)
    for i in range(len(vocab)):
        screens = render_action_repetition(vocab[i], env, num_steps)
        string = repr(to_str(vocab[i]))
        commit = False
        if i == len(vocab) - 1:
            commit = True
        wandb.log({f'actions/action{i}_len{len(vocab[i])}': wandb.Video(screens, caption=f'{num_steps} steps {string}', fps=40, format='mp4')}, commit=commit)


# render individual transitions inside extended actions
def render_extended_action_episode(model, env, vocab, deterministic=False):
    import cv2
    screens = []
    obs = env.reset()
    screen = env.render(mode='rgb_array')
    # viewer doesn't get instantiated til screen is called
    if hasattr(env, 'viewer'):
        env.viewer.cam.elevation = -70
        env.viewer.cam.distance = 40
        env.viewer.cam.lookat[0] = 9.
        env.viewer.cam.lookat[1] = 9.
    states = None
    done = False
    used_actions = []
    while not done:
        action, states = model.predict(obs, state=states, episode_start=1.0, deterministic=deterministic)
        used_actions.append(int(action))
        if vocab is not None:
            action_list = vocab[int(action)]
        else:
            action_list = [int(action)]
        for action_ix in action_list:
            screen = env.render(mode='rgb_array')
            screen = cv2.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2))
            screens.append(screen.transpose(2, 0, 1))
            obs, reward, done, info = env.step(action_ix)
            if done:
                break
    screens = np.stack(screens, axis=0)
    return screens, used_actions


def render_action_repetition(action, env, num_steps=200):
    import cv2
    screens = []
    assert isinstance(action, list)
    step_count = 0
    obs = env.reset()
    screen = env.render(mode='rgb_array')
    # viewer doesn't get instantiated til screen is called
    if hasattr(env, 'viewer'):
        env.viewer.cam.elevation = -70
        env.viewer.cam.distance = 40
        env.viewer.cam.lookat[0] = 9.
        env.viewer.cam.lookat[1] = 9.
    while step_count < num_steps:
        for action_ix in action:
            screen = env.render(mode='rgb_array')
            # if 'Kitchen' in env.unwrapped.__class__.__name__:
                # mask = get_mujoco_mask(env, ['panda', 'gripper', 'end_effector'])
                # screen = screen * mask[:, :, None]
            screen = cv2.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2))
            screens.append(screen.transpose(2, 0, 1))
            obs, reward, done, info = env.step(action_ix)
            step_count += 1
            if step_count >= num_steps:
                break
    screens = np.stack(screens, axis=0)
    return screens


def get_mujoco_mask(env, patterns):
    screen = env.render(mode='rgb_array', segmentation=True)
    ids, types = screen[:, :, 0], screen[:, :, 1]
    mask = np.zeros_like(ids, dtype=np.bool)
    # for i in np.unique(ids):
        # for j in range(23):  # loop over all _mjtObj
            # 1 is _mjtObj Body, more reliable than Geom
            # name = env.sim.model.id2name(i, j)
            # match = False
            # for pattern in patterns:
                # if pattern in name:
                    # match = True
            # if match:
                # mask[ids == i] = 1
                # names.append(name)
    # 0 gives us the floor
    # (1, 35) gives us all parts of robot except left finger
    # 35 is last before 49, 35 gives us left finger
    for i in range(1, 40):
        mask[ids == i] = 1
    names = [env.sim.model.id2name(k, 1) for k in range(100)]
    return mask


def to_str(int_list):
    return ''.join([chr(i) for i in int_list])


def render_trajectory(env, traj_dict):
    import cv2
    screens = []
    env.reset()
    for i in range(len(traj_dict['infos/qpos'])):
        env.unwrapped._wrapped_env.set_state(traj_dict['infos/qpos'][i], traj_dict['infos/qvel'][i])
        xy = traj_dict['observations'][i][:2]
        # goal = traj_dict['infos/goal'][i]
        reward = traj_dict['rewards'][i]
        terminal = traj_dict['terminals'][i]
        timeout = traj_dict['timeouts'][i]
        screen = env.render(mode='rgb_array')
        # render goal
        # env.viewer.cam.distance,elevation,azimuth
        # defaults are elevation=-45, distance=17.0
        if hasattr(env, 'viewer'):
            env.viewer.cam.elevation = -70
            env.viewer.cam.distance = 40
            env.viewer.cam.lookat[0] = 9.
            env.viewer.cam.lookat[1] = 9.
        screen = cv2.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2))
        screens.append(screen.transpose(2, 0, 1))
    screens = np.stack(screens, axis=0)
    return screens


def render_kitchen_trajectory(env, traj_dict):
    import cv2
    screens = []
    env.reset()
    for i in range(len(traj_dict['observations'])):
        env.sim.set_state(traj_dict['observations'][i])
        # advance simulation to render correctly
        # https://stackoverflow.com/questions/75621876/resetting-mujoco-environments-to-a-given-state
        env.sim.forward()
        screen = env.render(mode='rgb_array')
        screen = cv2.resize(screen, (screen.shape[1] // 2, screen.shape[0] // 2))
        screens.append(screen.transpose(2, 0, 1))
    screens = np.stack(screens, axis=0)
    return screens


def log_demonstration_videos(env, traj_dataset, num_demos=10):
    import wandb
    for i in range(num_demos):
        traj_dict = {k: traj_dataset[k][i] for k in traj_dataset}
        if 'infos/qpos' in traj_dict:
            screens = render_trajectory(env, traj_dict)
        elif 'Kitchen' in env.unwrapped.__class__.__name__:
            traj_dict['observations'] = traj_dict['observations'][:, :-1]  # last observation is useless
            screens = render_kitchen_trajectory(env, traj_dict)
        commit = False
        if i == num_demos - 1:
            commit = True
        wandb.log({f'demonstrations/demo{i}': wandb.Video(screens, caption=f'trajectory {i}', fps=40, format='mp4')}, commit=commit)
