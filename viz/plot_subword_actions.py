import os
import pickle

import d4rl
import gym
import numpy as np
from PIL import Image


def main():
    # will need to manually edit antmaze environment file to remove walls
    expdir = './'
    env_id = 'antmaze-medium-diverse-v1'
    savedir = os.path.join('figure', env_id)
    os.makedirs(savedir, exist_ok=True)

    action_path = os.path.join(expdir, 'discrete_actions.pkl')
    with open(action_path, 'rb') as f:
        actions, primitives = pickle.load(f)

    vocab_path = os.path.join(expdir, 'vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        if type(vocab) == tuple:
            vocab = vocab[0]

    env = gym.make(env_id)
    env = DiscreteActionWrapper(env, primitives)

    num_steps = 100
    for i in range(len(vocab)):
        screens = render_action_repetition(vocab[i], env, num_steps)
        string = repr(to_str(vocab[i]))

        # subsample screens
        screens = screens[::10]

        # scale so earlier frames are less opaque
        alphas = np.linspace(0.4, 1.0, len(screens))
        screens = (alphas[:, None, None, None] * screens)

        # take elementwise maximum over frames to preserve most opaque
        image = screens.max(axis=0).transpose(1, 2, 0)

        # add alpha channel so background is transparent
        alpha = 255 * np.ones_like(image)[:, :, :1]
        image = np.concatenate([image, alpha], axis=2)
        mask = (image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)
        image[mask, 3] = 0

        # save composite
        im = Image.fromarray(image.astype(np.uint8))
        im.save(f'{savedir}/subword_{i}.png')


def render_action_repetition(action, env, num_steps=200):
    screens = []
    assert isinstance(action, list)
    step_count = 0
    obs = env.reset()
    screen = env.render(mode='rgb_array')
    # viewer doesn't get instantiated til screen is called
    if hasattr(env, 'viewer'):  # specifically for antmaze
        env.viewer.cam.elevation = -80
        env.viewer.cam.distance = 15
        env.viewer.cam.lookat[0] = 0.
        env.viewer.cam.lookat[1] = 0.
    while step_count < num_steps:
        for action_ix in action:
            screen = env.render(mode='rgb_array')
            if 'Kitchen' in env.unwrapped.__class__.__name__:
                mask = get_kitchen_mujoco_mask(env, ['panda', 'gripper', 'end_effector'])
                screen = screen * mask[:, :, None]
            else:
                mask = get_antmaze_mujoco_mask(env)
                screen = screen * mask[:, :, None]
            screens.append(screen.transpose(2, 0, 1))
            obs, reward, done, info = env.step(action_ix)
            step_count += 1
            if step_count >= num_steps:
                break
    screens = np.stack(screens, axis=0)
    return screens


def get_kitchen_mujoco_mask(env, patterns):
    screen = env.render(mode='rgb_array', segmentation=True)
    ids, types = screen[:, :, 0], screen[:, :, 1]
    mask = np.zeros_like(ids, dtype=bool)
    # commented is how it should work according to mujoco, but incorrect
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


def get_antmaze_mujoco_mask(env):
    screen = env.viewer.cam
    env.viewer.render(500, 500, segmentation=True)
    screen = env.viewer.read_pixels(500, 500, depth=False)
    ids = screen[::-1, :, 0]  # mask is upside down
    mask = np.zeros_like(ids, dtype=bool)
    # for i in range(40, 53):  # when walls included
    for i in range(2, 15):  # when walls excluded in d4rl maze_env code
        mask[ids == i] = 1
    return mask


# following https://github.com/openai/gym/tree/master/gym/wrappers
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, discrete_actions):
        super().__init__(env)
        self.discrete_actions = discrete_actions
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

    def action(self, act_ix):
        act = self.discrete_actions[act_ix]
        return act


def to_str(int_list):
    return ''.join([chr(i) for i in int_list])


if __name__ == '__main__':
    main()
