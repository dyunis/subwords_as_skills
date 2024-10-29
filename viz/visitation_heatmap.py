import gzip
import os
import pickle

# d4rl.locomotion.ant AntMazeEnv (gives 4.0 scaling)
from d4rl.locomotion import maze_env
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# - get maze layout and plot as lines
#   lines 177-196 in maze_env
# - read in replay buffer
# - plot walls in matplotlib
# - plot xy coordinates as heatmap
# - save

def main():
    maze_size_scaling = 4.0
    width = maze_size_scaling
    wall_centers = get_maze_wall_centers(
        # maze_map=maze_env.BIG_MAZE_TEST,
        maze_map=maze_env.HARDEST_MAZE_TEST,
        maze_size_scaling=maze_size_scaling
    )
    x, y = list(zip(*wall_centers))
    xlim = (min(x) - width/2, max(x) + width/2)
    ylim = (min(y) - width/2, max(y) + width/2)
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_aspect('equal')
    for (x, y) in wall_centers:
        rect = matplotlib.patches.Rectangle((x - width/2, y - width/2), width, width, color='black')
        ax.add_patch(rect)
    plt.xlim(xlim)
    plt.ylim(ylim)

    # use spirl env for pickle
    # visitation = load_ssp_visitation()
    # use bet env for pickle
    # visitation = load_sfp_visitation()
    visitation = load_sb3_visitation()
    # remove [0, 0] from visitation for unfilled entries
    visitation = visitation[~np.all(visitation == 0, axis=1)]

    # first 2 dims are x, y of body
    # put single visitation at each corner of the maze to make frequency look good?
    expanded_vis = np.concatenate([
        visitation, np.array([[xlim[0], ylim[0]], [xlim[1], ylim[1]]])
    ], axis=0)
    heatmap, xedges, yedges = np.histogram2d(expanded_vis[:, 0], expanded_vis[:, 1], bins=100, density=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    normalize = matplotlib.colors.PowerNorm(gamma=0.2)
    vmax = None
    heatmap = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='Reds', norm=normalize, vmax=vmax)
    # fig.colorbar(heatmap, ax=ax)
    start = matplotlib.patches.Circle((0, 0), 0.5, color='silver')
    ax.add_patch(start)
    # medium
    # goal = matplotlib.patches.Circle((20, 20), 0.5, color='mediumseagreen')
    # large
    goal = matplotlib.patches.Circle((33, 25), 0.5, color='mediumseagreen')
    ax.add_patch(goal)
    plt.axis('off')
    plt.savefig('visit_subwords.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    q_value_visitation()


# q value visualization with each critic and target
def q_value_visitation():
    import gym
    import d4rl
    import sac_discrete
    import torch
    from data import DiscreteActionWrapper, TemporallyExtendedDiscreteActionWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv
    savedir = '.'
    OnlineAlg = sac_discrete.TemporallyExtendedSACDiscrete
    # vocab = list(range(16))
    # primitives = list(range(16))
    action_path = os.path.join(savedir, 'discrete_actions.pkl')
    with open(action_path, 'rb') as f:
        actions, primitives = pickle.load(f)
    vocab_path = os.path.join(savedir, 'vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab, tokenizer = pickle.load(f)
    vlen = [len(v) for v in vocab]
    print('avg len: ', sum(vlen) / len(vlen))

    # env_id = 'antmaze-umaze-diverse-v1'
    env_id = 'antmaze-medium-diverse-v1'
    def _init():
        env = gym.make(env_id)
        # env = DiscreteActionWrapper(env, primitives)
        env = TemporallyExtendedDiscreteActionWrapper(env, primitives, vocab)
        return env

    vecenv = DummyVecEnv([_init])
    model = OnlineAlg.load(
        os.path.join(savedir, 'online_rl_sb3_model_model_last.zip'),
        env=vecenv,
        device='auto',
        reset_num_timesteps=False,
        verbose=1,
    )

    path = os.path.join(savedir, 'online_rl_sb3_model_replay_buffer__last.pkl')
    with open(path, 'rb') as f:
        replay_buffer = pickle.load(f)
    observations = torch.from_numpy(replay_buffer.observations[:replay_buffer.pos]).to(model.device)
    observations = observations.reshape(-1, observations.size(-1))
    actions = torch.from_numpy(replay_buffer.actions[:replay_buffer.pos]).to(model.device)
    actions = actions.reshape(-1, actions.size(-1))
    x = observations[:, 0].cpu().numpy()
    y = observations[:, 1].cpu().numpy()

    # evaluate q on replay buffer slices
    with torch.no_grad():
        qvals = model.policy.q_net(observations)
        # qvals = model.policy.critic(observations)
        # mean qval at each observation
        qvals_ent = tuple(qv.log_softmax(dim=-1) for qv in qvals)
        qvals_ent = tuple(-(qv * qv.exp()).sum(dim=-1) for qv in qvals_ent)
        qvals_mean = tuple(qv.mean(dim=1) for qv in qvals)
        # qval of selected action
        qvals_act = tuple(torch.gather(qv, dim=1, index=actions.long()).squeeze() for qv in qvals)
    
    # critic value heatmap
    qvals_mean = sum(qv.cpu().numpy() for qv in qvals_mean) / len(qvals_mean)
    qvals_ent = sum(qv.cpu().numpy() for qv in qvals_ent) / len(qvals_ent)
    qvals_act = sum(qv.cpu().numpy() for qv in qvals_act) / len(qvals_act)
    plot_xyz_heatmap(x, y, qvals_mean, 'visit_q.png')
    plot_xyz_heatmap(x, y, qvals_ent, 'visit_q_ent.png')

    # policy entropy heatmap
    if hasattr(model, 'actor'):
        with torch.no_grad():
            log_probs = model.actor.get_action_distrib(observations.float())
            entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
            entropies = entropies.cpu().numpy()
        plot_xyz_heatmap(x, y, entropies, 'visit_ents.png')

    # fixed reward model heatmap
    if hasattr(model, 'reward_net'):
        with torch.no_grad():
            rm_rewards = model.reward_net(observations.float()).squeeze()
            rm_rewards = rm_rewards.cpu().numpy()
        plot_xyz_heatmap(x, y, rm_rewards, 'visit_rm.png')

    # RND heatmap
    if hasattr(model, 'rnd_predictor'):
        with torch.no_grad():
            rnd_rewards = model._get_rnd_reward(observations, actions).squeeze()
            rnd_rewards = rnd_rewards.cpu().numpy()
        plot_xyz_heatmap(x, y, rnd_rewards, 'visit_rnd.png')


def plot_xyz_heatmap(x, y, z, path):
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
    # normalize = matplotlib.colors.PowerNorm(gamma=0.2)
    fig, ax = plt.subplots(tight_layout=True)
    extent = [X[0], X[-1], Y[0], Y[-1]]
    vmin = np.unique(z_grid)[1] if np.unique(z_grid)[0] == 0.0 else np.unique(z_grid)[0]
    vmax = np.unique(z_grid)[-2] if np.unique(z_grid)[-1] == 0.0 else np.unique(z_grid)[-1]

    # nans for empty spaces
    z_grid[counts == 0] = np.nan

    ax.imshow(z_grid, extent=extent, origin='lower', cmap='Reds', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def load_sb3_visitation():
    replay_buffer_dir = f'.'
    visitations = []
    # for seed in range(5):
    # sac with very low exploration (deterministic) also explores well
    # for seed in [4]:
    for seed in range(5):
        path = os.path.join(replay_buffer_dir, f'{exp_name}-{seed}', 'online_rl_sb3_model_replay_buffer__last.pkl')
        with open(path, 'rb') as f:
            replay_buffer = pickle.load(f)
        visitations.append(replay_buffer.observations[:, 0, :2])
    visitation = np.concatenate(visitations, axis=0)
    return visitation


def load_ssp_visitation():
    visitations = []
    for seed in range(5):
        # ssp
        path = '.'
        # spirl open-loop
        # path = '.'
        assert os.path.exists(path)
        with gzip.open(path, 'rb') as f:
            replay_buffer = pickle.load(f)
        visitations.append(replay_buffer['observation'][:, :2])
    visitation = np.concatenate(visitations, axis=0)
    return visitation


def load_sfp_visitation():
    visitations = []
    for seed in range(5):
        path = '.'
        assert os.path.exists(path)
        replay_buffer = np.load(path, allow_pickle=True).item()
        visitations.append(replay_buffer['obs_buf'][:, :, :2].reshape(-1, 2))
    visitation = np.concatenate(visitations, axis=0)
    return visitation


def get_maze_wall_centers(maze_map=maze_env.BIG_MAZE_TEST, maze_size_scaling=4.0):
    torso_x, torso_y = _find_robot(maze_map, maze_size_scaling)
    wall_centers = []
    width = 0.5 * maze_size_scaling
    for i in range(len(maze_map)):
        for j in range(len(maze_map[0])):
            struct = maze_map[i][j]
            if struct == 1:
                # 4 borders of wall geom (is square)
                # (x1, y1), (x2, y2)
                x = j * maze_size_scaling - torso_x
                y = i * maze_size_scaling - torso_y
                wall_centers.append((x, y))
    return wall_centers


def _find_robot(maze_map, maze_size_scaling):
    structure = maze_map
    size_scaling = maze_size_scaling
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == maze_env.RESET:
                return j * size_scaling, i * size_scaling
    raise ValueError('No robot in maze specification.')


if __name__ == '__main__':
    main()
