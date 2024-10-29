import argparse
import hashlib
import json
import os


def setup_wandb(config):
    import wandb
    if config.wandb_name is None:
        dhash = hashlib.md5()
        encoded = json.dumps(vars(config), sort_keys=True).encode()
        wandb_name = hashlib.md5(encoded).hexdigest()[:8]
    else:
        wandb_name = config.wandb_name

    # specify wandb dir
    if config.wandb_dir is None:
        if 'WANDB_DIR' in os.environ and os.environ['WANDB_DIR'] is not None:
            wandb_dir = os.environ['WANDB_DIR']
        else:
            wandb_dir = 'wandb_folder'
    else:
        wandb_dir = os.path.join(config.wandb_dir, config.wandb_project, config.wandb_group)
    os.makedirs(wandb_dir, exist_ok=True)

    # specify individual run savedir
    savedir = os.path.join(wandb_dir, wandb_name)
    os.makedirs(savedir, exist_ok=True)

    # need to generate id independently from name as ids are only allowed once
    # per project, so there will be conflicts if you ever need to delete runs
    if os.path.exists(os.path.join(savedir, 'wandb_id.txt')):
        with open(os.path.join(savedir, 'wandb_id.txt'), 'r') as f:
            wandb_id = f.read()
    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(savedir, 'wandb_id.txt'), 'w') as f:
            f.write(wandb_id)

    # exit if run is finished
    if os.path.exists(os.path.join(savedir, 'done.txt')):
        wandb.finish()
        return

    wandb.init(config=config, project=config.wandb_project, group=config.wandb_group, name=wandb_name, id=wandb_id, resume='allow', dir=wandb_dir, settings=wandb.Settings(start_method='thread'))
    config = wandb.config
    return config, wandb_dir, wandb_name, wandb_id, savedir


ENVS = [
    'maze2d-umaze-v1',
    'maze2d-large-v1',
    'kitchen-mixed-v0',
    'antmaze-umaze-v1',
    'antmaze-medium-diverse-v1',
    'halfcheetah-expert-v0'
    'ant-expert-v0',
    'walker2d-medium-v0',
    'pen-human-v0',
]

parser = argparse.ArgumentParser()
# general args
parser.add_argument('--savedir', type=str, default='bpe_rl_run', help='savedir for checkpoints')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--env_id', type=str, default='maze2d-large-v1', help='gym/d4rl environment for training')
parser.add_argument('--num_train_envs', type=int, default=1, help='number of environments to run in parallel')
parser.add_argument('--num_eval_envs', type=int, default=10, help='number of environments to run in parallel')
parser.add_argument('--num_evals_per_env', type=int, default=10, help='number of evnironments to evaluate over')
parser.add_argument('--time_limit', type=int, default=int(3.5 * 60 * 60), help='time limit for slurm jobs')

# logging
parser.add_argument('--progress_bar', action='store_true', default=False, help='show progress bar')
parser.add_argument('--log_demonstrations', action='store_true', default=False, help='render and log 10 demonstration trajectories')
parser.add_argument('--time_rollouts', action='store_true', default=False, help='time rollouts')

# offline data args
parser.add_argument('--trajectory_fraction', type=float, default=None, help='fraction of trajectories to use for offline RL')
parser.add_argument('--subset_seed', type=int, default=0, help='seed for subsampling offline data')
parser.add_argument('--exclude_terminal_states', action='store_true', default=False, help='exclude terminal/timeout states from dataset')
parser.add_argument('--filter_inplace_transitions', action='store_true', default=False, help='remove inplace transitions before running tokenization algorithm (antmaze has a lot of them)')

# observation/action space
parser.add_argument('--use_goals', action='store_true', default=False, help='append goals to observation in BC and RL')
parser.add_argument('--stochastic_goals', action='store_true', default=False, help='use stochastic goals like RvS-G, sampling from any state after current')
parser.add_argument('--normalize_observations', action='store_true', default=False, help='normalize observations for subword merging')
parser.add_argument('--normalize_actions', action='store_true', default=False, help='normalize actions before kmeans')
parser.add_argument('--continuous_actions', action='store_true', default=False, help='use cts actions instead of kmeans discretized')
parser.add_argument('--num_clusters', type=int, default=8, help='number of clusters to discretize actions into')
parser.add_argument('--k_actions', type=str, default=None, help='use k-actions', choices=[None, 'subword', 'random', 'repeated'])
parser.add_argument('--tokenizer', type=str, default='bpe', choices=['bpe', 'wordpiece', 'unigram'], help='subword tokenization to use')
parser.add_argument('--vocab_size', type=int, default=16, help='number of subwords in BPE vocab')
parser.add_argument('--prune_vocab', action='store_true', default=False, help='if true, prune the vocab to primitives and non-overlapping extensions')
parser.add_argument('--max_vocab_size', type=int, default=1000, help='number of subwords in BPE vocab to prune from when using pruning')
parser.add_argument('--max_subword_len', type=int, default=10, help='max length of found subwords')
parser.add_argument('--log_action_videos', action='store_true', default=False, help='log action videos at start of training')
parser.add_argument('--stochastic_action_prob', type=float, default=0.0, help='probability for action noise')
parser.add_argument('--stochastic_action_noise', type=float, default=0.0, help='relative magnitude of stochastic action noise')

# online RL args
parser.add_argument('--online_rl', action='store_true', default=False, help='train a policy online in the environment with PPO')
parser.add_argument('--negative_rewards', action='store_true', default=False, help='use -1/0 rewards instead of 0/1 rewards')
parser.add_argument('--stochastic_evals', action='store_true', default=False, help='use stochastic policy for evaluation instead of deterministic')
parser.add_argument('--online_algorithm', type=str, default='sac', choices=['ppo', 'sac'], help='RL algorithm for online learning')
parser.add_argument('--num_steps', type=int, default=10_000_000, help='number of environment steps to train for')
parser.add_argument('--log_interval', type=int, default=10, help='number of iterations to log')
parser.add_argument('--online_log_training', action='store_true', default=False, help='log every training step')
parser.add_argument('--online_save_interval', type=int, default=1_000_000, help='environment step interval to save')
parser.add_argument('--online_eval_interval', type=int, default=1_000_000, help='environment step interval to evaluate')
parser.add_argument('--online_render_interval', type=int, default=1_000_000, help='environment step interval to render videos')
parser.add_argument('--online_lr', type=float, default=3e-4, help='learning rate for online training')
parser.add_argument('--online_batch_size', type=int, default=64, help='online batch size for RL training')
parser.add_argument('--online_dropout', type=float, default=0.0, help='dropout for online training')
parser.add_argument('--online_weight_decay', type=float, default=0.0, help='weight decay for online training')
parser.add_argument('--multiprocess', action='store_true', default=False, help='vectorize across cores')
parser.add_argument('--separate_networks', action='store_true', default=False, help='use separate networks for value estimation and policy')
# PPO specific
parser.add_argument('--ppo_epochs', type=int, default=30, help='number of passes over rollouts when updating ppo')
parser.add_argument('--ppo_ent_coef', type=float, default=0.001, help='entropy bonus for stable baselines 3 ppo')
parser.add_argument('--ppo_gae_lambda', type=float, default=0.95, help='lambda for GAE, 0 is one-step advantage estimate, 1 is sum of rewards')
parser.add_argument('--ppo_unnormalized_advantage', action='store_true', default=False, help='leave advantage estimate unnormalized')
# SAC specific
parser.add_argument('--no_ent_critic_loss', action='store_true', default=False, help='dont use entropy bonus in critic loss')
parser.add_argument('--sac_ent_coef', type=str, default='auto_1.0', help='entropy coef alpha for sac, use auto for learned')
parser.add_argument('--sac_train_freq', type=int, default=1, help='update model every train_freq steps')
parser.add_argument('--sac_tgt_ent_mult', type=float, default=0.0, help='fraction of max entropy that becomes target entropy')
parser.add_argument('--sac_reward_scale', type=float, default=-1.0, help='reward scale compared to q value for diverging q')
parser.add_argument('--sac_gradient_steps', type=int, default=-1, help='how many gradient steps to take, -1 is same as environment steps')
parser.add_argument('--sac_n_critics', type=int, default=2, help='how many critics to use, 10 for redq, but defaults to 2 which is standard')
parser.add_argument('--sac_warmup', type=int, default=5000, help='warmup steps acting randomly before training')

# wandb
parser.add_argument('--use_wandb', action='store_true', default=False, help='log with wandb')
parser.add_argument('--wandb_project', type=str, help='wandb project to log in')
parser.add_argument('--wandb_group', type=str, help='wandb group for runs')
parser.add_argument('--wandb_dir', type=str, help='base wandb directory')
parser.add_argument('--wandb_name', type=str, help='wandb run id')

config = parser.parse_args()
