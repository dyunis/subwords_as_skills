# Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning

[arXiv](https://arxiv.org/abs/2309.04459)

This repository contains the official code for the paper Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning.

## Installation

```bash
# download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME"/mc3
rm Miniconda3-latest-Linux-x86_64.sh
source "$HOME"/mc3/bin/activate

conda install python=3.9
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# mujoco
# need older Cython to build extensions properly
pip install 'Cython<3.0.0'
# install mujoco separately
# see https://github.com/openai/mujoco-py
mkdir "$HOME"/mujoco_tmp
cd "$HOME"/mujoco_tmp
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
cd "$HOME" 
# If you want to specify a nonstandard location for the package, use the env variable MUJOCO_PY_MUJOCO_PATH.

# gym, need old pip to install
# https://stackoverflow.com/questions/77124879/pip-extras-require-must-be-a-dictionary-whose-values-are-strings-or-lists-of 
pip install setuptools==65.5.0 pip==21
pip install wheel==0.38.0
pip install gym==0.21.0

# d4rl
# specify D4RL_DATASET_DIR for path to download datasets
cd $HOME
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
cd $HOME
mkdir d4rl_tmp  # for d4rl data
# WARNING:
# the kitchen environment in d4rl automatically renders at every step even
# though the observations are joint-based, thus you can speed up the
# environment by removing the rendering step in the d4rl code

pip install -U scikit-learn  # for clustering actions
pip install stable-baselines3[extra]  # for rl
pip install wandb  # for logging
pip install moviepy imageio  # for logging videos
pip install opencv-python
# for mujoco compilation issues
# https://github.com/openai/mujoco-py/issues/627
pip install patchelf
pip install tokenizers  # for action merging

# run these commands to check for the correct python version
python -c "import sys; assert (3,7,0) <= sys.version_info <= (3,10,0), 'python is incorrect version'; print('ok')"
python -c "import platform; assert platform.architecture()[0] == '64bit', 'python is not 64-bit'; print('ok')"
pip install procgen  # for coinrun

# exports to find everything, needed for running code
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/mujoco_tmp/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=$HOME/mujoco_tmp/mujoco210
export MUJOCO_PY_MUJOCO_BUILDPATH="$HOME"/mujoco_tmp
export D4RL_DATASET_DIR="$HOME"/d4rl_tmp
export MUJOCO_GL="osmesa"

# extract coinrun data, assuming subwords_as_skills in $HOME
cd $HOME/subwords_as_skills
tar -czf coinrun_data.tar.gz
```

if you encounter a lock error with `mujoco_py`, delete the lockfile manually:
```
https://github.com/openai/mujoco-py/issues/424
```

## Running the code

```bash
# as an example
python rl.py --env_id antmaze-medium-diverse-v2 --k_actions subword --log_action_videos --max_vocab_size 1_000_000 --num_clusters 16 --num_steps 1_000_000 --prune_vocab --seed 0 --vocab_size 16 --use_wandb --wandb_name test_rl --online_algorithm sac
# see sweeps/ for configurations used in experiments
```

## File list

`subwords_as_skills/`
- `rl.py` - main script for running the code
- `config.py` - arguments for the code
- `data.py` - environment wrappers and utilities for data
- `subwords.py` - code for merging subwords
- `evaluate.py` - code for evaluation and logging
- `callbacks.py` - code for Stable Baselines 3 callbacks needed
- `ppo.py` - temporally extended PPO code for Stable Baselines 3
- `sac_discrete.py` - code for implementing SAC discrete for Stable Baselines 3
- `sac_discrete_policies.py` - code for implementing networks for SAC discrete for Stable Baselines 3
- `viz/` - visualization code for the paper, caution these will need editing for correct paths
  - `csv_to_plot.py` - code for making plots in the paper
  - `plot_subword_actions.py` - code for making plots of subword action rollouts
  - `visitation_heatmap.py` - code for generating visitation heatmaps
- `sweeps/` - hyperparameter configurations for experiments in the paper
