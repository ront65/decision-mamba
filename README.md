This repo presents a research project to test the potential of Mamba model for RL tasks.
The research is summarized in the "research_summary.pdf" file.

All run examples in this page should be run from the atari directory.

## Downloading datasets
In order to run the code, one has to dowload the dqn-replay data set.
To do so, first you need to have gsutil installed. Instructions for installing gsutil can be found [here](https://cloud.google.com/storage/docs/gsutil_install#install).

After installing gsutil, create a directory for the dataset and load the dataset using gsutil.
Recommended location for the data directory is under the atari folder, and recommended name is dqn/replay.


Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)
```
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

Atari games also require ROMS. 
This repo's installation includes installing gymnasium, which should include istallation of ROMS. However,  it is recommended to follow the instructions [here](https://github.com/google-research/batch_rl?tab=readme-ov-file#:~:text=Important%20notes%20on%20Atari%20ROM%20versions) in order to have ROMS which are compatible with those used to create the training dataset (the dqn-replay set).

## Installation

Dependencies can be installed with the following command:

```
source installation.sh
```

## Example usage

```
CUDA_VISIBLE_DEVICES=0 python run_dt_atari.py --seed 123 --block_size 90 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 --block_type 'mamba' --num_layers 6 --sticky_actions_prob 0.25 --data_dir_prefix [DIRECTORY_NAME]
```

To run multiple seeds:
```
for seed in 12345 23451 34512
do
    CUDA_VISIBLE_DEVICES=0 python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128 --block_type 'mamba' --num_layers 6 --sticky_actions_prob 0.25 --data_dir_prefix [DIRECTORY_NAME]
done
```

## Using WanDB
This repo contains the branch "wandb_support" to support usage of WanDB account.
To use it, first install wandb
```
pip install wandb
```
When running the code, pass your project name as follows:
```
CUDA_VISIBLE_DEVICES=0 python run_dt_atari.py --seed 123 --block_size 90 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 --block_type 'mamba' --num_layers 6 --sticky_actions_prob 0.25 --data_dir_prefix [DIRECTORY_NAME] --wandb_project [YOUR_WANDB_PROJECT_NAME]
```

