import csv
import logging
import sys
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig, TrainerRec
import torch
#import blosc
import argparse
from create_dataset import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--block_type', type=str, default='transformer')
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
args = parser.parse_args()

set_seed(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

class StateActionReturnDataset_AllTraj(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.available_idx = np.zeros(done_idxs.shape, dtype=done_idxs.dtype)
        self.available_idx[1:] = np.array(done_idxs[:-1])
        self.available_idx[0] = 0
        self.max_block_size = int(np.max(done_idxs-self.available_idx))
        self.block_size = max(timesteps)
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.available_idx)

    def __getitem__(self, idx):
        idx2 = self.available_idx[idx]
        done_idx = self.done_idxs[idx]
        idx = idx2
        block_size = done_idx - idx
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size,
                                                                                              -1)  # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64).unsqueeze(1)
        out = [states, actions, rtgs, torch.ones(rtgs.shape), timesteps]
        out2 = [torch.cat([r, torch.zeros((self.max_block_size-block_size,r.shape[1]), dtype=r.dtype)], dim=0) for r in out]
        return out2

class StateActionReturnDataset_AllTraj2(Dataset):
    def __init__(self, data, actions, done_idxs, rtgs, timesteps, skipper, max_size=None):
        self.available_idx = np.zeros(done_idxs.shape, dtype=done_idxs.dtype)
        self.available_idx[1:] = np.array(done_idxs[:-1])
        self.available_idx[0] = 0
        self.block_size = int(np.max(done_idxs-self.available_idx))
        self.real_block_size = self.block_size
        if max_size is not None:
            self.block_size = min(max_size, self.block_size)
            print(f"Dataset_TotalTraj max_size of dataset {self.block_size}")
        print(f"Dataset_TotalTraj Min length dataset {int(np.min(done_idxs - self.available_idx))}")
        print(f"Dataset_TotalTraj Max length dataset {int(np.max(done_idxs - self.available_idx))}")
        self.vocab_size = max(actions) + 1
        self.skipper = skipper
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.max_size = max_size

    def set_mode(self, max_size):
        if max_size == 0:
            self.max_size = None
            self.block_size = self.real_block_size
        else:
            self.max_size = max_size
            self.block_size = min(max_size, self.real_block_size)
        return

    def __len__(self):
        return len(self.data) // self.skipper

    def __getitem__(self, idx):
        idx = idx * self.skipper
        #idx = idx + int(np.any(idx == self.done_idxs))
        j = 0
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                #done_idx = int(i)
                if i - idx <= 5*self.skipper:
                    done_idx = i
                elif idx-j<=100:
                    done_idx = min(j + 100, i)
                else:
                    done_idx = idx
                break
            j = i
        idx = j
        block_size = done_idx - idx
        if self.max_size is not None:
            block_size = min(block_size, self.block_size)
            done_idx = idx + block_size

        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size,
                                                                                              -1)  # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64).unsqueeze(1)
        out = [states, actions, rtgs, torch.ones(rtgs.shape), timesteps]
        out2 = [torch.cat([r, torch.zeros((self.block_size-block_size,r.shape[1]), dtype=r.dtype)], dim=0) for r in out]
        #print(f"DEBUGDATASET {out2[0].shape} X {out2[1].shape} X {out2[2].shape} X {out2[3].shape} X {out2[4].shape} X")
        return out2

obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
)

if args.block_type != "recc":
    train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)
else:
    train_dataset = StateActionReturnDataset_AllTraj(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer = args.num_layers, n_head=8, n_embd=128, 
                  model_type=args.model_type, max_timestep=max(timesteps), block_type=args.block_type)
model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
if args.block_type != "recc":
    trainer = Trainer(model, train_dataset, None, tconf)
else:
    trainer = TrainerRec(model, train_dataset, None, tconf)

trainer.train()
