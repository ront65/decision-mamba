import csv
import logging
import sys
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig, GPT_EncDec
from mingpt.trainer_atari import Trainer, TrainerConfig, TrainerRec, TrainerRecEnc
import torch
#import blosc
import argparse
from create_dataset import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--test_evals', type=int, default=10)
parser.add_argument('--jumper', type=int, default=15)
parser.add_argument('--encdec_rtgs', type=float, default=0)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--batch_accum', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--train_dropout', type=float, default=0)
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

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, stepwise_returns):
        self.available_idx = np.zeros(done_idxs.shape, dtype=done_idxs.dtype)
        self.available_idx[1:] = np.array(done_idxs[:-1])
        self.available_idx[0] = 0
        self.max_block_size = int(np.max(done_idxs-self.available_idx))
        self.total_trainable_points = (done_idxs-self.available_idx).sum()
        self.block_size = max(timesteps)
        self.vocab_size = max(actions) + 1
        self.idx_by_rewards = sorted(list(range(len(self.available_idx))),
                                     key=lambda z: rtgs[self.available_idx[z]], reverse=True)
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rewards = stepwise_returns
        self.rtgs = rtgs

        print(f"real Dataset size: {len(self.available_idx)}, maxsize: {self.max_block_size}, trainable_pts: {self.total_trainable_points}")
        print(f"Regular Dataset size: {len(self.data) - block_size}")
        #print(f"altered Dataset size: {len(self.available_idx) }, maxsize: {self.max_block_size}")

    def __len__(self):
        return len(self.available_idx)

    def get_best_k_paths(self, k=1):
        k = min(k, len(self.available_idx))
        states, actions, rtgs, masks, rewards = [], [], [], [], []
        for j in range(k):
            items = self.__getitem__(self.idx_by_rewards[j])
            i_states, i_actions, i_rtgs, i_masks, i_rewards = [x.unsqueeze(0) for x in items]
            states.append(i_states)
            actions.append(i_actions)
            rtgs.append(i_rtgs)
            masks.append(i_masks)
            rewards.append(i_rewards)
        output = [states, actions, rtgs, masks, rewards]
        return [torch.cat(y, dim=0) for y in output]


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
        rewards = torch.tensor(self.rewards[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        #timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64).unsqueeze(1)
        out = [states, actions, rtgs, torch.ones(rtgs.shape), rewards]
        out2 = [torch.cat([r, torch.zeros((self.max_block_size-block_size,r.shape[1]), dtype=r.dtype)], dim=0) for r in out]
        return out2

obss, actions, _, done_idxs, rtgs, timesteps, stepwise_returns = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
)

if args.block_type not in ["recc", "recc_enc"]:
    train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)
else:
    train_dataset = StateActionReturnDataset_AllTraj(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps, stepwise_returns)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer = args.num_layers, n_head=8, n_embd=128, 
                  model_type=args.model_type, max_timestep=max(timesteps), block_type=args.block_type,
                  embd_pdrop=args.dropout)
if args.block_type != "recc_enc":
    model = GPT(mconf)
    num_params_require_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'num params: {num_params} ; num params that require grad: {num_params_require_grad}')
else:
    model_enc, model_dec = GPT_EncDec(mconf)
    num_params_require_grad = sum(p.numel() for p in model_enc.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model_enc.parameters())
    print(f'ENC: num params: {num_params} ; num params that require grad: {num_params_require_grad}')
    num_params_require_grad = sum(p.numel() for p in model_dec.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model_dec.parameters())
    print(f'DEC: num params: {num_params} ; num params that require grad: {num_params_require_grad}')

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, batch_accum=args.batch_accum, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps),
                      train_dropout=args.train_dropout, test_evals=args.test_evals, encdec_rtgs=args.encdec_rtgs)
if args.block_type not in ["recc", "recc_enc"]:
    trainer = Trainer(model, train_dataset, None, tconf)
elif args.block_type == "recc":
    tconf.warmup_tokens = train_dataset.total_trainable_points
    tconf.final_tokens = int(train_dataset.total_trainable_points * epochs * 0.8)
    tconf.jumper = args.jumper
    trainer = TrainerRec(model, train_dataset, None, tconf)
else:
    tconf.warmup_tokens = train_dataset.total_trainable_points
    tconf.final_tokens = int(train_dataset.total_trainable_points * epochs * 0.8)
    tconf.jumper = args.jumper
    trainer = TrainerRecEnc(model_enc, model_dec, train_dataset, None, tconf)

trainer.train()
