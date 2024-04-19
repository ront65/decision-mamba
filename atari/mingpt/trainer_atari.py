"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample, sample_rec, sample_rec_enc CustomFrameStack
from torch.nn import functional as F
#import atari_py
from collections import deque
import random
import cv2
import torch


import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed) 


        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset

            g = torch.Generator()
            g.manual_seed(0)
            loader = DataLoader(
                data, shuffle=True, pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                worker_init_fn=seed_worker,
                generator=g,
            )

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')
        
        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

            # -- pass in target returns
            if self.config.model_type == 'naive':
                eval_return = self.get_returns(0)
            elif self.config.model_type == 'reward_conditioned':
                if self.config.game == 'Breakout':
                    eval_return = self.get_returns(90)
                elif self.config.game == 'Seaquest':
                    eval_return = self.get_returns(1150)
                elif self.config.game == 'Qbert':
                    eval_return = self.get_returns(14000)
                elif self.config.game == 'Pong':
                    eval_return = self.get_returns(20)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def get_returns(self, ret):
        self.model.train(False)
        env = gym.make(f"ALE/{self.config.game}-v5", repeat_action_probability = 0, frameskip=1, max_episode_steps=108e3)
        env = CustomFrameStack(AtariPreprocessing(env, scale_obs=True), num_stack=4)

        T_rewards, T_Qs = [], []
        done = True
        for i in range(10):
            state = env.reset(seed = self.config.seed + i)[0]
            state = torch.from_numpy(np.array(state))
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(seed = self.config.seed + i)[0], 0, False
                    state = torch.from_numpy(np.array(state))
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, terminated, truncated, info = env.step(action)
                state = torch.from_numpy(np.array(state)) # state is a LazyFrame object, we need to trandform it into numpy onject.
                done = terminated or truncated
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
        env.close()
        eval_return = sum(T_rewards)/10.
        print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return eval_return


class TrainerRec:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, r, m, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                m = m.to(self.device)
                t = t.to(self.device)

                if config.train_dropout > 0:
                    mp = torch.rand_like(m) < config.train_dropout
                    m = m * mp
                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, _ = model(x, y, y, r, t)

                    new_m = m.reshape(-1)
                    new_y = y.reshape(-1)[new_m > 0]
                    new_logits = logits.reshape(-1, logits.size(-1))[new_m > 0]
                    loss = F.cross_entropy(new_logits, new_y)

                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    if it % config.batch_accum == 0:
                        model.zero_grad()
                    loss.backward()
                    if (it+1) % config.batch_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (m > 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')

        best_return = -float('inf')

        self.tokens = 0  # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

            if epoch % self.config.jumper != 0 and epoch+1 != config.max_epochs:
                continue
            # -- pass in target returns
            if self.config.model_type == 'naive':
                eval_return = self.get_returns(0)
            elif self.config.model_type == 'reward_conditioned':
                if self.config.game == 'Breakout':
                    eval_return = self.get_returns(90, real_rewards=True)
                    eval_return = self.get_returns(90)
                elif self.config.game == 'Seaquest':
                    eval_return = self.get_returns(1150, real_rewards=True)
                    eval_return = self.get_returns(300)
                elif self.config.game == 'Qbert':
                    eval_return = self.get_returns(14000, real_rewards=True)
                    eval_return = self.get_returns(500)
                elif self.config.game == 'Pong':
                    eval_return = self.get_returns(20)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def get_returns(self, ret, real_rewards=False):
        self.model.train(False)
        env = gym.make(f"ALE/{self.config.game}-v5", repeat_action_probability = 0, frameskip=1, max_episode_steps=108e3)
        env = CustomFrameStack(AtariPreprocessing(env, scale_obs=True), num_stack=4)

        T_rewards, T_Qs = [], []
        done = True
        for i in range(self.config.test_evals):
            state = env.reset(seed = self.config.seed + i)[0]
            state = torch.from_numpy(np.array(state))
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            mamba_states = self.model.module.get_model_init_state(1)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action, mamba_states = sample_rec(self.model.module, state, 1, temperature=1.0, sample=True, actions=None,
                                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(
                                        -1),
                                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device), mamba_states=mamba_states)

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(seed = self.config.seed + i)[0], 0, False
                    state = torch.from_numpy(np.array(state))
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, terminated, truncated, info = env.step(action)
                state = torch.from_numpy(np.array(state)) # state is a LazyFrame object, we need to trandform it into numpy onject.
                done = terminated or truncated
                reward_sum += reward
                j += 1

                if j > 5000:
                    done = True
                    print("Cutoff for inference run of length 7000")
                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                if reward > 0:
                    sub_reward = 1
                elif reward == 0:
                    sub_reward = 0
                else:
                    sub_reward = -1
                if real_rewards:
                    sub_reward = reward
                rtgs += [rtgs[-1] - sub_reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action, mamba_states = sample_rec(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                        actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(
                                            1).unsqueeze(0),
                                        rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(
                                            0).unsqueeze(-1),
                                        timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1),
                                                                                                 dtype=torch.int64).to(
                                            self.device)), mamba_states=mamba_states)
        env.close()
        eval_return = sum(T_rewards) * 1.0 / self.config.test_evals
        print(f"Rewards given: {T_rewards}")
        print(f"Mean target return (real_reward? {real_rewards}): {ret}, eval return: {eval_return}")
        self.model.train(True)
        return eval_return

class TrainerRecEnc:
    def __init__(self, model_enc, model_dec, train_dataset, test_dataset, config):
        self.model_enc = model_enc
        self.model_dec = model_dec
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model_enc = torch.nn.DataParallel(self.model_enc).to(self.device)
            self.model_dec = torch.nn.DataParallel(self.model_dec).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model_enc, model_dec, config = self.model_enc, self.model_dec, self.config
        raw_model_enc = model_enc.module if hasattr(self.model_enc, "module") else model_enc
        raw_model_dec = model_dec.module if hasattr(self.model_dec, "module") else model_dec
        optim_groups = raw_model_enc.configure_optimizers(config) + raw_model_dec.configure_optimizers(config)
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model_enc.train(is_train)
            model_dec.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, r, m, w) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                m = m.to(self.device)
                w = w.to(self.device)

                if config.train_dropout > 0:
                    mp = torch.rand_like(m) < config.train_dropout
                    m = m * mp
                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    encode = model_enc(x, y, y, w).unsqueeze(1)
                    logits, rtg_exp = model_dec(x, y, y, w, encode)

                    new_m = m.reshape(-1)
                    new_y = y.reshape(-1)[new_m > 0]

                    new_logits = logits.reshape(-1, logits.size(-1))[new_m > 0]
                    loss1 = F.cross_entropy(new_logits, new_y)
                    loss2 = ((rtg_exp - r) ** 2)[new_m > 0].mean()
                    loss = loss1 + loss2

                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    if it % config.batch_accum == 0:
                        optimizer.zero_grad()
                    loss.backward()
                    if (it+1) % config.batch_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model_enc.parameters(), config.grad_norm_clip)
                        torch.nn.utils.clip_grad_norm_(model_dec.parameters(), config.grad_norm_clip)
                        optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (m > 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')

        best_return = -float('inf')

        self.tokens = 0  # counter used for learning rate decay

        for epoch in range(config.max_epochs):
            self.model_enc.train(True)
            self.model_dec.train(True)
            run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

            if epoch % self.config.jumper != 0 and epoch+1 != config.max_epochs:
                continue
            # -- pass in target returns

            # pre eval:
            self.model_enc.train(False)
            self.model_dec.train(False)
            top_5_enc = self.train_dataset.get_best_k_paths(5)
            x, y, _, _, w = [q.to(self.device) for q in top_5_enc]
            enc_dat = self.model_enc(x, y, y, w).unsqueeze(1)


            if self.config.model_type == 'naive':
                eval_return = self.get_returns(0)
            elif self.config.model_type == 'reward_conditioned':
                if self.config.game == 'Breakout':
                    #eval_return = self.get_returns(90, real_rewards=True)
                    eval_return = self.get_returns(90, enc_dat)
                elif self.config.game == 'Seaquest':
                    #eval_return = self.get_returns(1150, real_rewards=True)
                    eval_return = self.get_returns(300, enc_dat)
                elif self.config.game == 'Qbert':
                    #eval_return = self.get_returns(14000, real_rewards=True)
                    eval_return = self.get_returns(500, enc_dat)
                elif self.config.game == 'Pong':
                    eval_return = self.get_returns(20, enc_dat)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def get_returns(self, ret, enc, real_rewards=False, enc_mode=0):
        env = gym.make(f"ALE/{self.config.game}-v5", repeat_action_probability = 0, frameskip=1, max_episode_steps=108e3)
        env = CustomFrameStack(AtariPreprocessing(env, scale_obs=True), num_stack=4)

        T_rewards, T_Qs = [], []
        done = True
        for i in range(self.config.test_evals):
            state = env.reset(seed = self.config.seed + i)[0]
            state = torch.from_numpy(np.array(state))
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            mamba_states = self.model_dec.module.get_model_init_state(1)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action, mamba_states = sample_rec_enc(self.model_dec.module, state, 1, temperature=1.0, sample=True, actions=None,
                                    step_rewards=None,
                                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device), mamba_states=mamba_states,
                                                          encode=enc, encode_type=enc_mode)

            j = 0
            all_states = state
            actions = []
            reward_sum, done = [], False
            while True:
                if done:
                    #state, reward_sum, done = env.reset(seed = self.config.seed + i)[0], 0, False
                    #state = torch.from_numpy(np.array(state))
                    pass
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, terminated, truncated, info = env.step(action)
                state = torch.from_numpy(np.array(state)) # state is a LazyFrame object, we need to trandform it into numpy onject.
                done = terminated or truncated
                reward_sum += reward
                j += 1

                if j > 5000:
                    done = True
                    print("Cutoff for inference run of length 7000")
                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                if reward > 0:
                    sub_reward = 1
                elif reward == 0:
                    sub_reward = 0
                else:
                    sub_reward = -1
                if real_rewards:
                    sub_reward = reward
                rtgs += [rtgs[-1] - sub_reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action, mamba_states = sample_rec_enc(self.model_dec.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                        actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(
                                            1).unsqueeze(0),
                                        step_rewards=torch.tensor([sub_reward], dtype=torch.long).to(self.device).unsqueeze(
                                            0).unsqueeze(-1),
                                        timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1),
                                                                                                 dtype=torch.int64).to(
                                            self.device)), mamba_states=mamba_states,
                                                          encode=enc, encode_type=enc_mode)
        env.close()
        eval_return = sum(T_rewards) * 1.0 / self.config.test_evals
        print(f"Rewards given: {T_rewards}")
        print(f"Mean target return (real_reward? {real_rewards}): {ret}, eval return: {eval_return}")
        return eval_return
