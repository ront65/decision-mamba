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

from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample, CustomFrameStack
import random
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

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)

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
