import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
from collections import deque

from dfp import Transition


class Agent:

    def __init__(self, env, collect_policy, replay_buffer, model, optimizer, scheduler=None):

        self.env = env
        self.collect_policy = collect_policy
        self.replay_buffer = replay_buffer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_metrics = deque(maxlen=10)
        self.collect_metrics = deque(maxlen=10)

        # global vars / counters
        self.env_steps = 0
        self.train_steps = 0
        self.episodes = 0
        self.episode_rewards = 0
        self.episode_steps = 0

        # state
        self.obs = None

    def fill_buffer(self):

        env = self.env
        action_space = env.action_space
        self.obs = env.reset()

        pbar = tqdm.tqdm(total=self.replay_buffer.capacity)
        while not self.replay_buffer.is_full():
            action = action_space.sample()  # random policy
            next_obs, reward, done, info = env.step(action)
            transition = Transition(self.obs, action, reward, next_obs, done)
            self.replay_buffer.push(transition)
            self.obs = next_obs

            if done:
                self.obs = env.reset()

            pbar.update(1)

        pbar.close()

    def env_step(self):
        env = self.env

        with torch.no_grad():
            action = self.collect_policy(self.obs)

        next_obs, reward, done, info = env.step(action)

        transition = Transition(self.obs, action, reward, next_obs, done)
        self.replay_buffer.push(transition)

        self.obs = next_obs

        # update counters
        self.env_steps += 1
        self.episode_steps += 1
        self.episode_rewards += reward

        if done:
            info['ep_return'] = self.episode_rewards
            info['ep_length'] = self.episode_steps
            self.collect_metrics.append(info)

            # update counters
            self.episodes += 1
            self.episode_rewards = 0
            self.episode_steps = 0

            # have to reset
            self.obs = env.reset()

    def train_step(self, batch_size):
        self.model.train()

        train_obs, actions, targets, target_masks = self.replay_buffer.sample(batch_size)
        predictions = self.model(train_obs, actions)

        # Replace invalid (exceeding episode length) time steps to cancel their gradients
        # NOTE: We replace the invalid targets with copies of the predictions.
        pred_clones = predictions.clone().detach()
        mask_invalid = ~target_masks
        targets[mask_invalid] = pred_clones[mask_invalid]

        loss = F.mse_loss(predictions, targets, reduction='sum')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        loss_sum = loss.item()
        full_loss = loss_sum / target_masks.numel()
        valid_loss = loss_sum / target_masks.sum().item()
        self.train_metrics.append(dict(pred_loss=full_loss, valid_pred_loss=valid_loss))

        self.train_steps += 1
        self.model.eval()

    @property
    def counters(self):
        return dict(env_steps=self.env_steps, episodes=self.episodes, train_steps=self.train_steps)

    def gather_metrics(self):
        d1 = pd.DataFrame.from_records(self.collect_metrics).mean().round(4).to_dict()
        d2 = pd.DataFrame.from_records(self.train_metrics).mean().round(4).to_dict()
        metrics = {**d1, **d2}

        return metrics
