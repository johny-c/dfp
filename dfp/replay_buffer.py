import numpy as np
import torch
import gym

from dfp import Transition


class DFPReplay:

    def __init__(self, capacity, ob_space, future_steps, min_horizon=4, obs_fn=None, target_fn=None, device=None):
        assert isinstance(ob_space, gym.spaces.Dict)
        assert 'meas' in ob_space.spaces

        self.capacity = capacity
        self.ob_space = ob_space
        self.future_steps = future_steps
        self.min_horizon = min_horizon
        self.obs_fn = obs_fn or (lambda x: x)
        self.target_fn = target_fn or (lambda x: x)
        self.device = device

        # dictionary observations (images, measurements)
        self._obs = {}
        for name, space in ob_space.spaces.items():
            self._obs[name] = np.empty(shape=(capacity, *space.shape), dtype=space.dtype)

        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminals = np.empty(capacity, dtype=np.float32)

        # book keeping
        self._load = 0
        self._pointer = 0
        self._current_episode = 0
        self._episode_idx = np.empty(capacity, dtype=np.int64)

    def push(self, transition: Transition):

        for k, v in transition.obs.items():
            self._obs[k][self._pointer] = v

        self._actions[self._pointer] = transition.action
        self._rewards[self._pointer] = transition.reward
        self._terminals[self._pointer] = transition.terminal

        self._episode_idx[self._pointer] = self._current_episode

        # NOTE: If episodes are terminated early (before done is True),
        # training data will be incorrect, namely, future time steps
        # from different episode will be seen as from the same episode.
        if transition.terminal:
            self._current_episode += 1

        self._pointer = (self._pointer + 1) % self.capacity
        self._load = min(self._load + 1, self.capacity)

    def sample(self, batch_size):

        valid_idx = self._sample_valid_indices(batch_size)

        observations = {k: self._obs[k][valid_idx] for k in self.ob_space.spaces}
        actions = self._actions[valid_idx]
        targets, target_masks = self._make_targets(valid_idx)

        observations = self.obs_fn(observations)
        targets = self.target_fn(targets)

        actions = torch.from_numpy(actions).to(self.device)
        target_masks = torch.from_numpy(target_masks).to(self.device)

        return observations, actions, targets, target_masks

    def _sample_valid_indices(self, batch_size):
        """ Sample transitions with enough future time steps """

        valid_idx = []
        num_valid_idx = 0
        while num_valid_idx < batch_size:

            # sample random indices and get their episode ids
            idx = np.random.randint(self._load, size=batch_size)
            episode_idx = self._episode_idx[idx]

            # consider the indices at t + min_horizon
            min_horizon_idx = (idx + self.min_horizon) % self.capacity
            min_horizon_episode_idx = self._episode_idx[min_horizon_idx]

            # if episode_id, is the same in the future, the sample is valid
            valid_samples_mask = episode_idx == min_horizon_episode_idx
            valid_samples = idx[valid_samples_mask]

            valid_idx.append(valid_samples)
            num_valid_idx += len(valid_samples)

        valid_idx = np.concatenate(valid_idx)

        return valid_idx[:batch_size]

    def _make_targets(self, items):

        # [B, 1]
        episode_idx = self._episode_idx[items].reshape(-1, 1)

        measurements = self._obs['meas']
        dim_meas = measurements.shape[1]

        # [B, Dm]
        meas_t = measurements[items]

        # [B, T]
        future_items = items.reshape(-1, 1) + self.future_steps
        future_items = future_items % self.capacity
        future_episode_idx = self._episode_idx[future_items]
        valid_times_mask = future_episode_idx == episode_idx

        # [B, T, Dm]
        future_meas = measurements[future_items]
        meas_diffs = future_meas - np.expand_dims(meas_t, 1)

        # [B, T x Dm]
        B = meas_diffs.shape[0]
        targets = meas_diffs.reshape(B, -1)

        # [B, T] --> [B, T x Dm]
        valid_target_mask = np.repeat(valid_times_mask, dim_meas, 1)

        return targets, valid_target_mask

    def _compute_target_scale(self):
        measurements = self._obs['meas']
        meas_std = np.nanstd(measurements, 0)
        meas_std[meas_std == 0] = 1  # avoid divisions by zero
        target_scale = np.tile(meas_std, len(self.future_steps))  # T x Dm
        return target_scale

    def __len__(self):
        return self._load

    def is_full(self):
        return self._load == self.capacity
