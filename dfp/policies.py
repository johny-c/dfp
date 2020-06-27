import numpy as np


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def step(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


class DFPPolicy:
    def __init__(self, model, obs_fn=None):
        self.model = model
        self.obs_fn = obs_fn or (lambda x: x)

    def __call__(self, obs):

        obs = self.obs_fn(obs)
        p = self.model(obs)  # [B, D, A]
        goal = obs['goal']   # [B, D]

        goal = goal.unsqueeze(1)   # [B, 1, D]
        action_values = goal @ p   # [B, 1, A]
        action_values.squeeze_(1)  # [B, A]
        # action_values = action_values.squeeze(1)
        actions = action_values.argmax(1)  # [B,]
        actions = actions.cpu().numpy()

        # if single environment, return the integer
        batch_size = p.shape[0]
        if batch_size == 1:
            return actions.item()

        return actions


class EpsilonGreedyPolicy:
    def __init__(self, greedy_policy, random_policy, exploration_schedule):
        self.greedy_policy = greedy_policy
        self.random_policy = random_policy
        self.exploration_schedule = exploration_schedule

    def __call__(self, obs):
        u = np.random.rand()
        if u < self.exploration_schedule.current:
            action = self.random_policy(obs)
        else:
            action = self.greedy_policy(obs)

        return action
