import gym
import numpy as np


def make_goal(meas_coeffs, temporal_coeffs):
    goal = meas_coeffs.reshape(1, -1) * temporal_coeffs.reshape(-1, 1)
    return goal.astype(np.float32).ravel()


class TargetMeasEnvWrapper(gym.Wrapper):

    def __init__(self, env, meas_coeffs, temporal_coeffs, sample_goals, goal_space='pos_neg'):
        assert goal_space in ('pos', 'pos_neg')
        assert isinstance(env.observation_space, gym.spaces.Dict), \
            f"{self.__class__.__name__} expects dictionary observations."

        super().__init__(env)

        self.meas_coeffs = meas_coeffs
        self.temporal_coeffs = temporal_coeffs
        self.sample_goals = sample_goals
        self.goal_space = goal_space

        self._fixed_goal = make_goal(meas_coeffs, temporal_coeffs)
        self._episode_goal = None

        spaces = env.observation_space.spaces
        g = self._fixed_goal
        low = -1 if goal_space == 'pos_neg' else 0
        spaces.update(goal=gym.spaces.Box(low=low, high=1, shape=g.shape, dtype=g.dtype))
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs):

        if self.sample_goals:
            self._episode_goal = self._sample_goal()  # sample goals during training
        else:
            self._episode_goal = self._fixed_goal.copy()  # set true goal during inference

        # obs should be a dict, containing key 'meas'
        obs = self.env.reset(**kwargs)
        obs.update(goal=self._episode_goal)

        return obs

    def _sample_goal(self):
        # sample random measurement from [0, 1]
        dim_meas = len(self.meas_coeffs)
        meas = self.rng.uniform(size=dim_meas).astype(np.float32)

        if self.goal_space == 'pos_neg':  # sample from [-1, 1]
            meas = 2 * meas - 1

        # goal is just a copy of the measurement over the temporal dimension
        goal = make_goal(meas, self.temporal_coeffs)
        return goal

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs.update(goal=self._episode_goal)

        return obs, reward, done, info
