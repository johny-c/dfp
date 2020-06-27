import vizdoom
import re
import cv2
import gym
import numpy as np
import itertools as it


class DoomSimulatorEnv(gym.Env):
    
    def __init__(self, config_path, color_mode='GRAY', frame_skip=4,
                 resolution=(160, 120), switch_maps=False, maps=('MAP01',)):

        self.config_path = config_path
        self.resolution = resolution
        self.frame_skip = frame_skip
        self.color_mode = color_mode
        self.switch_maps = switch_maps
        self.maps = maps

        self._game = vizdoom.DoomGame()
        self._game.load_config(config_path)
        self.curr_map = 0
        self._game.set_doom_map(self.maps[self.curr_map])

        # set color mode
        if self.color_mode == 'RGB':
            self._game.set_screen_format(vizdoom.ScreenFormat.CRCGCB)
            self.num_channels = 3
        elif self.color_mode == 'GRAY':
            self._game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
            self.num_channels = 1
        else:
            raise ValueError("Unknown color mode")
        
        # set resolution
        try:
            w, h = resolution
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, f"RES_{w}X{h}"))
            self.resize = False
            self.image_shape = [h, w, self.num_channels]
        except Exception as e:
            print("Requested resolution not supported:", e, ". Setting to 160x120 and resizing")
            self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_160X120'))
            self.resize = True
            self.image_shape = [120, 160, self.num_channels]


        self.available_controls, self.continuous_controls, self.discrete_controls = self._analyze_controls(config_path)
        self.num_buttons = self._game.get_available_buttons_size()
        assert(self.num_buttons == len(self.discrete_controls) + len(self.continuous_controls))
        assert(len(self.continuous_controls) == 0) # only discrete for now
        self.num_meas = self._game.get_available_game_variables_size()
        self.meas_tags = [f'meas{i}' for i in range(self.num_meas)]
        self._last_meas = None

        self.episode_count = 0
        self.game_initialized = False
        self.rng = None

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=256, shape=self.image_shape, dtype=np.uint8),
            'meas': gym.spaces.Box(low=0, high=2**32, shape=(self.num_meas,), dtype=np.float32)
        })

        # generate all 2**num_buttons permutations as possible actions
        actions = list(it.product(range(2), repeat=self.num_buttons))
        self._actions = [list(a) for a in actions]
        self.action_space = gym.spaces.Discrete(n=len(actions))

    def _analyze_controls(self, config_path):
        with open(config_path, 'r') as f:
            config = f.read()
        m = re.search('available_buttons[\s]*\=[\s]*\{([^\}]*)\}', config)
        avail_controls = m.group(1).split()
        cont_controls = np.array([bool(re.match('.*_DELTA', c)) for c in avail_controls])
        discr_controls = np.invert(cont_controls)
        return avail_controls, np.squeeze(np.nonzero(cont_controls)), np.squeeze(np.nonzero(discr_controls))

    def close(self):
        if self.game_initialized:
            self._game.close()
            self.game_initialized = False

    def reset(self):
        if not self.game_initialized:
            self._game.init()
            self.game_initialized = True
            self._game.set_window_visible(False)

        if self.switch_maps:
            self.curr_map = (self.curr_map + 1) % len(self.maps)
            self._game.set_doom_map(self.maps[self.curr_map])

        self.episode_count += 1
        self._game.new_episode()

        obs = self._get_obs()
        return obs

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        return [self.rng]

    def _get_obs(self):
        state = self._game.get_state()

        if state is None:
            # assert self._game.is_episode_finished()
            image = np.zeros(self.image_shape, dtype=np.uint8)
            meas = np.zeros(self.num_meas, dtype=np.float32)
        else:
            image = state.screen_buffer
            meas = np.asarray(state.game_variables, dtype=np.float32)

            # store measurements to report
            self._last_meas = meas.copy()

        if self.resize:
            image = cv2.resize(image, self.resolution)

        if self.color_mode == 'GRAY' and image.ndim == 2:
            image = np.expand_dims(image, -1)  # [H, W, 1]

        obs = dict(image=image, meas=meas)
        return obs

    def step(self, action: int):
        """

        :param action: int, index of the action

        :return:
            obs: dict(image=..., meas=...)
            reward: float
            done: bool, terminal state
            info: dict, the measurements at the end of an episode
        """

        action_list = self._actions[action]
        reward = self._game.make_action(action_list, self.frame_skip)
        obs = self._get_obs()
        done = self._game.is_episode_finished() or self._game.is_player_dead()
        info = {}

        if done:
            info = dict(zip(self.meas_tags, self._last_meas))
            
        return obs, reward, done, info
