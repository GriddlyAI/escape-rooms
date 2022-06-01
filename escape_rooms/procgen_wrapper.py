import gym
import numpy as np


class UniformSeedSettingWrapper(gym.Wrapper):
    def __init__(self, env, max_seed=100000000):
        # intialize state counter
        self.max_seed = max_seed
        super().__init__(env)

    def reset(self):
        # reset state counter when env resets
        rnd_seed = np.random.randint(self.max_seed)
        return self.env.reset(rnd_seed)


class SequentialSeedSettingWrapper(gym.Wrapper):
    def __init__(self, env, max_seed=800):
        # intialize state counter
        self.max_seed = max_seed
        self.current_seed = 0
        super().__init__(env)

    def reset(self):
        # reset state counter when env resets
        rnd_seed = self.current_seed
        self.current_seed = (self.current_seed + 1) % self.max_seed
        return self.env.reset(rnd_seed)
