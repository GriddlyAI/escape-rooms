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

class SeedListWrapper(gym.Wrapper):
    def __init__(self, env, seeds):
        # intialize state counter
        self.seeds = seeds.copy()
        self.completed_seeds = False
        super().__init__(env)


    def step(self, action):
        if self.completed_seeds:
            obs, reward, done, info = self.env.step(0)
            info['ignore'] = True
            return obs, reward, done, info
        else:
            return self.env.step(action)


    def reset(self):
        if len(self.seeds) == 0:
            self.completed_seeds = True
            seed = 0
        else:
            seed = self.seeds.pop()
        return self.env.reset(seed=seed)

class SequentialSeedSettingWrapper(gym.Wrapper):
    def __init__(self, env, seed_offset=0, max_seed=800):
        # intialize state counter
        self.max_seed = max_seed
        self.current_seed = 0
        self.seed_offset = seed_offset
        super().__init__(env)

    def reset(self):
        # reset state counter when env resets
        rnd_seed = self.current_seed
        self.current_seed = (self.current_seed + 1) % self.max_seed
        return self.env.reset(seed=self.seed_offset+rnd_seed)
