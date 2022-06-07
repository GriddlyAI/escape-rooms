import numpy as np


class LevelGenerator:
    def __init__(self, gdy):
        self._gdy = gdy

    def generate(self, seed):
        raise NotImplementedError
