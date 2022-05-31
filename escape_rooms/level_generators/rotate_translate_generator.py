

class RotateTranslateGenerator():

    def _flip_level_string(self, flips):

    def _rotate_level_string(self, rots):

    def _augment_level_string(self, level_string):

        split_level_string = [[level_string.split('\n')] for ]
        for rots in range(4):
            for flips in range(2):




    def __init__(self, gdy):
        self._original_levels = gdy['Environment']['Levels']

        self._all_levels = []

        for level_string in self._original_levels:
            self._all_levels.extend(self._augment_level_string(level_string))


    def generate(self, seed):

        assert seed < len(self._all_levels)
        return self._all_levels[seed];

