import numpy as np


class RotateTranslateGenerator():

    def _string_to_array(self, level_string):
        lines = level_string.splitlines()
        array_lines = []
        for line in lines:
            chars = line.split()
            array_lines.append(chars)

        return np.array(array_lines)


    def _array_to_string(self, array):

        string_shape = array.shape

        level_string = []

        for h in range(0, string_shape[0]):
            for w in range(0, string_shape[1]):
                level_string.append(array[h, w].ljust(4))
            level_string.append("\n")

        return "".join(level_string)



    def _augment_level_string(self, level_string):
        chararray = self._string_to_array(level_string)

        augmented_levels = []

        for r in range(4):
            for f in range(2):
                if f == 1:
                    new_level = np.rot90(np.flip(chararray, 1), r)
                else:
                    new_level = np.rot90(chararray, r)

                augmented_levels.append(self._array_to_string(new_level))

        return augmented_levels

    def __init__(self, gdy):
        self._original_levels = gdy['Environment']['Levels']
        self._all_levels = []

        for level_string in self._original_levels:
            self._all_levels.extend(self._augment_level_string(level_string))


    def generate(self, seed):

        assert seed < len(self._all_levels)
        return self._all_levels[seed]



if __name__ == "__main__":

    level_string = """.   p/G .   .   .   G   G   G   .   .   .   .   G   d/G G   .   .   .   .   
.   G   .   .   G   .   .   .   G   .   .   G   .   W   .   G   .   .   .   
.   G   .   .   G   .   .   .   G   W   W   G   L   L   .   G   .   .   .   
W   G   L   L   G   .   .   .   G   .   .   G   .   L   .   G   .   .   .   
.   G   .   .   G   .   .   .   G   .   .   G   .   W   .   G   .   .   .   
.   G   .   .   G   .   .   .   G   .   .   G   .   L   .   G   .   .   .   
d   G   .   .   G   .   W   .   G   .   .   G   .   L   .   G   .   .   .   
.   G   .   W   G   L   L   .   G   .   .   G   .   W   .   G   .   .   .   
.   G   .   .   d/G .   L   .   G   .   .   G   .   W   .   G   .   .   W   
t   G   T   .   G   W   L   L   G   .   .   G   .   G   G   G   W   W   L   
T   G   T   .   G   .   .   .   G   .   .   G   .   .   .   G   .   .   L   
.   s/G .   .   .   G   G   G   .   .   .   .   G   d/G G   .   .   .   +/G 
"""

    gdy = {
        "Environment": {
            "Levels": [level_string]
        }
    }

    generator = RotateTranslateGenerator(gdy)

    generator.generate(0)