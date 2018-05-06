import numpy as np


class NumpyDynamic:

    def __init__(self, dtype, array_size=(100,)):
        self.data = np.zeros(array_size, dtype)
        self.array_size = list(array_size)
        self.size = 0

    def add(self, x):
        if self.size == self.array_size[0]:
            self.array_size[0] *= 2
            newdata = np.zeros(self.array_size, self.data.dtype)
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        return self.data[:self.size]