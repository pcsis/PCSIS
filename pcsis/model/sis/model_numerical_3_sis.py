from pcsis.model.model import Model
import numpy as np


class Numerical3SIS(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 6
        self.degree_input = 0

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x3 = x[:, 3]
        x4 = x[:, 4]
        x5 = x[:, 5]

        x_next = [None] * 6
        step = 0.01

        x_next[0] = x0 + step * (x1 * x3 - x0**3)
        x_next[1] = x1 + step * (-3 * x0 * x3 - x1**3)
        x_next[2] = x2 + step * (-x2 - 3 * x0 * x3 ** 3)
        x_next[3] = x3 + step * (-x3 + x0 * x2)
        x_next[4] = x4 + step * (-x4 + x5 ** 3)
        x_next[5] = x5 + step * (-x4 - x5 + x2 ** 4)

        return x_next


