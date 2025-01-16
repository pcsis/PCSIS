from pcsis.model.model import Model
import numpy as np


class Vanderpol3DCSIS(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 3
        self.degree_input = 1

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        u0 = u[:, 0]

        x0 = np.repeat(x0, len(u0))
        x1 = np.repeat(x1, len(u0))
        x2 = np.repeat(x2, len(u0))
        u0 = np.tile(u0, len(x[:, 0]))

        x_next = [None] * 3
        step = 0.01

        x_next[0] = x0 + step * x1 * (-2)
        x_next[1] = x1 + step * (0.8 * x0 - 2.1 * x1 + x2 + 10*x0**2*x1)
        x_next[2] = x2 + step * (-x2 + x2 ** 3 + 0.5 * u0)

        return x_next


