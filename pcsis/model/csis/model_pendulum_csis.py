from pcsis.model.model import Model
import numpy as np


class PendulumCSIS(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 2
        self.degree_input = 1

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        u0 = u[:, 0]

        x0 = np.repeat(x0, len(u0))
        x1 = np.repeat(x1, len(u0))
        u0 = np.tile(u0, len(x[:, 0]))

        x_next = [None] * 2
        step = 0.01

        # x_next[0] = x0 + step * x1
        # x_next[1] = x1 + step * (-10 * np.sin(x0) - 0.1 * x1 + u0)
        x_next[0] = x0 + step * x1
        x_next[1] = x1 + step * (-14.715 * np.sin(x0) - 0.3 * x1 + u0)

        return x_next


