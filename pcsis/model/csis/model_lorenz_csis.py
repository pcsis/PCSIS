from pcsis.model.model import Model
import numpy as np


class LorenzCSIS(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 12
        self.degree_input = 3

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x3 = x[:, 3]
        x4 = x[:, 4]
        x5 = x[:, 5]
        x6 = x[:, 6]
        x7 = x[:, 7]
        x8 = x[:, 8]
        x9 = x[:, 9]
        x10 = x[:, 10]
        x11 = x[:, 11]
        u0 = u[:, 0]
        u1 = u[:, 1]
        u2 = u[:, 2]


        x0 = np.repeat(x0, len(u0))
        x1 = np.repeat(x1, len(u0))
        x2 = np.repeat(x2, len(u0))
        x3 = np.repeat(x3, len(u0))
        x4 = np.repeat(x4, len(u0))
        x5 = np.repeat(x5, len(u0))
        x6 = np.repeat(x6, len(u0))
        x7 = np.repeat(x7, len(u0))
        x8 = np.repeat(x8, len(u0))
        x9 = np.repeat(x9, len(u0))
        x10 = np.repeat(x10, len(u0))
        x11 = np.repeat(x11, len(u0))

        u0 = np.tile(u0, len(x[:, 0]))
        u1 = np.tile(u1, len(x[:, 0]))
        u2 = np.tile(u2, len(x[:, 0]))

        x_next = [None] * 12
        step = 0.01

        x_next[0] = x0 + step * ((x1 - x10) * x11 - x0 + 2)
        x_next[1] = x0 + step * ((x2 - x11) * x0 - x1 + 2)
        x_next[2] = x0 + step * ((x3 - x0) * x1 - x2 + 2)
        x_next[3] = x0 + step * ((x4 - x1) * x2 - x3 + 2)
        x_next[4] = x0 + step * ((x5 - x2) * x3 - x4 + 2)
        x_next[5] = x0 + step * ((x6 - x3) * x4 - x5 + 2)
        x_next[6] = x0 + step * ((x7 - x4) * x5 - x6 + 2)
        x_next[7] = x0 + step * ((x8 - x5) * x6 - x7 + 2)
        x_next[8] = x0 + step * ((x9 - x6) * x7 - x8 + 2)
        x_next[9] = x0 + step * ((x10 - x7) * x8 - x9 + 2 + u0)
        x_next[10] = x0 + step * ((x11 - x8) * x9 - x10 + 2 + u1)
        x_next[11] = x0 + step * ((x0 - x9) * x10 - x11 + 2 + u2)

        return x_next


