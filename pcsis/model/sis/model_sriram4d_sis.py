from pcsis.model.model import Model
import numpy as np


class Sriram4DSIS(Model):
    '''
    Sankaranarayanan, Sriram, and Xin Chen.
    "Lyapunov Function Synthesis using Handelman Representations."
    IFAC Proceedings Volumes 46.23 (2013): 576-581.
    '''
    def __init__(self):
        super().__init__()
        self.degree_state = 4
        self.degree_input = 0

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x3 = x[:, 3]

        x_next = [None] * 4
        step = 0.01

        x_next[0] = x0 + step * (-x0 + x1 ** 3 - 3 * x2 * x3)
        x_next[1] = x1 + step * (-x0 - x1 ** 3)
        x_next[2] = x2 + step * (x0 * x3 - x2)
        x_next[3] = x3 + step * (x0 * x2 - x3 ** 3)

        return x_next


