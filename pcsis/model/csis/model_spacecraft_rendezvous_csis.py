from pcsis.model.model import  Model
import numpy as np

class Spacecraft_RendezvousCSIS(Model):
    """
    SEEV: Synthesis with Efficient Exact Verification for ReLU Neural Barrier Functions

    https://arxiv.org/pdf/2410.20326

    Obstacle_Avoidance
    """
    def __init__(self):
        self.degree_state = 6
        self.degree_input = 3

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x3 = x[:, 3]
        x4 = x[:, 4]
        x5 = x[:, 5]

        u0 = u[:, 0]
        u1 = u[:, 1]
        u2 = u[:, 2]

        x0 = np.repeat(x0, len(u0))
        x1 = np.repeat(x1, len(u0))
        x2 = np.repeat(x2, len(u0))
        x3 = np.repeat(x3, len(u0))
        x4 = np.repeat(x4, len(u0))
        x5 = np.repeat(x5, len(u0))
        u0 = np.tile(u0, len(x[:, 0]))
        u1 = np.tile(u1, len(x[:, 0]))
        u2 = np.tile(u2, len(x[:, 0]))

        step = 0.01

        n = 0.015245516260

        x_next = [None] * 6
        x_next[0] = x0 + step * x3
        x_next[1] = x1 + step * x4
        x_next[2] = x2 + step * x5
        x_next[3] = x3 + step * (3 * n ** 2 * x0 + 2 * n * x4 + u0)
        x_next[4] = x4 + step * (-2 * n * x3 + u1)
        x_next[5] = x5 + step * (-1 * n ** 2 * x2 + u2)

        return x_next

