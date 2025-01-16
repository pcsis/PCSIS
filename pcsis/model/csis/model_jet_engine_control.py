from pcsis.model.model import Model
import numpy as np

 
class JetControl(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 2
        self.degree_input = 2

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        u0 = u[:, 0]
        u1 = u[:, 1]

        x0 = np.repeat(x0, len(u0))
        x1 = np.repeat(x1, len(u0))
        u0 = np.tile(u0, len(x[:, 0]))
        u1 = np.tile(u1, len(x[:, 0]))

        x_next = [None] * 2
        step = 0.01

        x_next[0] = x0 + step * (-x1 - 1.5*x0**2 - 0.5*x0**3 - 0.5 + u0)
        x_next[1] = x1 + step * (3 * x0 - x1 + u1)

        return x_next


