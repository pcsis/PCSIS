from pcsis.model.model import Model


class Numerical1SIS(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 2
        self.degree_input = 0

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]

        x_next = [None] * 2
        step = 0.01
        x_next[0] = x0 + step * x1
        x_next[1] = (1-step) * x1 + step * x0 * (x0**2 - 1)

        return x_next

