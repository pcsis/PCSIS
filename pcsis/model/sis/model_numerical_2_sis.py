from pcsis.model.model import Model


class Numerical2SIS(Model):
    """
    Wang, Zheming, and Raphaël M. Jungers.
    "Data-driven computation of invariant sets of discrete time-invariant black-box systems."
    arXiv preprint arXiv:1907.12075 (2019).
    """
    def __init__(self):
        super().__init__()
        self.degree_state = 2
        self.degree_input = 0

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]

        x_next = [None] * 2

        x_next[0] = 2 * x0 ** 2 + x1
        x_next[1] = -2 * (2 * x0 ** 2 + x1) ** 2 - 0.8 * x0

        return x_next

