from pcsis.model.model import Model


class VanderpolSIS(Model):
    """
    Henrion, Didier, and Milan Korda.
    "Convex computation of the region of attraction of polynomial control systems."
    IEEE Transactions on Automatic Control 59.2 (2013): 297-312.
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
        step = 0.01

        x_next[0] = x0 + step * x1 * (-2)
        x_next[1] = x1 + step * (0.8 * x0 + 10 * (x0 ** 2 - 0.21) * x1)

        return x_next

