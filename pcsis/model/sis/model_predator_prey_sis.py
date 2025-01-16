from pcsis.model.model import Model


class PreySIS(Model):
    """
    Henrion, Didier, and Milan Korda.
    "Convex computation of the region of attraction of polynomial control systems."
    IEEE Transactions on Automatic Control 59.2 (2013): 297-312.
    """
    def __init__(self):
        super().__init__()
        self.degree_state = 2

    @staticmethod
    def fx(x, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]

        x_next = [None] * 2
        step = 0.01

        x_next[0] = 0.5 * x0 - x0 * x1
        x_next[1] = -0.5 * x1 + x0 * x1

        return x_next

