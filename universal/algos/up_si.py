import numpy as np
import pandas as pd

from .. import tools
from ..algo import Algo
from ..hinter import Hinter
from .up import UP
from .greedy import Greedy

class UPSI(Algo):
    """Universal Portfolios with Side Information

    Reference:
        Cover #TODO
    """

    PRICE_TYPE = "ratio"
    REPLACE_MISSING = True

    def __init__(self, hinter, eval_points=1e4, leverage=1.0):
        """
        :param hinter: *trained* Hint object for generating hints at each time step
        :param eval_points: Number of evaluated points (approximately). Complexity of the
            algorithm is O(time * eval_points * nr_assets**2) because of matrix multiplication.
        :param leverage: Maximum leverage used. leverage == 1 corresponds to simplex,
            leverage == 1/nr_stocks to uniform CRP. leverage > 1 allows negative weights
            in portfolio.
        """

        super().__init__(min_history=0, hinted=True, hinter=hinter)

        # input check
        if hinter is not None and not isinstance(hinter, Hinter):
            raise ValueError('hinter should be a Hinter')

        self.hinter = hinter
        self.eval_points = int(eval_points)
        self.leverage = leverage

    def init_weights(self, columns, hint):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X, hint):
        """Create a mesh on simplex and keep wealth of all strategies for each stock."""
        m = X.shape[1]

        # create set of CRPs
        self.W = np.matrix(tools.mc_simplex(m - 1, self.eval_points))
        self.S = np.matrix(np.ones(self.W.shape)).T

        # stretch simplex based on leverage (simple calculation yields this)
        leverage = max(self.leverage, 1.0 / m)
        stretch = (leverage - 1.0 / m) / (1.0 - 1.0 / m)
        self.W = (self.W - 1.0 / m) * stretch + 1.0 / m

    def step(self, x, last_b, history, hint):
        # hint gives side information corresponding to stock with highest pred. return
        s = np.argmax(hint)

        # calculate new wealth of all CRPs
        self.S[s, :] = np.multiply(self.S[s].T, self.W * np.matrix(x).T).T
        b = self.W.T * self.S[s].T

        return b / sum(b)

if __name__ == "__main__":
    tools.quickrun(UPSI())
