import numpy as np
import pandas as pd

from .. import tools
from ..algo import Algo
from ..hinter import Hinter


class Greedy(Algo):
    """Greedy hinted strategy-- places all weight on stock with maximum predicted return,
    breaking ties arbitrarily (ties never occure a.s.)
    """

    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(self, hinter=None):
        """
        :param hinter: *trained* Hint object for generating hints at each time step
        """
        super().__init__(min_history=1, hinted=True, hinter=hinter)

        # input check
        if hinter is not None and not isinstance(hinter, Hinter): 
            raise ValueError('hinter should be a Hinter')

        self.hinter = hinter

    def greedy_portfolio(self, hint):
        m = hint.shape[0]
        b = np.zeros(m)
        
        b[np.argmax(hint)] = 1
        return b

    def init_weights(self, columns, hint):
        return self.greedy_portfolio(hint)

    def init_step(self, X, hint):
        pass

    def step(self, x, last_b, history, hint):
        return self.greedy_portfolio(hint)# .reshape(-1, 1)


