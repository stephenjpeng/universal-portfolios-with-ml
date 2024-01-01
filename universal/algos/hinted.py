import numpy as np
import pandas as pd

from .. import tools
from ..algo import Algo
from ..hinter import Hinter
from .up import UP
from .greedy import Greedy

class Hinted(Algo):
    """Online Portfolio Selection with ML

    Reference:
        WORKING PAPER #TODO
    """

    PRICE_TYPE = "ratio"
    REPLACE_MISSING = True

    def __init__(self, ll=0.5, unhinted_strategy=UP(), hinted_strategy=Greedy(),
            how='add', hinter=None):
        """
        :param ll: lambda parameter in the paper; amount of trust in the hint
        :param unhinted_strategy: unhinted strategy to use
        :param hinted_strategy: hinted strategy to use
        :param how: 'add' or 'mult': type of average to use to combine strategies
        :param hinter: *trained* Hint object for generating hints at each time step
        """

        super().__init__(min_history=0, hinted=True, hinter=hinter)

        # input check
        if ll < 0 or ll > 1:
            raise ValueError('ll should be in [0, 1]')
        if how not in ('add', 'mult'):
            raise ValueError('how should be in ["add", "mult"]')
        if hinter is not None and not isinstance(hinter, Hinter): 
            raise ValueError('hinter should be a Hinter')

        self.ll = ll
        self.unhinted_strategy = unhinted_strategy
        self.hinted_strategy = hinted_strategy
        self.r = 1
        self.how = how

        self.last_b_hat = None
        self.last_f = None

    def _combine_strategies(self, b_hat, f):
        """
        Returns a weighted average of unhinted and hinted weights based on self.r
        :param b_hat: weights according to the unhinted strategy
        :param f: weights according to the hinted strategy
        """
        f = f.reshape(b_hat.shape)
        if self.how=='add':  # weighted arithmetic mean
            return (1 - self.ll / self.r) * b_hat + (self.ll / self.r) * f
        else:  # weighted geometric mean
            return b_hat**(1 - self.ll / self.r) * f**(self.ll / self.r)

    def init_weights(self, columns, hint):
        self.last_b_hat = self.unhinted_strategy.init_weights(columns)
        self.last_f = self.hinted_strategy.init_weights(columns, hint)
        return self._combine_strategies(
                self.last_b_hat,
                self.last_f
                )

    def init_step(self, X, hint):
        self.unhinted_strategy.init_step(X)
        self.hinted_strategy.init_step(X, hint)

    def step(self, x, last_b, history, hint):
        # update r using last weights
        return_f = np.dot(self.last_f.flatten(), x)
        return_b_hat = np.dot(self.last_b_hat.flatten(), x)
        
        if return_f < return_b_hat:
            self.r = np.sqrt(
                self.r**2 +
                (return_b_hat - return_f) / return_b_hat
            ).item()

        # unhinted strategy
        ## when relevant, make sure to use the last_b used by unhinted strategy
        b_hat = self.unhinted_strategy.step(x, self.last_b_hat, history)

        # hinted strategy
        f = self.hinted_strategy.step(x, self.last_f, history, hint)

        # update last b_hat and f
        self.last_b_hat = b_hat
        self.last_f = f

        return self._combine_strategies(b_hat, f)


if __name__ == "__main__":
    tools.quickrun(Hinted())
