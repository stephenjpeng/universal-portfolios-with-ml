import numpy as np
import pandas as pd

from .. import tools
from ..algo import Algo
from up import UP


class Hinted(Algo):
    """Online Portfolio Selection with ML

    Reference:
        WORKING PAPER #TODO
    """

    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(self, ll=0.5, unhinted_strategy=UP(), hinted_strategy=Greedy(),
            how='add'):
        """
        :param ll: lambda parameter in the paper; amount of trust in the hint
        :param unhinted_strategy: unhinted strategy to use
        :param hinted_strategy: hinted strategy to use
        :param how: 'add' or 'mult': type of average to use to combine strategies
        """

        super().__init__(min_history=1, hinted=True)

        # input check
        if ll < 0 or ll > 1:
            raise ValueError('ll should be in [0, 1]')
        if not isinstance(unhinted_strategy, Algo): 
            raise ValueError('unhinted_strategy should be an Algo')
        if not isinstance(hinted_strategy, Algo): 
            raise ValueError('hinted_strategy should be an Algo')
        if how not in ('add', 'mult'):
            raise ValueError('how should be in ["add", "mult"]')

        self.ll = ll
        self.strategy = strategy
        self.r = 1
        self.how = how

        self.last_b_hat = None

    def combine_strategies(self, b_hat, f):
        """
        Returns a weighted average of unhinted and hinted weights based on self.r
        :param b_hat: weights according to the unhinted strategy
        :param f: weights according to the hinted strategy
        """
        if self.how='add':  # weighted arithmetic mean
            return (1 - self.r) * b_hat + self.r * f
        else:  # weighted geometric mean
            return b_hat**(1 - self.r) * f**self.r

    def init_weights(self, columns):
        return self.combine_strategies(
                self.unhinted_strategy.init_weights(),
                self.hinted_strategy.init_weights()
                )

    def init_step(self, X):
        self.last_b_hat = self.unhinted_strategy.init_step(X)
        self.last_f = self.hinted_strategy.init_step(X)
        return self.combine_strategies(
                self.last_b_hat,
                self.last_f
                )

    def step(self, x, last_b, history, hint):
        # update r using last weights
        return_f = np.dot(self.last_f, x)
        return_b_hat = np.dot(self.last_b_hat, x)
        if return_f < return_b_hat:
            self.r = np.sqrt(
                self.r**2 +
                (return_b_hat - return_f) / return_b_hat
             )

        # unhinted strategy
        ## make sure to use the last_b used by unhinted strategy
        b_hat = self.unhinted_strategy.step(x, self.last_b_hat, history)

        # hinted strategy
        f = self.hinted_strategy.step(x, self.last_f, history, hint)

        # update last b_hat, r_t, etc?
        self.last_b_hat = b_hat
        self.last_f = f

        b = self.combine_strategies(b_hat, f)

        return b


if __name__ == "__main__":
    tools.quickrun(Hinted())
