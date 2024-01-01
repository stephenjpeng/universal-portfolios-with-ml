import numpy as np
import pandas as pd

from ..hinter import Hinter


class ShakyOracle(Hinter):
    """This hint simply adds noise to the next price vector
    """

    def __init__(self, n, gen=None):
        """
        :param n: shape of the price vector
        :param gen: generator of noise to add to the next price vector. Should be a
        function which, when called, returns a sample of size n
        """
        super().__init__(n)

        if gen is None:
            self.gen = lambda : np.random.default_rng().normal(size=self.n) / 2
        else:
            self.gen = gen

    def train(self, X, y):
        pass

    def get_hint(self, next_x, history):
        return next_x + self.gen()
