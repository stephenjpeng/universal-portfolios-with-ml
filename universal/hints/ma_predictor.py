import numpy as np
import pandas as pd

from ..hinter import Hinter


class MAPredictor(Hinter):
    """This hint predicts reversion to the moving average of the past w time periods.
    Inspired by Li and Hoi, 2012 (https://arxiv.org/ftp/arxiv/papers/1206/1206.4626.pdf)
    """

    def __init__(self, n, w):
        """
        :param n: number of stocks
        :param w: maximum size of the window for moving average
        """
        super().__init__(n)

        self.w = w

    def train(self, X, y):
        pass

    def get_hint(self, next_x, history):
        trunc_history = history[-(self.w-1):]  # shape: min(w-1, history.shape[0]), n
        
        cprod = np.flipud(
            np.cumprod(np.flipud(trunc_history), axis=0))  # shape: min(w-1, history.shape[0]), n

        return (1 +
            np.sum(np.power(cprod, -1), axis=0)
        ) / (cprod.shape[0] + 1)
