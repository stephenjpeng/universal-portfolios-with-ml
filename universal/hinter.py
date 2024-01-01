from typing import Optional

import numpy as np
import pandas as pd


class Hinter(object):
    """Base class for providing hints to hinted algorithms. Sub-classes with specific
    implementations inherit from this class and implement __init__, train, and get_hint functions.
    """

    def __init__(self, n, **kwargs):
        """Subclass to define specific parameters here.
        :param n: number of stocks / shape of the price vector
        """
        self.n = n

    def train(self, X, y):
        raise NotImplementedError

    def get_hint(self, next_x, history):
        raise NotImplementedError
