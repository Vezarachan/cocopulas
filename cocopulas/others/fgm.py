# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: fgm.py
@Author: 
@Date: 2020/4/21
@Purpose:
@Description: 
"""
from typing import Union
import numpy as np
from cocopulas.core.base import BaseCopula
from cocopulas.core.errors import NotFittedError
from cocopulas.core.types import Array
from cocopulas.utils import split_matrix


class FGM(BaseCopula):
    copula_name = "FGM"
    alpha: float
    alpha_intervals = [-1, 1]

    def __init__(self, alpha: float = None):
        self.alpha = alpha

    def fit(self, data: Array, x0: np.ndarray = None, method="simplex"):
        pass

    def check_fit(self):
        if not self.alpha:
            raise NotFittedError
        lower, upper = self.alpha_intervals
        if self.alpha < lower or self.alpha > upper:
            msg = "The parameter  alpha {} computed for {} copula is invalid"
            raise ValueError(msg.format(self.alpha, self.copula_name))

    def cdf(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        return u * v * (1 + self.alpha * (1 - u) * (1 - v))

    def ppf(self, x: Array) -> Union[float, np.ndarray]:
        pass

    def pdf(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        return 1 + self.alpha * (2 * u - 1) * (2 * v - 1)

    def logpdf(self, x: Array) -> Union[float, np.ndarray]:
        return np.log(self.pdf(x))

    def to_dict(self):
        return {
            "copula family", "Archimedean Family",
            "copula name", self.copula_name.name,
            "param value", self.alpha
        }