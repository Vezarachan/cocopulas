# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: base.py
@Author: Lou Xiayin
@Date: 2020/4/21
@Purpose:
@Description: 
"""
from enum import Enum
from typing import Union, List
import numpy as np
from cocopulas.core.base import BaseCopula
from cocopulas.core.errors import NotFittedError
from cocopulas.core.types import Array


class ArchimedeanTypes(Enum):
    AMH = 0
    CLAYTON = 1
    FRANK = 2
    GUMBEL_HOUGAARD = 3


class ArchimedeanBaseCopula(BaseCopula):
    copula_name: ArchimedeanTypes
    alpha: float
    alpha_intervals: List
    alpha_invalids: List

    def __init__(self, alpha: float = None):
        self.alpha = alpha

    def fit(self, data: Array, x0: np.ndarray = None, method="simplex"):
        pass

    def check_fit(self):
        if not self.alpha:
            raise NotFittedError
        self.check_alpha()

    def check_alpha(self):
        lower, upper = self.alpha_intervals
        if self.alpha < lower or self.alpha > upper or (self.alpha in self.alpha_invalids):
            msg = "The parameter  alpha {} computed for {} copula is invalid"
            raise ValueError(msg.format(self.alpha, self.copula_name))

    def cdf(self, x: Array) -> Union[float, np.ndarray]:
        return self.cumulative_distribution_function(x)

    def cumulative_distribution_function(self, x: Array) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def ppf(self, x: Array) -> Union[float, np.ndarray]:
        return self.percent_point_function(x)

    def percent_point_function(self, x: Array) -> Union[float, np.ndarray]:
        # raise NotImplementedError
        pass

    def pdf(self, x: Array) -> Union[float, np.ndarray]:
        return self.probability_density_function(x)

    def probability_density_function(self, x: Array) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def logpdf(self, x: Array) -> Union[float, np.ndarray]:
        return np.log(self.probability_density_function(x))

    def generator(self, x: Array) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def to_dict(self):
        return {
            "copula family", "Archimedean Family",
            "copula name", self.copula_name.name,
            "param value", self.alpha
        }