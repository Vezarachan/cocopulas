# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: base.py
@Author: 
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


class EllipticalTypes(Enum):
    GAUSSIAN = 0
    STUDENT = 1


class EllipticalBaseCopula(BaseCopula):
    copula_name: EllipticalTypes
    rho: float
    rho_intervals: [-1, 1]
    rho_invalids: []

    def __init__(self, rho: float = None):
        self.rho = rho

    def fit(self, data: Array, x0: np.ndarray = None, method="simplex"):
        pass

    def check_fit(self):
        if not self.rho:
            raise NotFittedError
        self.check_rho()

    def check_rho(self):
        lower, upper = self.rho_intervals
        if (self.rho < lower or self.rho > upper) or (self.rho in self.rho_invalids):
            msg = "The parameter  alpha {} computed for {} copula is invalid"
            raise ValueError(msg.format(self.rho, self.copula_name))

    def cdf(self, x: Array) -> Union[float, np.ndarray]:
        return self.cumulative_distribution_function(x)

    def cumulative_distribution_function(self, x: Array) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def ppf(self, x: Array) -> Union[float, np.ndarray]:
        return self.percent_point_function(x)

    def percent_point_function(self, x: Array) -> Union[float, np.ndarray]:
        pass

    def pdf(self, x: Array) -> Union[float, np.ndarray]:
        return self.probability_density_function(x)

    def probability_density_function(self, x: Array) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def logpdf(self, x: Array) -> Union[float, np.ndarray]:
        return np.log(self.probability_density_function(x))

    def to_dict(self):
        return {
            "copula family": "Elliptical Family",
            "copula name": self.copula_name.name,
            "param value": self.alpha
        }