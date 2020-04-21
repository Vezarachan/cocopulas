# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: amh.py
@Author: Lou Xiayin
@Date: 2020/4/21
@Purpose:
@Description: 
"""
from typing import Union
import numpy as np
from cocopulas.utils import split_matrix
from cocopulas.archimedean.base import ArchimedeanBaseCopula, ArchimedeanTypes
from cocopulas.core.types import Array


class AMH(ArchimedeanBaseCopula):
    """  Ali–Mikhail–Haq Copula

    """
    copula_name = ArchimedeanTypes.AMH
    alpha_intervals = [-1, 1]
    alpha_invalids = [1]

    def cumulative_distribution_function(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        a = self.alpha
        m = u * v
        n = 1 - a * (1 - u) * (1 - v)
        return np.divide(m, n)

    def probability_density_function(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        a = self.alpha
        m = 1 - a - a * (1 + a) * (1 - u) * (1 - v) + 2 * a * u * v
        c = 1 - a * (1 - u) * (1 - v)
        n = np.power(c, 3)
        return np.divide(m, n)

    def generator(self, x: Array) -> Union[float, np.ndarray]:
        a = self.alpha
        return np.log(np.divide(1 - a - a * x, x))
