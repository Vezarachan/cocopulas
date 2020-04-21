# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: gumbel_hougaard.py
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


class GumbelHougaard(ArchimedeanBaseCopula):
    """ Gumbel Hougaard Copula

    """
    copula_name = ArchimedeanTypes.GUMBEL_HOUGAARD
    alpha_intervals = [1, float("inf")]
    alpha_invalids = [1]

    def cumulative_distribution_function(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        a = self.alpha
        b = np.power(-np.log(u), a) + np.power(-np.log(v), a)
        c = -np.power(b, np.divide(1, a))
        return np.exp(c)

    def probability_density_function(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        a = self.alpha
        w = np.power(-np.log(u), a) + np.power(-np.log(v), a)
        m = np.power(u * v, -1)
        n = np.power(np.log(u) * np.log(v), a -1)
        l = np.power(w, 2/a - 2) + (a - 1) * np.power(w, 1/a - 2)
        return m * n * l * self.cumulative_distribution_function(x)

    def generator(self, x: Array) -> Union[float, np.ndarray]:
        return np.power(-np.log(x), self.alpha)
