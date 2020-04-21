# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: frank.py
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


class Frank(ArchimedeanBaseCopula):
    copula_name = ArchimedeanTypes.FRANK
    alpha_intervals = [float("-inf"), float("inf")]
    alpha_invalids = [0]

    def cumulative_distribution_function(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        a = self.alpha
        up = np.multiply(np.exp(-a * u) - 1, np.exp(-a * v) - 1)
        down = np.exp(-a) - 1
        return -np.divide(1, a) * np.log(1 + np.divide(up, down))

    def probability_density_function(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        a = self.alpha
        up = a * (1 - np.exp(-a)) * np.exp(-a * (u + v))
        c = np.exp(-a) - 1 + (np.exp(-a * u) - 1) * (np.exp(-a * v) - 1)
        down = np.power(c, 2)
        return np.divide(up, down)

    def generator(self, x: Array) -> Union[float, np.ndarray]:
        a = self.alpha
        up = np.exp(-a * x) - 1
        down = np.exp(-a) - 1
        return -np.log(np.divide(up, down))