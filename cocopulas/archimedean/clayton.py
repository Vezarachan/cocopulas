# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: clayton.py
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


class Clayton(ArchimedeanBaseCopula):
    copula_name = ArchimedeanTypes.CLAYTON
    alpha_intervals = [0, float("inf")]
    alpha_invalids = [0]

    def cumulative_distribution_function(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        a = self.alpha
        c = np.power(u, -a) + np.power(v, -a) - 1
        return np.power(c, -np.divide(1, a))

    def probability_density_function(self, x: Array) -> Union[float, np.ndarray]:
        self.check_fit()
        u, v = split_matrix(x)
        a = self.alpha
        up = (a + 1) * np.power(u * v, a)
        c = np.power(u, a) + np.power(v, a) - np.power(u * v, a)
        down = np.power(c, 1 / a + 2)
        return np.divide(up, down)

    def generator(self, x: Array) -> Union[float, np.ndarray]:
        a = self.alpha
        return np.divide(1, a) * (np.power(x, -a) - 1)
