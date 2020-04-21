# -*- coding: utf-8 -*-
"""
@Project: 
@File: gaussian.py
@Author: 
@Date: 2020/4/21
@Purpose:
@Description: 
"""
from typing import Union
import numpy as np
from scipy.stats import norm
from cocopulas.utils import split_matrix
from cocopulas.core.types import Array
from cocopulas.elliptical.base import EllipticalTypes, EllipticalBaseCopula


class Gaussian(EllipticalBaseCopula):
    copula_name = EllipticalTypes.GAUSSIAN

    def cumulative_distribution_function(self, x: Array) -> Union[float, np.ndarray]:
        u, v = split_matrix(x)
        r = self.rho
        s = norm.ppf(u)
        t = norm.ppf(v)

    def probability_density_function(self, x: Array) -> Union[float, np.ndarray]:
        pass
