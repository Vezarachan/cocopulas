# -*- coding: utf-8 -*-
"""
@Project: 
@File: estimator.py
@Author: 
@Date: 2020/4/25
@Purpose:
@Description: 
"""
from typing import Dict, Optional
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from cocopulas.core.base import BaseCopula
from cocopulas.core.types import Array


class Estimator:

    def __init__(self, copula: BaseCopula, data: np.ndarray, x0: np.ndarray,
                 method="ml", optimset: Optional[Dict] = None):
        self.copula = copula
        if np.any(data) < 0 or np.any(data) > 1:
            raise ValueError("data value must be between 0 and 1 !!")
        self.data = data
        self.x0 = x0
        self.method = method
        self.optimset = optimset or {}

    def fit(self):
        return minimize(self.log_likelihood, self.x0, **self.optimset)

    def log_likelihood(self, params: np.ndarray) -> float:
        self.copula.params = params
        return -np.sum(self.copula.logpdf(self.data))


