# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: base.py
@Author: Lou Xiayin
@Date: 2020/4/21
@Purpose:
@Description: 
"""
from cocopulas.core.types import Array
from typing import Union, Optional, Dict
import numpy as np
from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod


class BaseCopula(ABC):
    """
    The base class of copula
    """
    @abstractmethod
    def fit(self, data: Array, x0: np.ndarray = None, method="ml", optimset: Optional[Dict]=None):
        # TODO implement parameter estimation
        raise NotImplementedError

    @abstractmethod
    def check_fit(self):
        raise NotImplementedError

    @abstractmethod
    def cdf(self, x: Array) -> Union[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def ppf(self, x: Array) -> Union[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x: Array) -> Union[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, x: Array) -> Union[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self):
        raise NotImplementedError

    @params.setter
    @abstractmethod
    def params(self, value: float):
        raise NotImplementedError

