# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: types.py
@Author: Lou Xiayin
@Date: 2020/4/21
@Purpose:
@Description: 
"""
from typing import Union, Iterable
import numpy as np

Array = Union[Iterable[int], Iterable[float], np.ndarray]
