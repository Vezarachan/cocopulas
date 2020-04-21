# -*- coding: utf-8 -*-
"""
@Project: cocopulas
@File: utils.py
@Author: Lou Xiayin
@Date: 2020/4/21
@Purpose:
@Description: 
"""
import numpy as np


def split_matrix(x: np.ndarray):
    if len(x):
        return x[:, 0], x[:, 1]
    return np.array([0]), np.array([0])