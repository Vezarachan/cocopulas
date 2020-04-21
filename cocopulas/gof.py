# -*- coding: utf-8 -*-
"""
@Project: 
@File: gof.py
@Author: 
@Date: 2020/4/21
@Purpose:
@Description: 
"""
import numpy as np
from typing import Dict


def goodness_of_fit(predicted: np.ndarray, observed: np.ndarray, k: int, n: int) -> Dict[str, float]:
    mean = np.mean(observed)
    sse = np.sum(np.power(predicted - observed, 2))
    sst = np.sum(np.power(observed - mean, 2))
    ssr = np.sum(np.power(predicted - mean, 2))
    r2 = 1 - np.divide(sse, sst)
    aic = 2 * k + n * np.log(np.divide(sse, n))
    bic = k * np.log(n) + n * np.log(np.divide(sse, n))
    hqic = np.log(np.log(n)) * k + n * np.log(np.divide(sse, n))
    return {
        "SSE": sse,
        "SSR": ssr,
        "SST": sst,
        "R2": r2,
        "AIC": aic,
        "BIC": bic,
        "HQIC": hqic
    }