"""exp_gbtype_lib.py -- Sigma 3 (60 deg <111>) deviation of cubic grain pairs, shared by exp_gbtype.py / model_gbtype.py"""

import os
import sys
import numpy as np
from . import orientations as eg

AX111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
R3 = np.r_[np.cos(np.pi / 6), np.sin(np.pi / 6) * AX111]


def sigma3_dev(qa, qb):
    """deviation (deg) of the misorientation qa^-1 qb from Sigma 3, minimised over the symmetry operators on both sides"""
    d = eg.qmul(qa * np.array([1, -1, -1, -1]), qb)
    dinv = d * np.array([1, -1, -1, -1])
    sRs = eg.qmul(
        eg.qmul(eg.SYM[:, None, :], R3[None, None, :]), eg.SYM[None, :, :]
    ).reshape(-1, 4)
    w = np.abs(eg.qmul(dinv[:, None, :], sRs[None, :, :])[..., 0]).max(1)
    return np.degrees(2 * np.arccos(np.clip(w, 0, 1)))
