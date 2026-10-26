import itertools
import numpy as np


def sites(R, lo, hi, shift=None):
    corners = np.array(list(itertools.product(*zip(lo, hi))), float)
    cc = corners @ R
    m = int(np.ceil(np.abs(cc).max())) + 3
    g = np.arange(-m, m + 1)
    I, J, K = np.meshgrid(g, g, g, indexing="ij")
    base = np.column_stack([I.ravel(), J.ravel(), K.ravel()]).astype(float)
    pts = np.vstack([base, base + 0.5])
    if shift is not None:
        pts = pts + np.asarray(shift, float)
    X = pts @ R.T
    eps = 1e-07
    ok = np.all(X >= np.array(lo) - eps, axis=1) & np.all(
        X < np.array(hi) - eps, axis=1
    )
    return X[ok]
