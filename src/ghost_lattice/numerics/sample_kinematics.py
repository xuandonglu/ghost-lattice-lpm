"""Raw quadratic FEM displacement and strain at exact physical coordinates."""

import numpy as np
from .quadratic_elements import shapes
from .point_location import locate


def sample(fe, q):
    ids = locate(fe, q)
    v = fe["p"][fe["t"][ids]]
    gd = q.shape[1]
    inv = np.linalg.inv(np.concatenate([np.ones((len(q), gd + 1, 1)), v], axis=2))
    lam = np.einsum("ni,nij->nj", np.column_stack([np.ones(len(q)), q]), inv)
    N, dN = shapes(lam, inv[:, 1:, :].transpose(0, 2, 1))
    dim = int(fe["field_dim"])
    ns = 3 if dim == 2 else 6
    uq = fe["u"].reshape(-1, dim)[fe["p2cells"][ids]]
    du = np.einsum("nai,naj->nij", uq, dN)
    strain = np.zeros((len(q), ns))
    for k in range(gd):
        strain[:, k] = du[:, k, k]
    for k, (i, j) in enumerate(
        [(0, 1)] if dim == 2 else [(1, 2), (0, 2), (0, 1)], start=dim
    ):
        if j < gd:
            strain[:, k] += du[:, i, j]
        if i < gd:
            strain[:, k] += du[:, j, i]
    result = {
        "strain": strain,
        "grain": fe["grain"][ids],
        "u": np.einsum("na,nai->ni", N, uq),
    }
    if dim == 2:
        result["ges"] = np.hypot((strain[:, 0] - strain[:, 1]) / 2, strain[:, 2] / 2)
    return result
