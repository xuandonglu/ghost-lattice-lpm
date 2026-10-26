"""Evaluate kinematics and continuum material histories at actual query points."""

import numpy as np
from .quadratic_elements import shapes
from .finite_elements import j2_update, cp_update
from .point_location import locate


def sample(fe, q, load=None, reverse=False):
    ids = locate(fe, q)
    v = fe["p"][fe["t"][ids]]
    gd = q.shape[1]
    affine = np.concatenate([np.ones((len(q), gd + 1, 1)), v], axis=2)
    inv = np.linalg.inv(affine)
    lam = np.einsum("ni,nij->nj", np.column_stack([np.ones(len(q)), q]), inv)
    grad = inv[:, 1:, :].transpose(0, 2, 1)
    N, dN = shapes(lam, grad)
    cells = fe["p2cells"][ids]
    grain = fe["grain"][ids]
    C = fe["material_C"][grain]
    dim = int(fe["field_dim"])
    ns = 3 if dim == 2 else 6
    ep = np.zeros((len(q), ns))
    alpha = np.zeros(len(q))
    law = str(fe["constitutive"])
    history = fe["u_history"]
    loads = fe["loads"]
    end = len(loads) - 1
    if load is not None:
        matches = np.flatnonzero(np.abs(loads - load) < 1e-07)
        if not len(matches):
            raise ValueError("Missing load checkpoint in FEM history")
        end = matches[-1] if reverse else matches[0]
    steps = [end] if law == "elastic" else range(end + 1)
    for step in steps:
        uq = history[step].reshape(-1, dim)[cells]
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
        if law == "elastic":
            stress = np.einsum("nij,nj->ni", C, strain)
        elif law == "cp":
            stress, ep, alpha, _ = cp_update(
                strain,
                ep,
                alpha,
                C,
                fe["slip_P"][grain],
                float(fe["cp_g0"]),
                float(fe["H"]),
                float(fe["dt"]) * float(fe["ey"]),
                tangent=False,
            )
        else:
            stress, ep, alpha, _ = j2_update(
                strain, ep, alpha, C, fe["yield_by_grain"][grain], float(fe["H"]), False
            )
    return {
        "strain": strain,
        "stress": stress,
        "alpha": alpha,
        "grain": grain,
        "u": np.einsum("na,nai->ni", N, uq),
    }
