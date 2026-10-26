"""Quadratic Galerkin triangle reference, same continuum material updates.

Six-point degree-four quadrature; quadratic Lagrange shape functions. This is
an accuracy diagnostic alternative to brute-force P1 refinement, not an LPM
operator and not a change to the manuscript's physical problem.
"""

import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from .finite_elements import FEM, Mesh, _batch_c


def enrich(mesh):
    pairs = [
        (i, j) for i in range(mesh.t.shape[1]) for j in range(i + 1, mesh.t.shape[1])
    ]
    edges = np.concatenate([np.sort(mesh.t[:, ij], axis=1) for ij in pairs])
    unique, inv = np.unique(edges, axis=0, return_inverse=True)
    mids = (len(mesh.p) + inv).reshape(len(pairs), len(mesh.t)).T
    return (
        np.vstack([mesh.p, mesh.p[unique].mean(1)]),
        np.column_stack([mesh.t, mids]),
    )


def shapes(lam, grad):
    """lam (...,3), grad (...,3,2); return N (...,6), dN (...,6,2)."""
    nv = lam.shape[-1]
    nn = [lam[..., i] * (2 * lam[..., i] - 1) for i in range(nv)]
    gg = [(4 * lam[..., i, None] - 1) * grad[..., i, :] for i in range(nv)]
    for i, j in [(i, j) for i in range(nv) for j in range(i + 1, nv)]:
        nn.append(4 * lam[..., i] * lam[..., j])
        gg.append(
            4
            * (
                lam[..., i, None] * grad[..., j, :]
                + lam[..., j, None] * grad[..., i, :]
            )
        )
    return (np.stack(nn, axis=-1), np.stack(gg, axis=-2))


def quadrature(gd=2):
    if gd == 3:
        from itertools import permutations

        rows = []
        weights = []
        for values, w in [
            ([0.7217942490673264] + [0.0927352503108912] * 3, 0.07349304311636196),
            ([0.06734224221009817] + [0.3108859192633006] * 3, 0.1126879257180159),
            (
                [0.04550370412564965] * 2 + [0.45449629587435036] * 2,
                0.04254602077708147,
            ),
        ]:
            unique = sorted(set(permutations(values)))
            rows.extend(unique)
            weights.extend([w] * len(unique))
        return (np.asarray(rows), np.asarray(weights))
    a = 0.445948490915965
    b = 0.10810301816807
    c = 0.091576213509771
    d = 0.816847572980459
    return (
        np.array([[a, a, b], [a, b, a], [b, a, a], [c, c, d], [c, d, c], [d, c, c]]),
        np.array([0.223381589678011] * 3 + [0.109951743655322] * 3),
    )


class FEM_P2(FEM):

    def __init__(self, mesh, C, band=0.0, fixed=None, field_dim=None):
        self.base_mesh = mesh
        points, cells = enrich(mesh)
        nt = len(mesh.t)
        self.p2cells = cells
        self.mesh = Mesh(points, cells, mesh.grain)
        self.geometry_dim = gd = mesh.p.shape[1]
        self.dim = d = field_dim or gd
        self.nstrain = ns = 3 if d == 2 else 6
        nv = gd + 1
        nd = cells.shape[1] * d
        X = mesh.p[mesh.t]
        affine = np.concatenate([np.ones((nt, nv, 1)), X], axis=2)
        determinant = np.linalg.det(affine)
        grad = np.linalg.inv(affine)[:, 1:, :].transpose(0, 2, 1)
        self.lamq, w = quadrature(gd)
        nq = len(w)
        lam = np.broadcast_to(self.lamq[None], (nt, nq, nv))
        gr = np.broadcast_to(grad[:, None], (nt, nq, nv, gd))
        N, dN = shapes(lam, gr)
        B = np.zeros((nt, nq, ns, nd))
        for j in range(gd):
            B[:, :, j, j::d] = dN[:, :, :, j]
        pairs = [(0, 1)] if d == 2 else [(1, 2), (0, 2), (0, 1)]
        for s, (i, j) in enumerate(pairs, start=d):
            if j < gd:
                B[:, :, s, i::d] = dN[:, :, :, j]
            if i < gd:
                B[:, :, s, j::d] = dN[:, :, :, i]
        dofs = (cells[:, :, None] * d + np.arange(d)).reshape(nt, nd)
        rows = np.broadcast_to(np.arange(nt * nq * ns).reshape(nt, nq, ns, 1), B.shape)
        cols = np.broadcast_to(dofs[:, None, None, :], B.shape)
        keep = B != 0
        self.B = sp.coo_matrix(
            (B[keep], (rows[keep], cols[keep])), shape=(nt * nq * ns, len(points) * d)
        ).tocsr()
        self.center = np.einsum("qa,tad->tqd", self.lamq, X).reshape(-1, gd)
        self.volume = (
            np.abs(determinant[:, None]) / math.factorial(gd) * w[None]
        ).ravel()
        self.qgrain = np.repeat(mesh.grain, nq)
        C = np.asarray(C)
        self.C = _batch_c(C, nt * nq) if C.ndim == 2 else np.repeat(C, nq, axis=0)
        lo, hi = (points.min(0), points.max(0))
        self.fixed_nodes = (
            np.any(
                (points <= lo + band + 1e-10) | (points >= hi - band - 1e-10), axis=1
            )
            if fixed is None
            else np.asarray(fixed, bool)
        )
        self.fixed = np.flatnonzero(np.repeat(self.fixed_nodes, d))
        self.free = np.flatnonzero(~np.repeat(self.fixed_nodes, d))
        self.u = np.zeros(len(points) * d)
        self.ep = np.zeros((nt * nq, ns))
        self.alpha = np.zeros(nt * nq)
        self.stress = np.zeros_like(self.ep)
        self.strain = np.zeros_like(self.ep)
        self.lam = 0.0
        self._rows = np.repeat(np.arange(nt * nq * ns), ns)
        self._cols = np.repeat(np.arange(nt * nq), ns * ns) * ns + np.tile(
            np.arange(ns), nt * nq * ns
        )
        self.Kelastic = self.stiffness(self.C)
        self._elastic_lu = sla.splu(self.Kelastic[self.free][:, self.free])
        self._elastic_fc = self.Kelastic[self.free][:, self.fixed]

    def save(self, path, **extra):
        np.savez_compressed(
            path,
            p=self.mesh.p,
            t=self.base_mesh.t,
            p2cells=self.p2cells,
            grain=self.base_mesh.grain,
            qgrain=self.qgrain,
            element_order=2,
            field_dim=self.dim,
            X=self.center,
            volume=self.volume,
            u=self.u,
            strain=self.strain,
            stress=self.stress,
            epsp=self.ep,
            alpha=self.alpha,
            **extra,
        )
