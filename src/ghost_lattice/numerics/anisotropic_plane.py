"""Elastic anisotropic extension using the six-parameter Chen 2016 cell energy.

The correction is calibrated on physically homogeneous sample-frame stiffness
tensors. A particle uses the homogeneous calibration corresponding to its own
material tensor. The actual heterogeneous affine residual is NEVER subtracted
in its entirety. No FEM field is used to construct or tune this operator.
"""

from pathlib import Path
import sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from .finite_elements import FEM, voronoi_mesh, rotate_plane
from .plane_lpm import Poly2D

V = np.sqrt(3) / 2
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
N0 = np.column_stack(
    [np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3)]
)
Q0 = np.column_stack([N0[:, 0] ** 2, N0[:, 1] ** 2, N0[:, 0] * N0[:, 1]])


def energy_matrix(k, T):
    return np.diag(k) / 4 + (T[:, None] + T[None, :]) / 8


MAP = []
for j in range(6):
    coef = np.eye(6)[j]
    kk = np.tile(coef[:3], 2)
    tt = np.tile(coef[3:], 2)
    C = Q0.T @ energy_matrix(kk, tt) @ Q0 / V
    MAP.append([C[a, b] for a, b in PAIRS])
MAP = np.asarray(MAP).T


def coefficients(C):
    coef = np.linalg.solve(MAP, np.array([C[a, b] for a, b in PAIRS]))
    return (coef[:3], coef[3:])


class AnisotropicCoupling:

    def __init__(
        self, W, theta, kind, material, correct=True, seed=7, seeds=None, shifts=None
    ):
        seeds = (
            np.array([[W / 4, W / 2], [3 * W / 4, W / 2]]) if seeds is None else seeds
        )
        shifts = np.array([[0.13, 0.27], [0.37, 0.61]]) if shifts is None else shifts
        self.ng = len(seeds)
        self.m = m = Poly2D(
            W, seeds, theta, shifts, kind=kind, band=6.0, coupling="energy", ret="scr"
        )
        mask = m.S == 0
        self.B = m.B[mask].tocsr()
        self.I = m.I[mask]
        self.n = m.n[mask]
        self.L = m.L0[mask]
        self.N = m.N
        self.nb = len(self.I)
        self.theta = np.asarray(theta)
        self.grain = m.grain
        g = self.grain[self.I]
        angle = np.arctan2(self.n[:, 1], self.n[:, 0]) - self.theta[g]
        self.family = np.rint(angle / (np.pi / 3)).astype(int) % 3
        self.S = sp.csr_matrix(
            (np.ones(self.nb), (self.I, np.arange(self.nb))), shape=(m.N, self.nb)
        )
        q = self.L[:, None] * np.column_stack(
            [self.n[:, 0] ** 2, self.n[:, 1] ** 2, self.n[:, 0] * self.n[:, 1]]
        )
        mm = np.zeros((m.N, 3, 3))
        np.add.at(mm, self.I, q[:, :, None] * q[:, None, :])
        c = np.einsum("nij,nj->ni", np.linalg.pinv(mm)[self.I], q)
        rows = (3 * self.I[:, None] + np.arange(3)[None]).ravel()
        cols = np.repeat(np.arange(self.nb), 3)
        self.G = sp.csr_matrix((c.ravel(), (rows, cols)), shape=(3 * m.N, self.nb))
        self.H = self.G @ self.B
        self.material = np.asarray(material)
        self.Kraw = self.stiffness(self.material)
        self.K = self.Kraw.copy()
        self.R = np.zeros((m.N, 2, 3))
        if correct:
            basis = []
            affine = np.stack(
                [
                    (m.X @ e.T).ravel()
                    for e in [
                        np.array([[1.0, 0], [0, 0]]),
                        np.array([[0.0, 0], [0, 1.0]]),
                        np.array([[0, 0.5], [0.5, 0]]),
                    ]
                ],
                axis=1,
            )
            for a, b in PAIRS:
                C = np.zeros((3, 3))
                C[a, b] = C[b, a] = 1.0
                Kh = self.stiffness(np.broadcast_to(C, (self.ng, 3, 3)))
                basis.append((Kh @ affine).reshape(m.N, 2, 3))
            for j, (a, b) in enumerate(PAIRS):
                self.R += self.material[self.grain, a, b, None, None] * basis[j]
            self.R[m.fixed[::2]] = 0.0
            ri = np.broadcast_to(
                2 * np.arange(m.N)[:, None, None] + np.arange(2)[None, :, None],
                self.R.shape,
            )
            ci = np.broadcast_to(
                3 * np.arange(m.N)[:, None, None] + np.arange(3)[None, None, :],
                self.R.shape,
            )
            rmat = sp.csr_matrix(
                (self.R.ravel(), (ri.ravel(), ci.ravel())), shape=(2 * m.N, 3 * m.N)
            )
            self.K = (self.Kraw - rmat @ self.H).tocsc()
        self.u = np.zeros(2 * m.N)

    def stiffness(self, Cg):
        kk = []
        tt = []
        for g, theta in enumerate(self.theta):
            kl, tl = coefficients(rotate_plane(Cg[g], -theta))
            kk.append(kl)
            tt.append(tl)
        kk, tt = (np.asarray(kk), np.asarray(tt))
        ownerg = self.grain[self.I]
        k = kk[ownerg, self.family]
        T = tt[ownerg, self.family]
        ST = sp.csr_matrix((T, (self.I, np.arange(self.nb))), shape=(self.N, self.nb))
        Kbond = sp.diags(k / 4) + (ST.T @ self.S + self.S.T @ ST) / 8
        return (self.B.T @ Kbond @ self.B).tocsc()

    def solve(self, E):
        m = self.m
        fixed = np.flatnonzero(m.fixed)
        free = np.flatnonzero(m.free)
        self.u[fixed] = (m.X @ E.T).ravel()[fixed]
        self.u[free] = sla.spsolve(
            self.K[free][:, free], -self.K[free][:, fixed] @ self.u[fixed]
        )
        self.strain = (self.H @ self.u).reshape(-1, 3)
        self.stress = np.einsum("nij,nj->ni", self.material[self.grain], self.strain)
        f = (self.K @ self.u).reshape(-1, 2)
        rel = np.linalg.norm(f[m.inner]) / np.linalg.norm(f[~m.inner])
        macro = f.T @ m.X / (m.W * m.W)
        return {
            "relative_force_residual": float(rel),
            "reaction_stress": macro.tolist(),
            "net_force_over_reactions": float(
                np.linalg.norm(f.sum(axis=0))
                / np.sum(np.linalg.norm(f[~m.inner], axis=1))
            ),
        }
