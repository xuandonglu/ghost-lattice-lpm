"""Memory-bounded assembly of existing rotated models; unchanged force laws."""

import numpy as np
import scipy.sparse as sp
from .rotated_plane import CompactRotated
from .linear_solver import inverse
from .raster_geometry import Raster
from .equiaxed_lpm import Plast3DEq


class MeasuredCP(CompactRotated):

    def __init__(self, W, seeds, theta, shift, *, cp, **kwargs):
        super().__init__(W, seeds, theta, shift, **kwargs)
        self.plast = "cp"
        self.cp = dict(cp)
        gsl = np.where(self.gid >= 0, self.gid, self.grain)
        self.Pcp = np.asarray(cp["P"], float)[gsl]
        self.reset()


def raster_cp(G, px, W, theta, shift, kind, **kwargs):
    R = Raster(G, px)
    assert kwargs.pop("plast") == "cp"
    kwargs.pop("coupling", None)
    m = MeasuredCP(W, R.centroids, theta, shift, kind=kind, raster=R, **kwargs)
    assert m.coupling == ("energy" if kind == "ghost" else "force")
    return m


class Equiaxed(Plast3DEq):

    def set_weights(self, Wb):
        nb, N = (self.nb, self.N)
        Wc = sp.coo_matrix(Wb)
        r = np.arange(nb)
        rows = np.r_[np.repeat(r, 3), np.repeat(Wc.row, 3)]
        cols = np.r_[
            np.stack([3 * self.i, 3 * self.i + 1, 3 * self.i + 2], 1).ravel(),
            np.stack([3 * Wc.col, 3 * Wc.col + 1, 3 * Wc.col + 2], 1).ravel(),
        ]
        vals = np.r_[self.n.ravel(), (-Wc.data[:, None] * self.n[Wc.row]).ravel()]
        self.B = sp.csr_matrix((vals, (rows, cols)), shape=(nb, 3 * N))
        self.Efar = sp.csr_matrix(
            (Wc.data, (Wc.row, 2 * Wc.col + self.s[Wc.row])), shape=(nb, 2 * N)
        )
        self.Wb = sp.csr_matrix(Wb)
        del rows, cols, vals, Wc
        D = self.Sd @ self.B
        if self.coupling == "energy":
            self.K = (
                self.B.T @ self.B.multiply(self.Kn[:, None]) + self.T * (D.T @ D)
            ).tocsr()
        else:
            F = 2 * self.B.multiply(self.Kn[:, None]) + self.T * (
                (self.Eown + self.Efar) @ D
            )
            self.K = (self.Aown @ F).tocsr()
            del F
        del D
        self.lu = inverse(self.K[self.free][:, self.free], self)
        self.Kfc = self.K[self.free][:, self.fixed].tocsc()
