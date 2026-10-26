"""gbp2d_raster.py -- gbp2d.Poly2D on a MEASURED grain map (raster of grain ids, e.g. from EBSD) instead of Voronoi seeds.
The module-level gbp2d.assign (Voronoi lookup) is temporarily replaced by a raster lookup while the model is built; the
seeds passed to Poly2D are the grain centroids (only their number and, for 'single', the first one matter); the
distance-to-boundary of every particle (m.dgb, used for GB / interior statistics) is taken from the raster.
  build(G, px, W, theta, shift, kind, **poly2d_kwargs) -> Poly2D model
  G (ny, nx) int grain ids 0..ng-1 covering [0, W) x [0, W) in model units, px = model length per pixel.
"""

import numpy as np
from scipy import ndimage
from . import plane_lpm as gbp2d


class Raster:

    def __init__(self, G, px):
        self.G, self.px = (np.asarray(G), float(px))
        self.ny, self.nx = self.G.shape
        b = np.zeros_like(self.G, bool)
        b[:, :-1] |= self.G[:, :-1] != self.G[:, 1:]
        b[:, 1:] |= self.G[:, :-1] != self.G[:, 1:]
        b[:-1, :] |= self.G[:-1, :] != self.G[1:, :]
        b[1:, :] |= self.G[:-1, :] != self.G[1:, :]
        self.dist = ndimage.distance_transform_edt(~b) * self.px
        ng = int(self.G.max()) + 1
        yy, xx = np.mgrid[0 : self.ny, 0 : self.nx]
        cnt = np.bincount(self.G.ravel(), minlength=ng)
        self.centroids = (
            np.column_stack(
                [
                    np.bincount(self.G.ravel(), xx.ravel() + 0.5, ng)
                    / np.maximum(cnt, 1),
                    np.bincount(self.G.ravel(), yy.ravel() + 0.5, ng)
                    / np.maximum(cnt, 1),
                ]
            )
            * self.px
        )

    def _ij(self, P):
        P = np.atleast_2d(P)
        ix = np.clip((P[:, 0] / self.px).astype(int), 0, self.nx - 1)
        iy = np.clip((P[:, 1] / self.px).astype(int), 0, self.ny - 1)
        return (iy, ix)

    def assign(self, P, seeds=None):
        if seeds is not None and len(seeds) == 1:
            return np.zeros(len(np.atleast_2d(P)), int)
        iy, ix = self._ij(P)
        return self.G[iy, ix]

    def dist_to_gb(self, P):
        iy, ix = self._ij(P)
        return self.dist[iy, ix]

    def other_grain(self, P):
        """grain on the other side of the nearest boundary, as in real/exp_gbtype.py for the HR-DIC data: the nearest
        boundary pixel (one distance transform with indices); if it lies in the point's own grain, its different-grain
        4-neighbour"""
        if not hasattr(self, "_oth"):
            G = self.G
            other = np.full(G.shape, -1, np.int64)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                sh = np.roll(G, (-dy, -dx), axis=(0, 1))
                valid = np.ones(G.shape, bool)
                if dy:
                    valid[-1 if dy == 1 else 0, :] = False
                if dx:
                    valid[:, -1 if dx == 1 else 0] = False
                m = valid & (sh != G) & (other < 0)
                other[m] = sh[m]
            _, (iy, ix) = ndimage.distance_transform_edt(other < 0, return_indices=True)
            gq = G[iy, ix]
            self._oth = np.where(gq != G, gq, other[iy, ix])
        iy, ix = self._ij(P)
        return self._oth[iy, ix]


def build(G, px, W, theta, shift, kind, **kw):
    R = Raster(G, px)
    old = gbp2d.assign
    gbp2d.assign = R.assign
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            m = gbp2d.Poly2D(W, R.centroids, theta, shift, kind=kind, **kw)
    finally:
        gbp2d.assign = old
    m.raster = R
    m.grain = R.assign(m.X)
    m.syp = (
        np.asarray(kw["sy"], float)[m.grain] if np.ndim(kw.get("sy", 0.0)) else m.syp
    )
    m.dgb = R.dist_to_gb(m.X)
    return m
