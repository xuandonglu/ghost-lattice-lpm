"""Memory-efficient assembly of the ORIGINAL planar rotated lattice operators.

Same sites, distance merging, MLS supports/weights, bond stars and force laws
as gbp2d.Poly2D. No trace reconstruction or weight optimization. Geometry and
sparse assembly are batched; equivalent energy assembly avoids bond-space Fop.
"""

import math
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from .plane_lpm import Poly2D, STAR, KN, TT, VV, mls_weights, merge_overlaps, R01, R12
from .linear_solver import inverse
from .finite_elements import _clip


def sites(theta, shift, lo, hi, W):
    c, s = (math.cos(theta), math.sin(theta))
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    local = corners @ np.array([[c, -s], [s, c]]) - np.asarray(shift)
    imin = int(np.floor(local[:, 0].min())) - 2
    imax = int(np.ceil(local[:, 0].max())) + 2
    jmin = int(np.floor(local[:, 1].min() / (np.sqrt(3) / 2))) - 2
    jmax = int(np.ceil(local[:, 1].max() / (np.sqrt(3) / 2))) + 2
    ii, jj = np.meshgrid(
        np.arange(imin, imax + 1), np.arange(jmin, jmax + 1), indexing="ij"
    )
    X = np.column_stack(
        [(ii + 0.5 * (jj % 2)).ravel(), (np.sqrt(3) / 2 * jj).ravel()]
    ) + np.asarray(shift)
    X = X @ np.array([[c, s], [-s, c]])
    return X[np.all((X >= 0) & (X < W), axis=1) & np.all((X >= lo) & (X <= hi), axis=1)]


def chunk_assign(X, assign):
    return (
        np.concatenate([assign(X[k : k + 32768]) for k in range(0, len(X), 32768)])
        if len(X)
        else np.empty(0, int)
    )


def voronoi_assign(X, seeds):
    return np.argmin(((X[:, None, :] - seeds[None, :, :]) ** 2).sum(-1), axis=1)


def bounds(seeds, W, raster=None):
    result = []
    for g in range(len(seeds)):
        if raster is not None:
            iy, ix = np.where(raster.G == g)
            result.append(
                None
                if not len(ix)
                else (
                    np.array([ix.min(), iy.min()]) * raster.px,
                    (np.array([ix.max(), iy.max()]) + 1) * raster.px,
                )
            )
            continue
        poly = np.array([[0.0, 0.0], [W, 0.0], [W, W], [0.0, W]])
        for h in range(len(seeds)):
            if h == g:
                continue
            poly = _clip(
                poly,
                seeds[h] - seeds[g],
                (seeds[h] @ seeds[h] - seeds[g] @ seeds[g]) / 2,
            )
            if len(poly) < 3:
                break
        result.append(None if len(poly) < 3 else (poly.min(0), poly.max(0)))
    return result


def strain_operator(m):
    N, nb = (m.N, m.nb)
    M = np.zeros((N, 3, 3))
    for start in range(0, nb, 131072):
        sl = slice(start, start + 131072)
        v = m.L0[sl, None] * m.nnL[sl] * np.array([1.0, 1.0, 2.0])
        np.add.at(M, m.I[sl], v[:, :, None] * v[:, None, :])
    for start in range(0, N, 65536):
        M[start : start + 65536] = np.linalg.pinv(M[start : start + 65536])
    count = np.bincount(m.I, minlength=N)
    ownerptr = np.r_[0, np.cumsum(count)]
    indptr = np.r_[0, np.cumsum(np.repeat(count, 3))].astype(np.int32)
    columns = np.empty(3 * nb, np.int32)
    data = np.empty(3 * nb)
    for start in range(0, nb, 131072):
        stop = min(start + 131072, nb)
        ids = np.arange(start, stop)
        I = m.I[start:stop]
        v = m.L0[start:stop, None] * m.nnL[start:stop] * np.array([1.0, 1.0, 2.0])
        coef = np.einsum("nij,nj->ni", M[I], v)
        rank = ids - ownerptr[I]
        for component in range(3):
            dest = indptr[3 * I + component] + rank
            columns[dest] = ids
            data[dest] = coef[:, component]
    return sp.csr_matrix((data, columns, indptr), shape=(3 * N, nb))


class CompactRotated(Poly2D):

    def __init__(
        self,
        W,
        seeds,
        theta,
        shift,
        kind="ghost",
        band=6.0,
        sy=0.0015,
        H=0.02,
        hole=None,
        raster=None,
        ret="scr",
        **kwargs,
    ):
        if kind not in ["ghost", "j04"]:
            raise ValueError("Only original ghost/merged construction")
        self.W = W
        self.seeds = np.asarray(seeds)
        self.kind = kind
        self.coupling = "energy" if kind == "ghost" else "force"
        self.far_state = "sa"
        self.hole = hole
        self.band = band
        self.ret = ret
        self.plast = "j2"
        self.cp = {}
        self._assign = (
            (lambda X: raster.assign(X))
            if raster is not None
            else lambda X: voronoi_assign(X, self.seeds)
        )
        boxes = bounds(self.seeds, W, raster)
        points = []
        labels = []
        for g, box in enumerate(boxes):
            if box is None:
                continue
            X = sites(
                theta[g],
                shift[g],
                np.maximum(box[0] - 1e-07, 0),
                np.minimum(box[1] + 1e-07, W),
                W,
            )
            X = X[chunk_assign(X, self._assign) == g]
            if hole is not None:
                X = X[np.linalg.norm(X - np.asarray(hole[:2]), axis=1) >= hole[2]]
            points.append(X)
            labels.append(np.full(len(X), g, np.int32))
        X = np.vstack(points)
        gid = np.concatenate(labels)
        if kind == "j04":
            X, gid = merge_overlaps(X, gid, 0.6)
        self.X = X
        self.gid = gid.astype(np.int32)
        self.N = N = len(X)
        self.grain = chunk_assign(X, self._assign)
        supports = []
        weights = []
        blocks = []
        if kind == "j04":
            pairs = cKDTree(X).query_pairs(R12, output_type="ndarray")
            I = np.r_[pairs[:, 0], pairs[:, 1]].astype(np.int32)
            J = np.r_[pairs[:, 1], pairs[:, 0]].astype(np.int32)
            order = np.argsort(I, kind="stable")
            I, J = (I[order], J[order])
            delta = X[I] - X[J]
            L = np.linalg.norm(delta, axis=1)
            self.I = I
            self.J = J
            self.L0 = L
            self.n = delta / L[:, None]
            self.S = (L >= R01).astype(np.int8)
            self.Gi = np.full(len(I), -1, np.int32)
            self.ghost = np.zeros(len(I), bool)
            self.ng = 0
            key = I.astype(np.int64) * N + J
            sort = np.argsort(key)
            reverse = J.astype(np.int64) * N + I
            self.rev = sort[np.searchsorted(key[sort], reverse)].astype(np.int32)
            del key, sort, reverse, pairs, order, delta
        else:
            trees = {
                g: (cKDTree(X[gid == g]), np.flatnonzero(gid == g))
                for g in np.unique(gid)
            }
            alltree = None
            for g, (tree, owners) in trees.items():
                for k, (length, angle) in enumerate(STAR):
                    vec = length * np.array(
                        [math.cos(angle + theta[g]), math.sin(angle + theta[g])]
                    )
                    query = X[owners] + vec
                    valid = np.ones(len(owners), bool)
                    if hole is not None:
                        valid = (
                            np.linalg.norm(query - np.asarray(hole[:2]), axis=1)
                            >= hole[2]
                        )
                    inside = np.all((query >= -1e-09) & (query < W + 1e-09), axis=1)
                    target = np.full(len(owners), g)
                    target[inside] = chunk_assign(query[inside], self._assign)
                    same = valid & (target == g)
                    distance, near = tree.query(query[same])
                    take = np.flatnonzero(same)[distance <= 1e-06]
                    if len(take):
                        blocks.append(
                            (
                                owners[take],
                                trees[g][1][near[distance <= 1e-06]],
                                np.full(len(take), -1),
                                np.full(len(take), k),
                            )
                        )
                    for h in np.unique(target[valid & (target != g)]):
                        take = np.flatnonzero(valid & (target == h))
                        _, near = trees[h][0].query(
                            query[take], k=min(12, len(trees[h][1]))
                        )
                        near = np.asarray(near).reshape(len(take), -1)
                        sup = trees[h][1][near]
                        gi = []
                        for row, index in enumerate(take):
                            ids = sup[row]
                            offset = X[ids] - query[index]
                            if (
                                len(ids) < 3
                                or np.linalg.matrix_rank(
                                    np.column_stack([np.ones(len(ids)), offset]),
                                    tol=1e-06,
                                )
                                < 3
                            ):
                                if alltree is None:
                                    alltree = cKDTree(X)
                                _, ids = alltree.query(query[index], k=min(36, N))
                                ids = np.atleast_1d(ids)
                                ids = ids[gid[ids] != g][:12]
                                offset = X[ids] - query[index]
                            w = mls_weights(offset)
                            keep = np.abs(w) > 1e-14
                            gi.append(len(supports))
                            supports.append(ids[keep].astype(np.int32))
                            weights.append(w[keep])
                        blocks.append(
                            (
                                owners[take],
                                np.full(len(take), -1),
                                np.asarray(gi),
                                np.full(len(take), k),
                            )
                        )
            I = np.concatenate([b[0] for b in blocks]).astype(np.int32)
            J = np.concatenate([b[1] for b in blocks]).astype(np.int32)
            Gi = np.concatenate([b[2] for b in blocks]).astype(np.int32)
            direction = np.concatenate([b[3] for b in blocks]).astype(np.int8)
            order = np.argsort(12 * I.astype(np.int64) + direction)
            self.I = I[order]
            self.J = J[order]
            self.Gi = Gi[order]
            direction = direction[order]
            frames = np.array(
                [[[math.cos(a + t), math.sin(a + t)] for r, a in STAR] for t in theta]
            )
            self.n = -frames[gid[self.I], direction]
            self.L0 = np.array([r for r, a in STAR])[direction]
            self.S = (direction >= 6).astype(np.int8)
            self.ghost = self.Gi >= 0
            self.ng = len(supports)
            table = np.full((N, 12), -1, np.int32)
            table[self.I, direction] = np.arange(len(self.I), dtype=np.int32)
            opposite = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
            self.rev = np.full(len(self.I), -1, np.int32)
            real = ~self.ghost
            self.rev[real] = table[self.J[real], opposite[direction[real]]]
            self.wst = np.where(table[self.I, opposite[direction]] >= 0, 0.5, 1.0)
            del table, blocks, trees, alltree, I, J, Gi, order, direction
        self.nb = nb = len(self.I)
        if (self.rev[~self.ghost] < 0).any() or not np.array_equal(
            self.I[self.rev[~self.ghost]], self.J[~self.ghost]
        ):
            raise RuntimeError("Missing reciprocal real bond")
        if self.ng:
            ptr = np.r_[0, np.cumsum([len(s) for s in supports])].astype(np.int32)
            self.Wg = sp.csr_matrix(
                (np.concatenate(weights), np.concatenate(supports), ptr),
                shape=(self.ng, N),
            )
        else:
            self.Wg = sp.csr_matrix((0, N))
        del supports, weights
        if kind == "j04":
            self.wst = np.ones(nb)
            count = np.bincount(self.I, minlength=N)
            ptr = np.r_[0, np.cumsum(count)]
            for degree in np.unique(count):
                if not degree:
                    continue
                owners = np.flatnonzero(count == degree)
                for start in range(0, len(owners), 4096):
                    o = owners[start : start + 4096]
                    b = ptr[o, None] + np.arange(degree)
                    v = self.n[b]
                    c = np.einsum("nai,nbi->nab", v, v)
                    labels = self.S[b]
                    same = labels[:, :, None] == labels[:, None, :]
                    self.wst[b] = np.where(((c < -0.99) & same).any(2), 0.5, 1.0)
        self._build_B_compact()
        rb = np.arange(nb, dtype=np.int32)
        self.Aown = sp.csr_matrix(
            (
                self.n.ravel(),
                (
                    np.column_stack([2 * self.I, 2 * self.I + 1]).ravel(),
                    np.repeat(rb, 2),
                ),
            ),
            shape=(2 * N, nb),
        )
        self.nnL = np.column_stack(
            [self.n[:, 0] ** 2, self.n[:, 1] ** 2, self.n[:, 0] * self.n[:, 1]]
        )
        boundary = np.any((X < band) | (X > W - band), axis=1)
        self.fixed = np.repeat(boundary, 2)
        self.free = ~self.fixed
        self.inner = ~boundary
        D = self.Sd @ self.B
        self.K = (KN * (self.B.T @ self.B) + TT * (D.T @ D)).tocsr()
        del D
        self.lu = inverse(self.K[self.free][:, self.free], self)
        self.Kfc = self.K[self.free][:, self.fixed].tocsc()
        self.sy = sy
        self.H = H
        self.syp = np.asarray(sy)[self.grain] if np.ndim(sy) else np.full(N, float(sy))
        self.Ci = self.star_stiffness()
        self.reset()
        if raster is not None:
            self.raster = raster
            self.dgb = raster.dist_to_gb(X)
        else:
            d, near = cKDTree(self.seeds).query(X, k=2)
            self.dgb = (d[:, 1] ** 2 - d[:, 0] ** 2) / (
                2
                * np.linalg.norm(
                    self.seeds[near[:, 1]] - self.seeds[near[:, 0]], axis=1
                )
            )
        self._assign = None

    def _build_B_compact(self):
        N, nb = (self.N, self.nb)
        count = np.full(nb, 4, np.int32)
        ghost = np.flatnonzero(self.ghost)
        sizes = np.diff(self.Wg.indptr)
        count[ghost] = 2 + 2 * sizes[self.Gi[ghost]]
        ptr = np.r_[0, np.cumsum(count)].astype(np.int32)
        cols = np.empty(int(ptr[-1]), np.int32)
        data = np.empty(int(ptr[-1]))
        cols[ptr[:-1]] = 2 * self.I
        cols[ptr[:-1] + 1] = 2 * self.I + 1
        data[ptr[:-1]] = self.n[:, 0]
        data[ptr[:-1] + 1] = self.n[:, 1]
        real = np.flatnonzero(~self.ghost)
        cols[ptr[real] + 2] = 2 * self.J[real]
        cols[ptr[real] + 3] = 2 * self.J[real] + 1
        data[ptr[real] + 2] = -self.n[real, 0]
        data[ptr[real] + 3] = -self.n[real, 1]
        for k in range(int(sizes.max()) if len(sizes) else 0):
            take = ghost[sizes[self.Gi[ghost]] > k]
            p = self.Wg.indptr[self.Gi[take]] + k
            for j in range(2):
                dest = ptr[take] + 2 + 2 * k + j
                cols[dest] = 2 * self.Wg.indices[p] + j
                data[dest] = -self.Wg.data[p] * self.n[take, j]
        self.B = sp.csr_matrix((data, cols, ptr), shape=(nb, 2 * N))
        self.B.sort_indices()
        rb = np.arange(nb, dtype=np.int32)
        self.Sd = sp.csr_matrix(
            (np.ones(nb), (2 * self.I + self.S, rb)), shape=(2 * N, nb)
        )
        self.Eown = self.Sd.T.tocsr()
        self.Efar = sp.csr_matrix(
            (np.ones(len(real)), (real, 2 * self.J[real] + self.S[real])),
            shape=(nb, 2 * N),
        )

    def enable_gfc(self):
        if self.coupling != "energy":
            raise ValueError("GFC only on original ghost coupling")
        self.Gstrain = strain_operator(self)
        R = np.zeros((self.N, 2, 3))
        for k, E in enumerate(
            [
                np.diag([1.0, 0.0]),
                np.diag([0.0, 1.0]),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
            ]
        ):
            dL = self.B @ (self.X @ E.T).ravel()
            R[:, :, k] = (
                self.B.T @ (KN * dL + TT * (self.Eown @ (self.Sd @ dL)))
            ).reshape(-1, 2)
        R[self.fixed[::2]] = 0.0
        if self.hole is not None:
            R[
                np.linalg.norm(self.X - np.asarray(self.hole[:2]), axis=1)
                <= self.hole[2] + 3.5
            ] = 0.0
        self.gfc = R
