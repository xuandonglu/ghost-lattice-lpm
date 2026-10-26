"""Ghost-lattice grain boundaries for 3D BCC LPM (LPM-C lattice 4 force law, paper T sign).

Every particle keeps the PERFECT bond star of its own grain (8 x <111>, 6 x <100>). A star site that holds a particle
of the same grain gives an ordinary bond; otherwise the site is a GHOST site whose displacement (and multibody sums
D) are interpolated with linear-exact moving least squares from nearby particles of the grain that owns that
location. No overlap merging, no distance-based GB bonds.

Two couplings:
  'force'  : forces computed per particle from its own star (as LPM-C does particle by particle). Patch test exact,
             non-symmetric K, no momentum conservation at elastically mismatched GBs.
  'energy' : E = 1/2 sum_b Kn dL_b^2 + 1/2 T sum_(i,s) D_is^2 over all directed bonds incl. ghost bonds; reactions of
             ghost bonds go to the interpolation particles. Symmetric, momentum conserving, patch test NOT exact.
Interpolation: interp='mls' (Gaussian-weighted linear MLS over the nmls nearest, width hmls) or interp='tet'
(barycentric in the Delaunay tetrahedron of the nmls nearest that contains the site, i.e. linear-FEM interpolation with
4 supports; outside the hull: the tetrahedron with the least negative barycentric coordinate, still linear exact).
interp='taylor': u_site = u_j + G_j (site - x_j), j = nearest particle of the owning grain, G_j = least-squares
displacement gradient of j from ITS OWN ALIVE ORDINARY BONDS (G_j = sum_k (u_k - u_j) c_k^T, c_k = (sum d d^T)^-1 d_k).
Linear exact (patch test) and crack aware by construction: a broken bond leaves j's star, so the gradient never
reaches across a crack.  Call rebuild_supports() after bonds break.
Support source: support='grain' (candidates from the grain that owns the site) or support='all' (the nmls nearest
particles of ANY grain except the bond owner: sites in the gap between two grains are then enclosed, not extrapolated;
still linear exact, so the patch test is unchanged).
Face awareness: particles flagged in self.exclude (e.g. crack-face particles that lost a bond) are dropped from the
interpolation candidates while at least 5 candidates remain (call rebuild_supports() after changing it).
Crack awareness: the interpolation support of a ghost site is restricted to candidates connected (through intact
ordinary bonds among the candidates) to the candidate nearest to the site; fallback = nearest particle only.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, Delaunay
from .bcc_lattice import BV, BSH, params
from .bcc_sites import sites


def build_grains(frames, assign, lo, hi, Lz, shifts=None):
    Xs, G = ([], [])
    for g, Rm in enumerate(frames):
        sh = (
            None
            if shifts is None or shifts[g] is None
            else Rm.T @ np.asarray(shifts[g], float)
        )
        Xg = sites(Rm, (lo[0], lo[1], 0.0), (hi[0], hi[1], Lz), shift=sh)
        k = assign(Xg) == g
        Xs.append(Xg[k])
        G.append(np.full(k.sum(), g))
    return (np.vstack(Xs), np.concatenate(G))


class GhostLPM:

    def __init__(
        self,
        X,
        gid,
        frames,
        assign,
        mat,
        Lz,
        lo,
        hi,
        nmls=16,
        hmls=0.9,
        tol=0.05,
        crack_aware=True,
        interp="mls",
        support="grain",
    ):
        self.X, self.gid, self.N = (X, gid, len(X))
        self.Lz, self.nmls, self.hmls, self.crack_aware, self.interp = (
            Lz,
            nmls,
            hmls,
            crack_aware,
            interp,
        )
        self.support = support
        p = params(mat["C11"], mat["C12"], mat["C44"], a=1.0, mapping="paper")
        self.p, self.V = (p, p["V"])
        self._off = np.array([lo[0] - 10, lo[1] - 10, 0.0])
        self.trees, self.idx = ({}, {})
        for g in np.unique(gid):
            k = np.nonzero(gid == g)[0]
            Y = X[k] - self._off
            Y[:, 2] = np.mod(Y[:, 2], Lz)
            self.trees[g] = cKDTree(Y, boxsize=np.array([1000000.0, 1000000.0, Lz]))
            self.idx[g] = k
        Ya = X - self._off
        Ya[:, 2] = np.mod(Ya[:, 2], Lz)
        self.tree_all = cKDTree(Ya, boxsize=np.array([1000000.0, 1000000.0, Lz]))
        owner, shell, nvec, L0, ghost, nbr, site, sgrain = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(self.N):
            g = gid[i]
            for k in range(14):
                bv = frames[g] @ BV[k]
                s = X[i] + bv
                d, j = self.trees[g].query(self._rel(s), k=1)
                if d < tol:
                    isg, jj, h = (False, self.idx[g][j], g)
                else:
                    if not (lo[0] <= s[0] < hi[0] and lo[1] <= s[1] < hi[1]):
                        continue
                    isg, jj, h = (True, -1, int(assign(s[None])[0]))
                owner.append(i)
                shell.append(BSH[k])
                nvec.append(-bv / np.linalg.norm(bv))
                L0.append(np.linalg.norm(bv))
                ghost.append(isg)
                nbr.append(jj)
                site.append(s)
                sgrain.append(h)
        self.nb = len(owner)
        self.i = np.array(owner)
        self.s = np.array(shell)
        self.n = np.array(nvec)
        self.d0 = np.array(L0)
        self.ghost = np.array(ghost)
        self.j = np.array(nbr)
        self.site = np.array(site)
        self.sgrain = np.array(sgrain)
        self.Kn = p["Kn"][self.s]
        self.T = p["T"]
        self.pair = np.arange(self.nb)
        o = np.nonzero(~self.ghost)[0]
        key = np.minimum(self.i[o], self.j[o]) * self.N + np.maximum(
            self.i[o], self.j[o]
        )
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        self.pair[o] = o[first][inv]
        self.alive = np.ones(self.nb, bool)
        self.exclude = np.zeros(self.N, bool)
        self.Gi = sp.csr_matrix(
            (np.ones(self.nb), (np.arange(self.nb), 2 * self.i + self.s)),
            shape=(self.nb, 2 * self.N),
        )
        self.G = sp.csr_matrix(
            (np.ones(self.nb), (2 * self.i + self.s, np.arange(self.nb))),
            shape=(2 * self.N, self.nb),
        )
        self.A = sp.csr_matrix(
            (
                self.n.ravel(),
                (
                    np.repeat(np.arange(self.nb), 3),
                    np.stack([3 * self.i, 3 * self.i + 1, 3 * self.i + 2], 1).ravel(),
                ),
            ),
            shape=(self.nb, 3 * self.N),
        )
        self.rebuild_supports()

    def _rel(self, s):
        r = s - self._off
        return np.array([r[0], r[1], np.mod(r[2], self.Lz)])

    def _adjacency(self):
        o = np.nonzero(~self.ghost & self.alive[self.pair])[0]
        return sp.csr_matrix(
            (np.ones(len(o)), (self.i[o], self.j[o])), shape=(self.N, self.N)
        )

    def _mls(self, cand, s):
        Y = self.X[cand] - s
        Y[:, 2] -= self.Lz * np.round(Y[:, 2] / self.Lz)
        d = np.linalg.norm(Y, axis=1)
        w = np.exp(-((d / self.hmls) ** 2))
        P = np.column_stack([np.ones(len(cand)), Y])
        Am = P.T @ (w[:, None] * P)
        if len(cand) < 6 or np.linalg.cond(Am) > 100000000.0:
            return [(cand[np.argmin(d)], 1.0)]
        phi = w * (P @ np.linalg.solve(Am, np.array([1.0, 0, 0, 0])))
        return list(zip(cand, phi))

    def _tet(self, cand, s):
        Y = self.X[cand] - s
        Y[:, 2] -= self.Lz * np.round(Y[:, 2] / self.Lz)
        if len(cand) < 5:
            return self._mls(cand, s)
        try:
            tri = Delaunay(Y, qhull_options="QJ Qbb Qc")
        except Exception:
            return self._mls(cand, s)
        best, bv, bk = (-np.inf, None, None)
        for k, sx in enumerate(tri.simplices):
            E = (Y[sx[1:]] - Y[sx[0]]).T
            if abs(np.linalg.det(E)) < 1e-06:
                continue
            lam = np.linalg.solve(E, -Y[sx[0]])
            bc = np.concatenate([[1 - lam.sum()], lam])
            if bc.min() > best:
                best, bv, bk = (bc.min(), bc, sx)
                if best >= -1e-12:
                    break
        if bk is None:
            return self._mls(cand, s)
        return list(zip(cand[bk], bv))

    def _taylor(self, j, s):
        bl = self._own[self._optr[j] : self._optr[j + 1]]
        bl = bl[~self.ghost[bl] & self.alive[self.pair[bl]]]
        if len(bl) < 3:
            return None
        D = -self.n[bl] * self.d0[bl][:, None]
        A = D.T @ D
        if np.linalg.cond(A) > 1000000.0:
            return None
        r = s - self.X[j]
        r[2] -= self.Lz * np.round(r[2] / self.Lz)
        wk = D @ np.linalg.solve(A, r)
        return [(j, 1.0 - wk.sum())] + list(zip(self.j[bl], wk))

    def rebuild_supports(self):
        adj = self._adjacency() if self.crack_aware else None
        if not hasattr(self, "_own"):
            self._own = np.argsort(self.i, kind="stable")
            self._optr = np.searchsorted(self.i[self._own], np.arange(self.N + 1))
        rows_B, cols_B, vals_B, rG, cG, vG = ([], [], [], [], [], [])
        self.nfallback = 0
        for b in range(self.nb):
            i = self.i[b]
            if not self.ghost[b]:
                sup = [(self.j[b], 1.0)]
            else:
                h = self.sgrain[b]
                if self.support == "all":
                    d, jl = self.tree_all.query(
                        self._rel(self.site[b]), k=self.nmls + 1
                    )
                    cand = jl[jl != i][: self.nmls]
                else:
                    d, jl = self.trees[h].query(self._rel(self.site[b]), k=self.nmls)
                    cand = self.idx[h][jl]
                keep = ~self.exclude[cand]
                if keep.sum() >= 5:
                    cand = cand[keep]
                if adj is not None:
                    sub = adj[cand][:, cand]
                    nc, lab = connected_components(sub, directed=False)
                    cand = cand[lab == lab[0]]
                if self.interp == "taylor":
                    sup = self._taylor(cand[0], self.site[b]) or self._mls(
                        cand, self.site[b]
                    )
                else:
                    sup = (
                        self._tet(cand, self.site[b])
                        if self.interp == "tet"
                        else self._mls(cand, self.site[b])
                    )
                if len(sup) == 1:
                    self.nfallback += 1
            for m, w in sup:
                rows_B += [b, b, b]
                cols_B += [3 * m, 3 * m + 1, 3 * m + 2]
                vals_B += list(-w * self.n[b])
                rG.append(b)
                cG.append(2 * m + self.s[b])
                vG.append(w)
            rows_B += [b, b, b]
            cols_B += [3 * i, 3 * i + 1, 3 * i + 2]
            vals_B += list(self.n[b])
        self.B0 = sp.csr_matrix((vals_B, (rows_B, cols_B)), shape=(self.nb, 3 * self.N))
        self.Gj = sp.csr_matrix((vG, (rG, cG)), shape=(self.nb, 2 * self.N))

    def stiffness(self, mode="force"):
        W = sp.diags(self.alive[self.pair].astype(float))
        B = (W @ self.B0).tocsr()
        GG = (self.Gi + self.Gj) @ self.G
        M = sp.diags(2 * self.Kn) + self.T * GG
        self._B, self._M = (B, (W @ M).tocsr())
        if mode == "force":
            return (self.A.T @ self._M @ B).tocsr()
        GB = (self.G @ B).tocsr()
        return (B.T @ sp.diags(self.Kn) @ B + self.T * (GB.T @ GB)).tocsr()

    def bond_force(self, u):
        dL = self._B @ u.ravel()
        return (self._M @ dL, dL)

    def stress(self, u):
        F, _ = self.bond_force(u)
        c = 0.5 / self.V * self.d0 * F
        out = np.zeros((self.N, 6))
        for k, (a, b) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
            np.add.at(out[:, k], self.i, c * self.n[:, a] * self.n[:, b])
        return out

    def solve_dirichlet(self, K, fixed, u_fixed):
        free = ~fixed
        u = np.zeros(3 * self.N)
        u[fixed] = u_fixed[fixed]
        u[free] = spla.spsolve(K[free][:, free].tocsc(), -K[free][:, fixed] @ u[fixed])
        return u.reshape(-1, 3)
