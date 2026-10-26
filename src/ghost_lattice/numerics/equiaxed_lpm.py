"""gbp3d_eq.py -- fully 3D, EQUIAXED polycrystal version of gbp3d (Paper A, 3D check beyond columnar grains): a cube
[0, L]^3 (no periodicity), 3D Voronoi grains with arbitrary 3D lattice orientations and rigid shifts, affine Dirichlet
band on all six faces.  Same constructions, couplings, returns and constitutive laws as gbp3d.Plast3D:
  'single' : one lattice (orientation frames[0]) everywhere, grain-wise material through the Voronoi grain (reference);
  'j04'    : J04 / PX construction (overlap merge 0.52 a, bonds by distance), non-periodic;
  'ghost'  : ghost lattice with complete 14-bond stars; sites outside the cube are dropped (free surface), sites in
             another grain are ghost sites interpolated from that grain (linear MLS).
Nothing of gbp3d / ghost3d is modified: the cube builders, a GhostLPM subclass with the z bounds and a Plast3D subclass
with a 3D assignment / band / distance-to-boundary live here; the periodic length is set to a value far beyond the cube
(BIG), which turns every periodic wrap of the inherited code into a no-op."""

import numpy as np
import scipy.sparse as sp
from . import columnar_lpm as gbp3d
from .columnar_lpm import Plast3D, bcc_slip, VO, gc, params
from .bcc_lattice import BCCModel
from .grain_geometry import merge_overlaps
from .ghost_bcc import GhostLPM, BV, BSH
from .bcc_sites import sites

BIG = 1000000.0


def random_rotation(rng):
    """uniform random rotation (unit quaternion -> matrix)"""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def microstructure(L, ng, seed):
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, L, (ng, 3))
    frames = [random_rotation(rng) for _ in range(ng)]
    shifts = [tuple(rng.uniform(0, 1, 3)) for _ in range(ng)]
    return (seeds, frames, shifts)


def cube_grains(frames, assign, L, shifts=None):
    Xs, G = ([], [])
    for g, Rm in enumerate(frames):
        sh = (
            None
            if shifts is None or shifts[g] is None
            else Rm.T @ np.asarray(shifts[g], float)
        )
        Xg = sites(Rm, (0.0, 0.0, 0.0), (L, L, L), shift=sh)
        k = assign(Xg) == g
        Xs.append(Xg[k])
        G.append(np.full(k.sum(), g))
    return (np.vstack(Xs), np.concatenate(G))


class GhostCube(GhostLPM):
    """ghost3d.GhostLPM on a non-periodic cube: identical construction, plus 'site outside the cube in z -> no bond'"""

    def __init__(self, X, gid, frames, assign, mat, L, nmls=16, hmls=0.9, tol=0.05):
        from scipy.spatial import cKDTree

        Lz = BIG
        lo, hi = ((0.0, 0.0, 0.0), (L, L, L))
        self.X, self.gid, self.N = (X, gid, len(X))
        self.Lz, self.nmls, self.hmls, self.crack_aware, self.interp = (
            Lz,
            nmls,
            hmls,
            False,
            "mls",
        )
        self.support = "grain"
        p = params(mat["C11"], mat["C12"], mat["C44"], a=1.0, mapping="paper")
        self.p, self.V = (p, p["V"])
        self._off = np.array([lo[0] - 10, lo[1] - 10, 0.0])
        self.trees, self.idx = ({}, {})
        for g in np.unique(gid):
            k = np.nonzero(gid == g)[0]
            Y = X[k] - self._off
            self.trees[g] = cKDTree(Y, boxsize=np.array([1000000.0, 1000000.0, Lz]))
            self.idx[g] = k
        self.tree_all = cKDTree(
            X - self._off, boxsize=np.array([1000000.0, 1000000.0, Lz])
        )
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
                    if not (
                        lo[0] <= s[0] < hi[0]
                        and lo[1] <= s[1] < hi[1]
                        and (lo[2] <= s[2] < hi[2])
                    ):
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


class Plast3DEq(Plast3D):
    """gbp3d.Plast3D on the equiaxed cube (3D Voronoi assignment, band on all six faces, 3D distance to the boundary)"""

    def __init__(
        self,
        kind,
        L,
        seeds,
        frames,
        shifts,
        mat=None,
        band=4.0,
        sy=0.2,
        H=0.0,
        ret="scr",
        far="sa",
        nmls=16,
        hmls=0.9,
        coupling="energy",
        plast="j2",
        cp=None,
    ):
        mat = mat or gc.material("iso")
        p = params(mat["C11"], mat["C12"], mat["C44"], a=1.0, mapping="paper")
        self.p, self.V, self.T = (p, p["V"], p["T"])
        self.Gsh = mat["C44"]
        self.kind, self.far, self.ret, self.L, self.coupling = (
            kind,
            far,
            ret,
            L,
            coupling,
        )
        seeds_all = np.asarray(seeds, float)

        def assign_all(X):
            return np.argmin(((X[:, None, :] - seeds_all[None]) ** 2).sum(-1), 1)

        if kind == "single":
            frames, shifts = (frames[:1], shifts[:1])
            assign = lambda X: np.zeros(len(X), int)
        else:
            assign = assign_all
        if kind == "j04":
            X, gid = cube_grains(frames, assign, L, shifts=shifts)
            X, gid, nm = merge_overlaps(X, gid, 0.52, BIG)
            bm = BCCModel(X, None, p, Lz=None)
            N, nb = (bm.N, bm.nb)
            self.i, self.j, self.s, self.n, self.L0 = (bm.I, bm.J, bm.sh, bm.n, bm.L0)
            self.ghost = np.zeros(nb, bool)
            self.rev = np.r_[np.arange(bm.npair) + bm.npair, np.arange(bm.npair)]
            self.B = bm.B.tocsr()
            r = np.arange(nb)
            self.Eown = sp.csr_matrix(
                (np.ones(nb), (r, 2 * self.i + self.s)), shape=(nb, 2 * N)
            )
            self.Efar = sp.csr_matrix(
                (np.ones(nb), (r, 2 * self.j + self.s)), shape=(nb, 2 * N)
            )
            self.Sd = bm.S.tocsr()
            self.Aown = bm.Aown.tocsr()
            self.Wb = sp.csr_matrix((np.ones(nb), (r, self.j)), shape=(nb, N))
            self.nmerge = nm
        else:
            X, gid = cube_grains(frames, assign, L, shifts=shifts)
            gl = GhostCube(X, gid, frames, assign, mat, L, nmls=nmls, hmls=hmls)
            N, nb = (gl.N, gl.nb)
            self.i, self.j, self.s, self.n, self.L0 = (gl.i, gl.j, gl.s, gl.n, gl.d0)
            self.ghost = gl.ghost.copy()
            real = ~self.ghost
            key = {
                (int(a), int(b)): k
                for k, (a, b) in enumerate(zip(self.i, self.j))
                if b >= 0
            }
            self.rev = np.array(
                [
                    (
                        key.get((int(self.j[k]), int(self.i[k])), -1)
                        if self.j[k] >= 0
                        else -1
                    )
                    for k in range(nb)
                ]
            )
            assert (self.rev[real] >= 0).all()
            self.B = gl.B0.tocsr()
            self.Eown = gl.Gi.tocsr()
            self.Efar = gl.Gj.tocsr()
            self.Sd = gl.G.tocsr()
            self.Aown = gl.A.T.tocsr()
            P = sp.csr_matrix(
                (np.ones(2 * N), (np.arange(2 * N), np.repeat(np.arange(N), 2))),
                shape=(2 * N, N),
            )
            self.Wb = (self.Efar @ P).tocsr()
            self.nmerge = 0
        self.X, self.gid, self.N, self.nb = (X, gid, N, nb)
        self.grain = assign_all(X)
        self.Kn = p["Kn"][self.s]
        self.nn6 = np.column_stack([self.n[:, a] * self.n[:, b] for a, b in VO])
        self.wst = np.full(nb, 0.5)
        order = np.argsort(self.i, kind="stable")
        ptr = np.searchsorted(self.i[order], np.arange(N + 1))
        for q in range(N):
            bl = order[ptr[q] : ptr[q + 1]]
            if len(bl):
                c = self.n[bl] @ self.n[bl].T
                same = self.s[bl][:, None] == self.s[bl][None, :]
                self.wst[bl] = np.where(((c < -0.99) & same).any(1), 0.5, 1.0)
        if kind == "j04":
            self.Wb = sp.csr_matrix(
                (np.ones(nb), (np.arange(nb), self.j)), shape=(nb, N)
            )
        self.lz = BIG
        bnd = np.any((X < band) | (X > L - band), axis=1)
        self.band = band
        self.fixed = np.repeat(bnd, 3)
        self.free = ~self.fixed
        self.inner = ~bnd
        self.set_weights(self.Wb)
        self.sy, self.H = (sy, H)
        self.syp = (
            np.asarray(sy, float)[self.grain] if np.ndim(sy) else np.full(N, float(sy))
        )
        self.Ci = self.star_stiffness() if ret == "scr" or plast == "cp" else None
        self.plast, self.cp = (plast, dict(cp or {}))
        if plast == "cp":
            gsl = (
                self.grain
                if kind == "single"
                else np.where(np.asarray(gid) >= 0, gid, self.grain)
            )
            Fr = np.asarray(self.cp["frames"], float)[gsl]
            s0, m0 = bcc_slip()
            sv = np.einsum("nij,aj->nai", Fr, s0)
            mv = np.einsum("nij,aj->nai", Fr, m0)
            Pt = 0.5 * (
                sv[:, :, :, None] * mv[:, :, None, :]
                + mv[:, :, :, None] * sv[:, :, None, :]
            )
            self.Pcp = np.stack([Pt[:, :, a, b] for a, b in VO], axis=2)
        d2 = ((X[:, None, :] - seeds_all[None]) ** 2).sum(-1)
        a = np.argsort(d2, 1)
        o = np.take_along_axis(d2, a[:, :2], 1)
        self.dgb = (o[:, 1] - o[:, 0]) / (
            2 * np.linalg.norm(seeds_all[a[:, 1]] - seeds_all[a[:, 0]], axis=1)
        )
        self.reset()
