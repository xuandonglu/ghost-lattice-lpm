"""gbp2d.py -- 2D LPM (LPM-C replica constants: hexagonal lattice, two shells, plane stress, spacing 1) with J2 plasticity
(LPM-C plmode 0, one-pass radial return from the converged state, as lpmc_skin_check/pl_lpm.py) on polycrystals whose grain
boundaries are built either
  'j04'   : the J04 / PX way -- every grain's lattice rotated (and shifted), clipped to its Voronoi cell, overlapping
            particles merged (threshold 1.2 R = 0.6), bonds by distance (shell 1 < 1.366, shell 2 < 1.866), or
  'ghost' : the ghost lattice -- every particle keeps the complete 12-bond star of its own grain; a lattice site that lies in
            another grain is a ghost site whose displacement (and multibody sum D) is interpolated from that grain's
            particles by linear-exact MLS; forces are particle-owned (force-based coupling: patch-test exact, as the 3D
            drivers' force mode).  Plastic state of the far end of a ghost bond: 'sa' = the plastic strain tensor
            interpolated at the site (state-aware, the surface rule C of PL_PLASTIC_SKIN_REPORT), 'own' = the owner's.
  'single': one grain (no boundary) on the same domain.
The material is isotropic (the hexagonal lattice is elastically isotropic, J2 is isotropic), so a polycrystal is PHYSICALLY
HOMOGENEOUS: with the affine Dirichlet band the exact answer is the uniform response of the single crystal, and every
deviation is a grain-boundary artefact.
Bond law (plmode 0): F_b = Kn (dLe_b + dLe_far) + T (D_own + D_far), trial F = 2 Kn dLe_b + 2 T D_own,
particle stress sigma_i = sum_b w_b / V L0 F n n (w = 1/2, or 1 if the opposite bond is missing), dLp_b = L0 n.eps_p.n.
"""

import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from .plane_material import mat2d, Cmat

MAT = mat2d(1.0, 0.3)
KN, TT, VV = (MAT["Kn1"], MAT["T"], MAT["V"])
GSH = 0.5 / (1 + 0.3)
SQ3 = math.sqrt(3.0)
R01, R12 = ((1 + SQ3) / 2, (SQ3 + 2) / 2)
STAR = [(1.0, math.radians(60 * k)) for k in range(6)] + [
    (SQ3, math.radians(30 + 60 * k)) for k in range(6)
]


def hex_points(theta, shift, lo, hi):
    L = 1.5 * max(hi[0] - lo[0], hi[1] - lo[1]) + 10
    n = int(L) + 4
    ii, jj = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    P = np.stack(
        [(ii + 0.5 * (jj % 2)).ravel(), (SQ3 / 2 * jj).ravel()], 1
    ) + np.asarray(shift)
    c, s = (math.cos(theta), math.sin(theta))
    P = P @ np.array([[c, s], [-s, c]])
    m = (P[:, 0] >= lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] < hi[1])
    return P[m]


def microstructure(W, ng, seed):
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, W, (ng, 2))
    theta = rng.uniform(0, math.pi / 3, ng)
    shift = rng.uniform(0, 1, (ng, 2))
    return (seeds, theta, shift)


def assign(P, seeds):
    return np.argmin(((P[:, None, :] - seeds[None, :, :]) ** 2).sum(-1), 1)


def grain_particles(W, seeds, theta, shift):
    X, gid = ([], [])
    for g in range(len(seeds)):
        P = hex_points(theta[g], shift[g], (0, 0), (W, W))
        k = assign(P, seeds) == g
        X.append(P[k])
        gid.append(np.full(k.sum(), g))
    return (np.vstack(X), np.concatenate(gid))


def merge_overlaps(X, gid, thr):
    while True:
        pr = cKDTree(X).query_pairs(thr, output_type="ndarray")
        if not len(pr):
            return (X, gid)
        n = len(X)
        G = sp.csr_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
        nc, lab = sp.csgraph.connected_components(G, directed=False)
        cnt = np.bincount(lab, minlength=nc)
        C = np.zeros((nc, 2))
        np.add.at(C, lab, X)
        g2 = np.full(nc, -1)
        g2[lab] = gid
        X, gid = (C / cnt[:, None], g2)


def tri_weights(Y):
    """barycentric weights of the origin in the Delaunay triangle of the support offsets Y (k, 2) that contains it (or, if
    outside the hull, the triangle with the least negative barycentric coordinate): linear exact, interpolatory at nodes
    """
    from scipy.spatial import Delaunay

    tri = Delaunay(Y)
    s = int(tri.find_simplex(np.zeros((1, 2)))[0])
    T = tri.transform
    if s < 0:
        b = np.einsum("sij,sj->si", T[:, :2, :], -T[:, 2, :])
        bb = np.column_stack([b, 1 - b.sum(1)])
        mn = np.where(np.isfinite(bb).all(1), bb.min(1), -np.inf)
        s = int(np.argmax(mn))
    b = T[s, :2, :] @ -T[s, 2, :]
    w = np.zeros(len(Y))
    w[tri.simplices[s]] = np.r_[b, 1 - b.sum()]
    return w


def mls_weights_q(Y, h=1.2):
    """quadratic-exact MLS weights of the origin (basis 1, x, y, x^2, xy, y^2); linear MLS if singular"""
    d = np.linalg.norm(Y, axis=1)
    w = np.exp(-((d / h) ** 2))
    P = np.column_stack(
        [np.ones(len(Y)), Y, Y[:, 0] ** 2, Y[:, 0] * Y[:, 1], Y[:, 1] ** 2]
    )
    A = P.T @ (w[:, None] * P)
    if len(Y) < 6 or np.linalg.cond(A) > 10000000000.0:
        return mls_weights(Y, 1.0)
    e = np.zeros(6)
    e[0] = 1.0
    return w * (P @ np.linalg.solve(A, e))


def mls_weights(Y, h=1.0):
    """linear-exact MLS weights of the origin from support offsets Y (k, 2)"""
    d = np.linalg.norm(Y, axis=1)
    w = np.exp(-((d / h) ** 2))
    P = np.column_stack([np.ones(len(Y)), Y])
    A = P.T @ (w[:, None] * P)
    return w * (P @ np.linalg.solve(A, np.array([1.0, 0.0, 0.0])))


class Poly2D:

    def __init__(
        self,
        W,
        seeds,
        theta,
        shift,
        kind="ghost",
        far_state="sa",
        nsup=12,
        hmls=1.0,
        sy=0.0015,
        H=0.0,
        band=2.1,
        ret="lpmc",
        coupling="force",
        interp="mls",
        hole=None,
        plast="j2",
        cp=None,
    ):
        self.W, self.kind, self.far_state, self.coupling = (
            W,
            kind,
            far_state,
            coupling,
        )
        seeds_all = seeds
        theta_all = np.asarray(theta, float)
        if kind == "single":
            seeds, theta, shift = (seeds[:1], theta[:1], shift[:1])
        X, gid = grain_particles(W, seeds, theta, shift)
        self.hole = hole
        if hole is not None:
            keep = np.hypot(X[:, 0] - hole[0], X[:, 1] - hole[1]) >= hole[2]
            X, gid = (X[keep], gid[keep])
        if kind == "j04":
            X, gid = merge_overlaps(X, gid, 0.6)
        self.X, self.gid, self.N = (X, gid, len(X))
        self.seeds = seeds_all
        self.grain = assign(X, seeds_all)
        I, J, Gi, S, NV, L0 = ([], [], [], [], [], [])
        gsup, gw = ([], [])
        if kind == "j04":
            pr = cKDTree(X).query_pairs(R12, output_type="ndarray")
            i = np.r_[pr[:, 0], pr[:, 1]]
            j = np.r_[pr[:, 1], pr[:, 0]]
            d = X[i] - X[j]
            L = np.linalg.norm(d, axis=1)
            I, J, L0 = (i, j, L)
            NV = d / L[:, None]
            S = (L >= R01).astype(int)
            Gi = np.full(len(I), -1)
        else:
            trees = {
                g: (cKDTree(X[gid == g]), np.nonzero(gid == g)[0])
                for g in np.unique(gid)
            }
            tree_all = cKDTree(X)
            for p in range(self.N):
                g = gid[p]
                for r, a in STAR:
                    v = r * np.array([math.cos(a + theta[g]), math.sin(a + theta[g])])
                    site = X[p] + v
                    inside = (
                        -1e-09 <= site[0] < W + 1e-09 and -1e-09 <= site[1] < W + 1e-09
                    )
                    if (
                        hole is not None
                        and math.hypot(site[0] - hole[0], site[1] - hole[1]) < hole[2]
                    ):
                        continue
                    h = (
                        assign(site[None], seeds)[0]
                        if kind == "ghost" and inside
                        else g
                    )
                    sh = int(r > 1.5)
                    if h == g:
                        dd, q = trees[g][0].query(site)
                        if dd > 1e-06:
                            continue
                        I.append(p)
                        J.append(trees[g][1][q])
                        Gi.append(-1)
                    else:
                        dd, q = trees[h][0].query(site, k=min(nsup, len(trees[h][1])))
                        sup = trees[h][1][np.atleast_1d(q)]
                        Yh = X[sup] - site
                        if (
                            len(sup) < 3
                            or np.linalg.matrix_rank(
                                np.column_stack([np.ones(len(sup)), Yh]), tol=1e-06
                            )
                            < 3
                        ):
                            dd, q = tree_all.query(site, k=3 * nsup)
                            q = np.atleast_1d(q)
                            q = q[gid[q] != g][:nsup]
                            sup = q
                            Yh = X[sup] - site
                        w = (
                            tri_weights(Yh)
                            if interp == "tri"
                            else (
                                mls_weights_q(Yh, 1.2 * hmls)
                                if interp == "quad"
                                else mls_weights(Yh, hmls)
                            )
                        )
                        sup, w = (sup[np.abs(w) > 1e-14], w[np.abs(w) > 1e-14])
                        I.append(p)
                        J.append(-1)
                        Gi.append(len(gsup))
                        gsup.append(sup)
                        gw.append(w)
                    S.append(sh)
                    NV.append(-v / r)
                    L0.append(r)
            I, J, Gi, S, NV, L0 = map(np.asarray, (I, J, Gi, S, NV, L0))
        self.I, self.J, self.Gi, self.S, self.n, self.L0 = (
            I,
            J,
            Gi,
            S,
            np.asarray(NV),
            L0,
        )
        self.nb = nb = len(I)
        self.ghost = Gi >= 0
        self.ng = len(gsup)
        N = self.N
        if self.ng:
            rows = np.repeat(np.arange(self.ng), [len(s) for s in gsup])
            self.Wg = sp.csr_matrix(
                (np.concatenate(gw), (rows, np.concatenate(gsup))), shape=(self.ng, N)
            )
        else:
            self.Wg = sp.csr_matrix((0, N))
        rb = np.arange(nb)
        real = ~self.ghost
        self._build_B()
        self.Aown = sp.csr_matrix(
            (
                np.r_[self.n[:, 0], self.n[:, 1]],
                (np.r_[2 * I, 2 * I + 1], np.r_[rb, rb]),
            ),
            shape=(2 * N, nb),
        )
        key = {
            (int(a), int(c)): k
            for k, (a, c) in enumerate(zip(I.tolist(), J.tolist()))
            if c >= 0
        }
        self.rev = np.array(
            [
                key.get((int(J[k]), int(I[k])), -1) if J[k] >= 0 else -1
                for k in range(nb)
            ]
        )
        assert (self.rev[real] >= 0).all(), "missing reverse bond"
        self.wst = np.full(nb, 1.0)
        order = np.argsort(I, kind="stable")
        ptr = np.searchsorted(I[order], np.arange(N + 1))
        for p in range(N):
            bl = order[ptr[p] : ptr[p + 1]]
            if not len(bl):
                continue
            c = self.n[bl] @ self.n[bl].T
            same = S[bl][:, None] == S[bl][None, :]
            has = ((c < -0.99) & same).any(1)
            self.wst[bl] = np.where(has, 0.5, 1.0)
        self.nnL = np.column_stack(
            [self.n[:, 0] ** 2, self.n[:, 1] ** 2, self.n[:, 0] * self.n[:, 1]]
        )
        self.band = band
        bnd = (
            (X[:, 0] < band)
            | (X[:, 0] > W - band)
            | (X[:, 1] < band)
            | (X[:, 1] > W - band)
        )
        self.fixed = np.repeat(bnd, 2)
        self.free = ~self.fixed
        self.inner = ~bnd
        self._build_K()
        self.sy = sy
        self.H = H
        self.syp = (
            np.asarray(sy, float)[self.grain] if np.ndim(sy) else np.full(N, float(sy))
        )
        self.ret = ret
        self.Ci = self.star_stiffness() if ret == "scr" or plast == "cp" else None
        self.plast, self.cp = (plast, dict(cp or {}))
        if plast == "cp":
            gsl = (
                self.grain
                if kind == "single"
                else np.where(self.gid >= 0, self.gid, self.grain)
            )
            if "P" in self.cp:
                self.Pcp = np.asarray(self.cp["P"], float)[gsl]
            else:
                th_g = (
                    np.asarray(self.cp["theta"], float)
                    if "theta" in self.cp
                    else theta_all
                )
                th = th_g[gsl][:, None] + np.radians([0.0, 60.0, 120.0])[None, :]
                sx, sy = (np.cos(th), np.sin(th))
                mx, my = (-sy, sx)
                self.Pcp = np.stack(
                    [sx * mx, sy * my, 0.5 * (sx * my + sy * mx)], axis=2
                )
        self.reset()
        if len(seeds_all) < 2:
            self.dgb = np.full(N, np.inf)
        else:
            d2 = ((X[:, None, :] - seeds_all[None, :, :]) ** 2).sum(-1)
            o = np.sort(d2, 1)
            a = np.argsort(d2, 1)
            s1, s2 = (seeds_all[a[:, 0]], seeds_all[a[:, 1]])
            self.dgb = (o[:, 1] - o[:, 0]) / (2 * np.linalg.norm(s2 - s1, axis=1))

    def _build_B(self):
        """dL = B u (ghost rows through the interpolation weights Wg), multibody gathers Sd / Eown / Efar"""
        I, J, Gi, S, nb, N = (self.I, self.J, self.Gi, self.S, self.nb, self.N)
        rb = np.arange(nb)
        real = ~self.ghost
        rows, cols, vals = ([rb, rb], [2 * I, 2 * I + 1], [self.n[:, 0], self.n[:, 1]])
        rows += [rb[real], rb[real]]
        cols += [2 * J[real], 2 * J[real] + 1]
        vals += [-self.n[real, 0], -self.n[real, 1]]
        ef_r, ef_c, ef_v = ([rb[real]], [2 * J[real] + S[real]], [np.ones(real.sum())])
        gb = np.nonzero(self.ghost)[0]
        if len(gb):
            Wc = self.Wg[Gi[gb]].tocoo()
            b = gb[Wc.row]
            rows += [b, b]
            cols += [2 * Wc.col, 2 * Wc.col + 1]
            vals += [-Wc.data * self.n[b, 0], -Wc.data * self.n[b, 1]]
            ef_r.append(b)
            ef_c.append(2 * Wc.col + S[b])
            ef_v.append(Wc.data)
        self.B = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nb, 2 * N),
        )
        self.Sd = sp.csr_matrix((np.ones(nb), (2 * I + S, rb)), shape=(2 * N, nb))
        self.Eown = sp.csr_matrix((np.ones(nb), (rb, 2 * I + S)), shape=(nb, 2 * N))
        self.Efar = sp.csr_matrix(
            (np.concatenate(ef_v), (np.concatenate(ef_r), np.concatenate(ef_c))),
            shape=(nb, 2 * N),
        )

    def _build_K(self):
        """elastic stiffness of the chosen coupling and its LU on the free dofs.
        force : F = 2 Kn dL + T (Eown + Efar) Sd dL, f = Aown F (owner-only, patch exact)
        energy: half-bond energy E = 1/2 sum_b Kn dL_b^2 + 1/2 T sum_(i,s) D_is^2 (all directed bonds incl. ghost
                half-bonds), f = B^T F^h, F^h_b = Kn dLe_b + T D_own: reactions of ghost half-bonds go to the interpolation
                supports (momentum conserving, symmetric; each side's half-bonds carry that side's own plastic state)
        """
        nb = self.nb
        self.Fop = (
            2 * KN * sp.identity(nb) + TT * (self.Eown + self.Efar) @ self.Sd
        ).tocsr()
        if self.coupling == "energy":
            self.Hop = (KN * sp.identity(nb) + TT * self.Eown @ self.Sd).tocsr()
            self.K = (self.B.T @ self.Hop @ self.B).tocsr()
        else:
            self.K = (self.Aown @ self.Fop @ self.B).tocsr()
        self.lu = spla.splu(self.K[self.free][:, self.free].tocsc())
        self.Kfc = self.K[self.free][:, self.fixed].tocsc()

    def set_weights(self, Wg):
        """replace the ghost-site interpolation weights (same or new sparsity) and rebuild the operators"""
        self.Wg = sp.csr_matrix(Wg)
        self._build_B()
        self._build_K()

    def reset(self):
        self.u = np.zeros(2 * self.N)
        self.lam = 0.0
        self.dLp = np.zeros(self.nb)
        self.alpha = np.zeros(self.N)
        self.epsp = np.zeros((self.N, 3))
        self.gG = np.zeros(self.N)
        self.chi = (
            np.zeros((self.N, self.Pcp.shape[1]))
            if getattr(self, "plast", "j2") == "cp" and hasattr(self, "Pcp")
            else None
        )

    def pstress(self, F):
        c = self.wst / VV * self.L0 * F
        return np.column_stack(
            [
                np.bincount(self.I, weights=c * self.nnL[:, k], minlength=self.N)
                for k in range(3)
            ]
        )

    def star_stiffness(self):
        """C_i (N, 3, 3): drop of the particle's trial stress [xx, yy, xy] per unit plastic strain [xx, yy, xy] (tensor
        components) projected on its own bonds -- the stiffness the return mapping actually meets.
        """
        Ci = np.zeros((self.N, 3, 3))
        for k, e in enumerate(np.eye(3)):
            ddLp = self.L0 * (
                e[0] * self.nnL[:, 0]
                + e[1] * self.nnL[:, 1]
                + 2 * e[2] * self.nnL[:, 2]
            )
            D = self.Sd @ ddLp
            Ci[:, :, k] = self.pstress(2 * KN * ddLp + 2 * TT * (self.Eown @ D))
        return Ci

    def constitutive_cp(self, dL):
        """planar rate-dependent crystal plasticity per particle (one pass from the converged state, as plmode 0):
        dgamma_a = c sign(tau_a) |tau_a / g|^n,  tau_a = P_a : (sigma_tr - C_i sum_b dgamma_b P_b),
        g = g0 + h (alpha + sum |dgamma|);  C_i = the particle's own star stiffness (star-consistent, cf. SCR).
        cp: g0 (CRSS, absolute), n, c (= gamma_dot_0 * dt per load step); h = self.H; alpha = accumulated slip.
        """
        dLe = dL - self.dLp
        D = self.Sd @ dLe
        S = self.pstress(2 * KN * dLe + 2 * TT * (self.Eown @ D))
        P = self.Pcp
        ns = P.shape[1]
        tau = (
            P[:, :, 0] * S[:, None, 0]
            + P[:, :, 1] * S[:, None, 1]
            + 2 * P[:, :, 2] * S[:, None, 2]
        )
        if getattr(self, "chi", None) is not None:
            tau = tau - self.chi
        g0 = (
            self.cp.get("g0", 0.5 * float(np.mean(self.syp)))
            * self.syp
            / np.mean(self.syp)
        )
        n, c, h = (self.cp.get("n", 10.0), self.cp["c"], self.H)
        gconv = g0 + h * self.alpha + self.gG
        act = (np.abs(tau) / gconv[:, None]).max(1) > 0.2
        dg = np.zeros((self.N, ns))
        if act.any():
            k = np.nonzero(act)[0]
            Pk = P[k]
            PC = np.einsum("nai,nij->naj", Pk * np.array([1.0, 1.0, 2.0]), self.Ci[k])
            A = np.einsum("naj,nbj->nab", PC, Pk)
            t0 = tau[k]
            gk = gconv[k]
            x = np.zeros((len(k), ns))
            for it in range(60):
                t = t0 - np.einsum("nab,nb->na", A, x)
                gg = gk + h * np.abs(x).sum(1)
                r = np.abs(t) / gg[:, None]
                R = x - c * np.sign(t) * r**n
                Jm = np.eye(ns)[None] + c * n * (r ** (n - 1) / gg[:, None])[
                    :, :, None
                ] * (A + (np.sign(t) * r)[:, :, None] * h * np.sign(x)[:, None, :])
                dx = np.linalg.solve(Jm, -R[:, :, None])[:, :, 0]
                lim = 0.5 * np.maximum(np.abs(x), c)
                dx = np.clip(dx, -lim, lim)
                x = x + dx
                if np.abs(R).max() < 1e-12 * max(c, 1e-30) and np.abs(
                    dx
                ).max() < 1e-10 * max(c, 1e-30):
                    break
            dg[k] = x
        de = np.einsum("na,nai->ni", dg, P)
        ddLp = self.L0 * (
            de[self.I, 0] * self.nnL[:, 0]
            + de[self.I, 1] * self.nnL[:, 1]
            + 2 * de[self.I, 2] * self.nnL[:, 2]
        )
        dal = np.abs(dg).sum(1)
        self._last_dg = dg
        return (self.dLp + ddLp, self.alpha + dal, self.epsp + de, dal > 0.001 * c)

    def constitutive(self, dL):
        if getattr(self, "plast", "j2") == "cp":
            return self.constitutive_cp(dL)
        dLe = dL - self.dLp
        D = self.Sd @ dLe
        Ftr = 2 * KN * dLe + 2 * TT * (self.Eown @ D)
        Sg = self.pstress(Ftr)
        m = (Sg[:, 0] + Sg[:, 1]) / 3
        sx, sy_, sz, sxy = (Sg[:, 0] - m, Sg[:, 1] - m, -m, Sg[:, 2])
        seq = np.sqrt(1.5 * (sx**2 + sy_**2 + sz**2 + 2 * sxy**2))
        f = seq - (self.syp + self.H * self.alpha)
        pl = f > 0
        dlam = np.where(pl, f / (3 * GSH + self.H), 0.0)
        if self.ret == "scr" and pl.any():
            k = np.nonzero(pl)[0]
            nt = 1.5 / seq[k, None] * np.column_stack([sx[k], sy_[k], sxy[k]])
            d = np.einsum("nij,nj->ni", self.Ci[k], nt)
            s0 = Sg[k]

            def Q(a, b):
                return (
                    a[:, 0] * b[:, 0]
                    - 0.5 * (a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0])
                    + a[:, 1] * b[:, 1]
                    + 3 * a[:, 2] * b[:, 2]
                )

            Y = self.syp[k] + self.H * self.alpha[k]
            A = Q(d, d) - self.H**2
            B = Q(s0, d) + Y * self.H
            Cc = Q(s0, s0) - Y**2
            disc = B * B - A * Cc
            ok = (A > 0) & (disc >= 0)
            dl = np.where(
                ok, Cc / (B + np.sqrt(np.maximum(disc, 0))), B / np.maximum(A, 1e-300)
            )
            dlam[k] = np.maximum(dl, 0.0)
        c = np.where(seq > 1e-300, 1.5 * dlam / np.maximum(seq, 1e-300), 0.0)
        de = np.column_stack([c * sx, c * sy_, c * sxy])
        ddLp = self.L0 * (
            de[self.I, 0] * self.nnL[:, 0]
            + de[self.I, 1] * self.nnL[:, 1]
            + 2 * de[self.I, 2] * self.nnL[:, 2]
        )
        return (self.dLp + ddLp, self.alpha + dlam, self.epsp + de, pl)

    def enable_gfc(self):
        """ghost-force correction for the energy coupling (cf. the dead-load ghost-force correction of the QC method):
        R_i (2 x 3) = residual force of particle i per unit uniform strain [exx, eyy, exy] (tensor components); in
        forces() the term R_i eps_e,i is subtracted, eps_e,i = least-squares elastic strain of particle i from its own
        bonds.  Uniform elastic strain -> exact patch test; the non-conservative part is proportional to the residual,
        which the optimised weights already make small."""
        assert self.coupling == "energy"
        N, nb = (self.N, self.nb)
        R = np.zeros((N, 2, 3))
        for k, E in enumerate(
            (
                np.array([[1.0, 0], [0, 0]]),
                np.array([[0, 0], [0, 1.0]]),
                np.array([[0, 1.0], [1.0, 0]]),
            )
        ):
            ua = (self.X @ E.T).ravel()
            dL = self.B @ ua
            f = self.B.T @ (KN * dL + TT * (self.Eown @ (self.Sd @ dL)))
            R[:, :, k] = f.reshape(-1, 2)
        R[self.fixed[0::2]] = 0.0
        if getattr(self, "hole", None) is not None:
            hx, hy, hr = self.hole
            R[np.hypot(self.X[:, 0] - hx, self.X[:, 1] - hy) <= hr + 3.5] = 0.0
        v = self.L0[:, None] * np.column_stack(
            [self.nnL[:, 0], self.nnL[:, 1], 2 * self.nnL[:, 2]]
        )
        M = np.zeros((N, 3, 3))
        np.add.at(M, self.I, v[:, :, None] * v[:, None, :])
        Minv = np.linalg.pinv(M)
        coef = np.einsum("bij,bj->bi", Minv[self.I], v)
        rows = (3 * self.I[:, None] + np.arange(3)[None, :]).ravel()
        cols = np.repeat(np.arange(nb), 3)
        self.Gstrain = sp.csr_matrix((coef.ravel(), (rows, cols)), shape=(3 * N, nb))
        self.gfc = R

    def forces(self, dL, dLp, epsp):
        dLe = dL - dLp
        D = self.Sd @ dLe
        if self.coupling == "energy":
            Fh = KN * dLe + TT * (self.Eown @ D)
            f = self.B.T @ Fh
            if getattr(self, "gfc", None) is not None:
                eps = (self.Gstrain @ dLe).reshape(-1, 3)
                f = f - np.einsum("nck,nk->nc", self.gfc, eps).ravel()
            return (f, 2 * Fh)
        far = np.empty(self.nb)
        real = ~self.ghost
        far[real] = dLe[self.rev[real]]
        if self.ghost.any():
            gb = np.nonzero(self.ghost)[0]
            if self.far_state == "sa":
                ep = self.Wg @ epsp
                e = ep[self.Gi[gb]]
                dLpf = self.L0[gb] * (
                    e[:, 0] * self.nnL[gb, 0]
                    + e[:, 1] * self.nnL[gb, 1]
                    + 2 * e[:, 2] * self.nnL[gb, 2]
                )
                far[gb] = dL[gb] - dLpf
            else:
                far[gb] = dLe[gb]
        F = KN * (dLe + far) + TT * (self.Eown @ D + self.Efar @ D)
        return (self.Aown @ F, F)

    def step(self, E, lam, tol=1e-09, maxit=600, mAA=8):
        free, fixed = (self.free, self.fixed)
        u = self.u.copy()
        ub = (self.X @ np.asarray(E).T).ravel()
        dfix = (lam - self.lam) * ub[fixed]
        u[fixed] += dfix
        u[free] += self.lu.solve(-(self.Kfc @ dfix))
        x = u[free].copy()
        hdx, hdf = ([], [])
        xp = gp = None
        best = np.inf
        for it in range(maxit):
            u[free] = x
            dL = self.B @ u
            dLp, al, ep, pl = self.constitutive(dL)
            f, F = self.forces(dL, dLp, ep)
            Rr = f[free]
            nR = np.linalg.norm(Rr)
            nref = max(np.linalg.norm(f[fixed]), 1e-30)
            if nR < tol * nref:
                break
            g = -self.lu.solve(Rr)
            if xp is not None:
                hdx.append(x - xp)
                hdf.append(g - gp)
                if len(hdx) > mAA:
                    hdx.pop(0)
                    hdf.pop(0)
            if nR > 50 * best:
                hdx, hdf = ([], [])
            best = min(best, nR)
            xp, gp = (x.copy(), g.copy())
            if hdx:
                dX = np.column_stack(hdx)
                dF = np.column_stack(hdf)
                gam = np.linalg.lstsq(dF, g, rcond=None)[0]
                x = x + g - (dX + dF) @ gam
            else:
                x = x + g
        else:
            print(
                "WARNING not converged lam %.4g |R|/ref %.2e" % (lam, nR / nref),
                flush=True,
            )
        u[free] = x
        self.u, self.lam = (u, lam)
        self.dLp, self.alpha, self.epsp = (dLp, al, ep)
        self.F = F
        return it

    def out_stress(self):
        return self.pstress(self.F)
