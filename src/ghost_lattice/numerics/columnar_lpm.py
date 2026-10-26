"""gbp3d.py -- 3D BCC LPM (LPM-C lattice 4 force law, paper T sign) with J2 plasticity (LPM-C plmode 0, one-pass radial
return from the converged state; option ret='scr' = star-consistent radial return) on columnar polycrystals (grains rotated
about z, periodic along z with Lz = 3 a, as the PX / gb3d studies).
  'j04'   : J04 / PX construction (gb_core.build_cloud: merge pairs closer than 0.52 a, bonds by distance) = bc_core.BCCModel
  'ghost' : ghost lattice (gb3d/ghost3d.GhostLPM): complete 14-bond stars, force coupling, linear MLS (16 supports);
            far plastic state of a ghost bond: 'sa' = plastic strain interpolated at the site, 'own' = the owner's
  'single': one grain.
Material 'iso' (Zener ratio 1): the polycrystal is physically homogeneous, the exact answer under affine Dirichlet loading is
uniform.  Particle stress: sigma_i = sum_b w_b / V L0 F n n, w = 1/2 (1 if no opposite bond in the same shell).
plast='cp': rate-dependent crystal plasticity on the 12 BCC {110}<111> systems of the particle's grain (star-consistent
update, as gbp2d.constitutive_cp); cp = dict(g0, n, c, frames = crystal->specimen rotation per grain).
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from . import cubic_material as gc
from .bcc_lattice import BCCModel, params
from .grain_geometry import build_cloud, rotz, R_A
from .ghost_bcc import GhostLPM, build_grains

LZ = 3.0
VO = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
VW = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def bcc_slip():
    """12 BCC {110}<111> slip systems in the crystal frame: unit slip directions s and plane normals m (12, 3)."""
    ns = [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
    ds = [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]
    S, M = ([], [])
    for m in ns:
        for d in ds:
            if abs(np.dot(m, d)) < 1e-12:
                S.append(np.array(d, float) / np.sqrt(3))
                M.append(np.array(m, float) / np.sqrt(2))
    assert len(S) == 12
    return (np.array(S), np.array(M))


def microstructure(L, ng, seed):
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, L, (ng, 2))
    angs = rng.uniform(0, 90, ng)
    frames = [rotz(a) @ R_A for a in angs]
    shifts = [tuple(rng.uniform(0, 1, 3)) for _ in range(ng)]
    return (seeds, frames, shifts)


class Plast3D:

    def __init__(
        self,
        kind,
        L,
        seeds,
        frames,
        shifts,
        mat=None,
        band=5.0,
        sy=0.2,
        H=0.0,
        ret="lpmc",
        far="sa",
        nmls=16,
        hmls=0.9,
        coupling="force",
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
        seeds_all = seeds

        def assign_all(X):
            return np.argmin(
                np.linalg.norm(X[:, None, :2] - seeds_all[None], axis=2), 1
            )

        if kind == "single":
            frames, shifts = (frames[:1], shifts[:1])
            assign = lambda X: np.zeros(len(X), int)
        else:
            assign = assign_all
        lo, hi = ((0.0, 0.0), (L, L))
        if kind == "j04":
            X, gid, nm = build_cloud(
                frames, assign, lo, hi, LZ, rmerge=0.52, shifts=shifts
            )
            bm = BCCModel(X, None, p, Lz=LZ)
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
        else:
            X, gid = build_grains(frames, assign, lo, hi, LZ, shifts=shifts)
            gl = GhostLPM(
                X,
                gid,
                frames,
                assign,
                mat,
                LZ,
                lo,
                hi,
                nmls=nmls,
                hmls=hmls,
                crack_aware=False,
            )
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
        self.lz = LZ
        bnd = (
            (X[:, 0] < band)
            | (X[:, 0] > L - band)
            | (X[:, 1] < band)
            | (X[:, 1] > L - band)
        )
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
        if len(seeds_all) < 2:
            self.dgb = np.full(N, np.inf)
        else:
            d2 = ((X[:, None, :2] - seeds_all[None]) ** 2).sum(-1)
            a = np.argsort(d2, 1)
            o = np.take_along_axis(d2, a[:, :2], 1)
            self.dgb = (o[:, 1] - o[:, 0]) / (
                2 * np.linalg.norm(seeds_all[a[:, 1]] - seeds_all[a[:, 0]], axis=1)
            )
        self.reset()

    def set_weights(self, Wb):
        """far-end weights per bond (nb x N; real bond: e_j, ghost bond: interpolation weights) -> B, Efar, K, LU.
        force : F = 2 Kn dL + T (Eown + Efar) Sd dL, f = Aown F (owner-only, patch exact, not momentum conserving)
        energy: half-bond energy 1/2 sum_b Kn dL_b^2 + 1/2 T sum D^2, f = B^T F^h, F^h = Kn dLe + T Eown D (conserving)
        """
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
        self.Fop = (
            sp.diags(2 * self.Kn) + self.T * (self.Eown + self.Efar) @ self.Sd
        ).tocsr()
        if self.coupling == "energy":
            self.Hop = (sp.diags(self.Kn) + self.T * self.Eown @ self.Sd).tocsr()
            self.K = (self.B.T @ self.Hop @ self.B).tocsr()
        else:
            self.K = (self.Aown @ self.Fop @ self.B).tocsr()
        self.lu = spla.splu(self.K[self.free][:, self.free].tocsc())
        self.Kfc = self.K[self.free][:, self.fixed].tocsc()

    def reset(self):
        self.u = np.zeros(3 * self.N)
        self.lam = 0.0
        self.dLp = np.zeros(self.nb)
        self.alpha = np.zeros(self.N)
        self.epsp = np.zeros((self.N, 6))

    def proj(self, e6):
        """bond elongation of a tensor-component strain per particle (Voigt, shear = tensor comps) on the owner's bonds"""
        e = e6[self.i]
        return self.L0 * (
            (self.nn6[:, :3] * e[:, :3]).sum(1)
            + 2 * (self.nn6[:, 3:] * e[:, 3:]).sum(1)
        )

    def pstress(self, F):
        c = self.wst / self.V * self.L0 * F
        return np.column_stack(
            [
                np.bincount(self.i, weights=c * self.nn6[:, k], minlength=self.N)
                for k in range(6)
            ]
        )

    def star_stiffness(self):
        Ci = np.zeros((self.N, 6, 6))
        for k in range(6):
            e = np.zeros((self.N, 6))
            e[:, k] = 1.0
            ddLp = self.proj(e)
            D = self.Sd @ ddLp
            Ci[:, :, k] = self.pstress(
                2 * self.Kn * ddLp + 2 * self.T * (self.Eown @ D)
            )
        return Ci

    @staticmethod
    def dev(S):
        m = S[:, :3].sum(1) / 3
        d = S.copy()
        d[:, :3] -= m[:, None]
        return d

    @staticmethod
    def Q(a, b):
        da, db = (Plast3D.dev(a), Plast3D.dev(b))
        return 1.5 * (
            (da[:, :3] * db[:, :3]).sum(1) + 2 * (da[:, 3:] * db[:, 3:]).sum(1)
        )

    def constitutive_cp(self, dL):
        """rate-dependent crystal plasticity per particle, one pass from the converged state (as gbp2d.constitutive_cp):
        dgamma_a = c sign(tau_a) |tau_a / g|^n, tau_a = P_a : (sigma_tr - C_i sum_b dgamma_b P_b), g = g0 + h (alpha +
        sum |dgamma|), C_i = the particle's own star stiffness; damped Newton on the active particles.
        """
        dLe = dL - self.dLp
        D = self.Sd @ dLe
        S = self.pstress(2 * self.Kn * dLe + 2 * self.T * (self.Eown @ D))
        P = self.Pcp
        ns = P.shape[1]
        tau = np.einsum("nak,nk->na", P * VW, S)
        g0 = self.cp["g0"] * self.syp / np.mean(self.syp)
        n, c, h = (self.cp.get("n", 10.0), self.cp["c"], self.H)
        gconv = g0 + h * self.alpha
        act = (np.abs(tau) / gconv[:, None]).max(1) > 0.2
        dg = np.zeros((self.N, ns))
        if act.any():
            k = np.nonzero(act)[0]
            Pk = P[k]
            PC = np.einsum("nai,nij->naj", Pk * VW, self.Ci[k])
            A = np.einsum("naj,nbj->nab", PC, Pk)
            t0 = tau[k]
            gk = gconv[k]
            x = np.zeros((len(k), ns))
            Id = np.eye(ns)[None]
            for it in range(80):
                t = t0 - np.einsum("nab,nb->na", A, x)
                gg = gk + h * np.abs(x).sum(1)
                r = np.abs(t) / gg[:, None]
                R = x - c * np.sign(t) * r**n
                Jm = Id + c * n * (r ** (n - 1) / gg[:, None])[:, :, None] * (
                    A + (np.sign(t) * r)[:, :, None] * h * np.sign(x)[:, None, :]
                )
                dx = np.linalg.solve(Jm, -R[:, :, None])[:, :, 0]
                lim = 0.5 * np.maximum(np.abs(x), c)
                dx = np.clip(dx, -lim, lim)
                x = x + dx
                if np.abs(R).max() < 1e-12 * c and np.abs(dx).max() < 1e-10 * c:
                    break
            dg[k] = x
        de = np.einsum("na,nak->nk", dg, P)
        dal = np.abs(dg).sum(1)
        return (
            self.dLp + self.proj(de),
            self.alpha + dal,
            self.epsp + de,
            dal > 0.001 * c,
        )

    def constitutive(self, dL):
        if self.plast == "cp":
            return self.constitutive_cp(dL)
        dLe = dL - self.dLp
        D = self.Sd @ dLe
        S = self.pstress(2 * self.Kn * dLe + 2 * self.T * (self.Eown @ D))
        sd = self.dev(S)
        seq = np.sqrt(np.maximum(self.Q(S, S), 0))
        f = seq - (self.syp + self.H * self.alpha)
        pl = f > 0
        dlam = np.where(pl, f / (3 * self.Gsh + self.H), 0.0)
        nt = 1.5 * sd / np.maximum(seq, 1e-300)[:, None]
        if self.ret == "scr" and pl.any():
            k = np.nonzero(pl)[0]
            d = np.einsum("nij,nj->ni", self.Ci[k], nt[k])
            Y = self.syp[k] + self.H * self.alpha[k]
            A = self.Q(d, d) - self.H**2
            B = self.Q(S[k], d) + Y * self.H
            C = self.Q(S[k], S[k]) - Y**2
            disc = B * B - A * C
            ok = (A > 0) & (disc >= 0)
            dlam[k] = np.maximum(
                np.where(
                    ok,
                    C / (B + np.sqrt(np.maximum(disc, 0))),
                    B / np.maximum(A, 1e-300),
                ),
                0,
            )
        de = dlam[:, None] * nt
        return (self.dLp + self.proj(de), self.alpha + dlam, self.epsp + de, pl)

    def enable_gfc(self):
        """ghost-force correction for the energy coupling (see gbp2d.Poly2D.enable_gfc): R_i (3 x 3) = residual force of
        particle i per unit in-plane strain [exx, eyy, exy] (the strains an affine field carries with periodic z);
        f -= R_i eps_i, eps_i = least-squares in-plane strain of particle i from its own bonds (elastic part).
        """
        assert self.coupling == "energy"
        N, nb = (self.N, self.nb)
        v6 = self.L0[:, None] * np.column_stack([self.nn6[:, :3], 2 * self.nn6[:, 3:]])
        R = np.zeros((N, 3, 6))
        for k in range(6):
            dL = v6[:, k]
            f = self.B.T @ (self.Kn * dL + self.T * (self.Eown @ (self.Sd @ dL)))
            R[:, :, k] = f.reshape(-1, 3)
        R[self.fixed[0::3]] = 0.0
        M = np.zeros((N, 6, 6))
        np.add.at(M, self.i, v6[:, :, None] * v6[:, None, :])
        Minv = np.linalg.pinv(M)
        coef = np.einsum("bij,bj->bi", Minv[self.i], v6)
        rows = (6 * self.i[:, None] + np.arange(6)[None, :]).ravel()
        cols = np.repeat(np.arange(nb), 6)
        self.Gstrain = sp.csr_matrix((coef.ravel(), (rows, cols)), shape=(6 * N, nb))
        self.gfc = R

    def forces(self, dL, dLp, epsp):
        dLe = dL - dLp
        D = self.Sd @ dLe
        if self.coupling == "energy":
            Fh = self.Kn * dLe + self.T * (self.Eown @ D)
            f = self.B.T @ Fh
            if getattr(self, "gfc", None) is not None:
                eps = (self.Gstrain @ dLe).reshape(-1, 6)
                f = f - np.einsum("nck,nk->nc", self.gfc, eps).ravel()
            return (f, 2 * Fh)
        far = np.empty(self.nb)
        real = ~self.ghost
        far[real] = dLe[self.rev[real]]
        if self.ghost.any():
            g = np.nonzero(self.ghost)[0]
            if self.far == "sa":
                e = (self.Wb @ epsp)[g]
                far[g] = dL[g] - self.L0[g] * (
                    (self.nn6[g, :3] * e[:, :3]).sum(1)
                    + 2 * (self.nn6[g, 3:] * e[:, 3:]).sum(1)
                )
            else:
                far[g] = dLe[g]
        F = self.Kn * (dLe + far) + self.T * (self.Eown @ D + self.Efar @ D)
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
                x = x + g - (dX + dF) @ np.linalg.lstsq(dF, g, rcond=None)[0]
            else:
                x = x + g
        else:
            print(
                "WARNING not converged lam %.4g |R|/ref %.2e" % (lam, nR / nref),
                flush=True,
            )
        u[free] = x
        self.u, self.lam = (u, lam)
        self.dLp, self.alpha, self.epsp, self.F = (dLp, al, ep, F)
        return it
