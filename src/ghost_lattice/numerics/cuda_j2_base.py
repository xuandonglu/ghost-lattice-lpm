"""Original planar single-lattice SCR J2 equations, resident FP64 CUDA solve.

The tangent differentiates the actual closed-form return, including its
existing fallback branch. This changes the numerical solve, not its solution.
"""

import time, types
import numpy as np
import scipy.sparse as sp
import torch as t
from .cuda_material import gpu, cpu
from .cuda_linear import sparse, norm, cg, gmres, AMG
from .plane_lpm import KN, TT, VV, GSH


class Solver:

    def __init__(self, m):
        assert (
            m.kind == "single"
            and m.plast == "j2"
            and (m.ret == "scr")
            and (not m.ghost.any())
            and (m.X.shape[1] == 2)
        )
        assert getattr(m, "gfc", None) is None
        self.m = m
        self.B = sparse(m.B)
        self.Bt = sparse(m.B.T)
        self.Sd = sparse(m.Sd)
        self.I = t.as_tensor(m.I.astype(np.int64), device="cuda")
        self.L = gpu(m.L0)
        self.nn = gpu(m.nnL)
        self.nneng = self.nn * gpu([1.0, 1.0, 2.0])
        self.owner = t.as_tensor((2 * m.I + m.S).astype(np.int64), device="cuda")
        rows = (3 * m.I[:, None] + np.arange(3)[None]).ravel()
        cols = np.repeat(np.arange(m.nb), 3)
        data = (m.wst / VV * m.L0)[:, None] * m.nnL
        self.Ps = sparse(
            sp.csr_matrix((data.ravel(), (rows, cols)), shape=(3 * m.N, m.nb))
        )
        self.free = t.as_tensor(np.flatnonzero(m.free), device="cuda")
        self.fixed = t.as_tensor(np.flatnonzero(m.fixed), device="cuda")
        self.C = gpu(m.Ci)
        self.sy = gpu(m.syp)
        self.H = m.H
        self.Q = gpu([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 3.0]])
        self.D = self.Q / gpu([1.0, 1.0, 2.0])[:, None]
        self.u = gpu(m.u)
        self.alpha = gpu(m.alpha)
        self.ep = gpu(m.epsp)
        self.dp = gpu(m.dLp)
        if hasattr(m.lu, "ml"):
            self.M = AMG(m.lu.ml, 2)
            self.K = self.M.levels[0]["A"]
        else:
            self.K = sparse(m.K[m.free][:, m.free])
            self.M = lambda x: gpu(m.lu.solve(cpu(x)))
        self._lam = m.lam

    def projection(self, ep):
        return self.L * (ep[self.I] * self.nneng).sum(1)

    def half(self, x):
        return KN * x + TT * (self.Sd @ x)[self.owner]

    def stress(self, x):
        return (self.Ps @ (2 * self.half(x))).reshape(self.m.N, 3)

    def material(self, dL, tangent=True):
        s = self.stress(dL - self.dp)
        q = t.sqrt(t.clamp(t.einsum("ni,ij,nj->n", s, self.Q, s), min=0.0))
        Y = self.sy + self.H * self.alpha
        active = q > Y
        n = s @ self.D.T / t.clamp(q[:, None], min=1e-300)
        d = t.einsum("nij,nj->ni", self.C, n)
        A = t.einsum("ni,ij,nj->n", d, self.Q, d) - self.H**2
        B = t.einsum("ni,ij,nj->n", s, self.Q, d) + Y * self.H
        Cc = q * q - Y * Y
        disc = B * B - A * Cc
        root = t.sqrt(t.clamp(disc, min=0.0))
        ok = (A > 0) & (disc >= 0)
        den = B + root
        safe_den = t.where(t.abs(den) > 1e-300, den, t.ones_like(den))
        dl = t.where(ok, Cc / safe_den, B / t.clamp(A, min=1e-300))
        dl = t.where(active, t.clamp(dl, min=0.0), t.zeros_like(dl))
        de = dl[:, None] * n
        ep = self.ep + de
        alpha = self.alpha + dl
        dp = self.dp + self.projection(de)
        derivative = None
        if tangent:
            qsafe = t.clamp(q, min=1e-300)
            nq = n * gpu([1.0, 1.0, 2.0])
            dn = (
                self.D[None] / qsafe[:, None, None]
                - n[:, :, None] * nq[:, None, :] / qsafe[:, None, None]
            )
            dd = t.einsum("nij,njk->nik", self.C, dn)
            Qd = d @ self.Q
            Qs = s @ self.Q
            dA = 2 * t.einsum("ni,nij->nj", Qd, dd)
            dB = Qd + t.einsum("ni,nij->nj", Qs, dd)
            dCc = 2 * Qs
            ddisc = 2 * B[:, None] * dB - Cc[:, None] * dA - A[:, None] * dCc
            dden = dB + ddisc / (2 * t.clamp(root[:, None], min=1e-100))
            ddl_good = (dCc * safe_den[:, None] - Cc[:, None] * dden) / safe_den[
                :, None
            ] ** 2
            aden = t.clamp(A, min=1e-100)
            daden = t.where((A > 1e-100)[:, None], dA, t.zeros_like(dA))
            ddl_bad = (dB * aden[:, None] - B[:, None] * daden) / aden[:, None] ** 2
            ddl = t.where(ok[:, None], ddl_good, ddl_bad)
            derivative = n[:, :, None] * ddl[:, None, :] + dl[:, None, None] * dn
            derivative = t.where(
                (active & (dl > 0))[:, None, None], derivative, t.zeros_like(derivative)
            )
            if not bool(t.isfinite(derivative).all()):
                raise RuntimeError("Nonfinite exact SCR tangent")
        return (dp, alpha, ep, derivative, int((active & ~ok).sum().item()))

    def evaluate(self, u, tangent=True):
        dL = self.B @ u
        state = self.material(dL, tangent)
        f = self.Bt @ self.half(dL - state[0])
        return (dL, state, f)

    def step(self, E, lam, tol=1e-09, **kwargs):
        started = time.time()
        m = self.m
        u = self.u.clone()
        ub = gpu((m.X @ np.asarray(E).T).ravel())
        boundary = t.zeros_like(u)
        boundary[self.fixed] = (lam - self._lam) * ub[self.fixed]
        rhs = -(self.Bt @ self.half(self.B @ boundary))[self.free]
        predictor, pit = cg(lambda x: self.K @ x, rhs, self.M, rtol=1e-10, maxiter=500)
        u[self.fixed] += boundary[self.fixed]
        u[self.free] += predictor
        counts = []
        for iteration in range(40):
            dL, state, f = self.evaluate(u)
            r = f[self.free]
            nr = norm(r)
            scale = max(norm(f[self.fixed]), 1e-30)
            if nr < tol * scale:
                break
            Dp = state[3]

            def matvec(v):
                full = t.zeros_like(u)
                full[self.free] = v
                ds = self.stress(self.B @ full)
                de = t.einsum("nij,nj->ni", Dp, ds)
                dp = self.projection(de)
                return self.K @ v - (self.Bt @ self.half(dp))[self.free]

            du, it = gmres(
                matvec,
                -r,
                self.M,
                rtol=min(0.001, max(1e-09, 0.05 * nr / scale)),
                restart=35,
                cycles=15,
            )
            counts.append(it)
            fac = 1.0
            for ls in range(18):
                trial = u.clone()
                trial[self.free] += fac * du
                _, _, ft = self.evaluate(trial, False)
                if norm(ft[self.free]) <= nr * (1 - 0.0001 * fac):
                    break
                fac *= 0.5
            else:
                raise RuntimeError(f"GPU SCR Newton line search failed at {lam}")
            u = trial
        else:
            raise RuntimeError(f"GPU SCR Newton not converged: {lam}, {nr / scale}")
        self.u = u
        self.dp, self.alpha, self.ep = state[:3]
        self._lam = lam
        m.u = cpu(u)
        m.dLp = cpu(self.dp)
        m.alpha = cpu(self.alpha)
        m.epsp = cpu(self.ep)
        m.lam = lam
        m._newton_log = {
            "solver": "FP64 GPU exact-branch SCR Newton",
            "iterations": iteration,
            "relative_force_residual": nr / scale,
            "linear_iterations": counts,
            "elastic_predictor_iterations": pit,
            "existing_material_fallback_particles": state[4],
            "seconds": time.time() - started,
        }
        return iteration


def attach(m):
    solver = Solver(m)
    m._gpu_solver = solver

    def step(self, E, lam, tol=1e-09, maxit=600, mAA=8):
        return solver.step(E, lam, tol)

    m.step = types.MethodType(step, m)
    return m
