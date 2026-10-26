"""FP64 numerical solver for the original rotated MLS/GFC and merging laws.

Reuse the verified original SCR material and exact derivative. Generalize only
the linear kinematic/force operators, retaining the original GFC in both the
residual and its derivative. No change to the coupling or material equations.
"""

import time, types
import numpy as np
import scipy.sparse as sp
import torch as t
from .cuda_j2_base import Solver as UniformSolver
from .cuda_linear import sparse, norm, cg, gmres, AMG
from .cuda_material import gpu, cpu
from .plane_lpm import VV


class Solver(UniformSolver):

    def __init__(self, m):
        if m.kind not in ["ghost", "j04"] or m.plast != "j2" or m.ret != "scr":
            raise ValueError("Original planar J2 only")
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
        cols = np.repeat(np.arange(m.nb, dtype=np.int32), 3)
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
        self.R = None if getattr(m, "gfc", None) is None else gpu(m.gfc)
        self.G = None if self.R is None else sparse(m.Gstrain)
        if hasattr(m.lu, "ml"):
            self.M = AMG(m.lu.ml, 2)
            self.K = self.M.levels[0]["A"]
        else:
            self.K = sparse(m.K[m.free][:, m.free])
            self.M = lambda x: gpu(m.lu.solve(cpu(x)))
        self._lam = m.lam

    def force(self, elastic_extension):
        result = self.Bt @ self.half(elastic_extension)
        if self.R is not None:
            strain = (self.G @ elastic_extension).reshape(self.m.N, 3)
            result -= t.einsum("nck,nk->nc", self.R, strain).ravel()
        return result

    def evaluate(self, u, tangent=True):
        dL = self.B @ u
        state = self.material(dL, tangent)
        return (dL, state, self.force(dL - state[0]))

    def step(self, E, lam, tol=1e-09, **kwargs):
        started = time.time()
        m = self.m
        u = self.u.clone()
        boundary = t.zeros_like(u)
        affine = gpu((m.X @ np.asarray(E).T).ravel())
        boundary[self.fixed] = (lam - self._lam) * affine[self.fixed]
        rhs = -self.force(self.B @ boundary)[self.free]

        def elastic(v):
            full = t.zeros_like(u)
            full[self.free] = v
            return self.force(self.B @ full)[self.free]

        if self.R is None:
            predictor, pit = cg(
                lambda x: self.K @ x, rhs, self.M, rtol=1e-10, maxiter=500
            )
        else:
            predictor, pit = gmres(
                elastic, rhs, self.M, rtol=1e-10, restart=35, cycles=15
            )
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
                extension = self.B @ full
                ds = self.stress(extension)
                de = t.einsum("nij,nj->ni", Dp, ds)
                return self.force(extension - self.projection(de))[self.free]

            du, it = gmres(
                matvec,
                -r,
                self.M,
                rtol=min(0.001, max(1e-09, 0.05 * nr / scale)),
                restart=35,
                cycles=15,
            )
            counts.append(it)
            factor = 1.0
            for ls in range(18):
                trial = u.clone()
                trial[self.free] += factor * du
                _, _, ft = self.evaluate(trial, False)
                if norm(ft[self.free]) <= nr * (1 - 0.0001 * factor):
                    break
                factor *= 0.5
            else:
                raise RuntimeError(
                    f"Original rotated J2 Newton line search failed at {lam}"
                )
            u = trial
        else:
            raise RuntimeError(
                f"Original rotated J2 Newton exhausted iterations at {lam}"
            )
        self.u = u
        self.dp, self.alpha, self.ep = state[:3]
        self._lam = lam
        m.u = cpu(u)
        m.dLp = cpu(self.dp)
        m.alpha = cpu(self.alpha)
        m.epsp = cpu(self.ep)
        m.lam = lam
        m._newton_log = {
            "solver": "FP64 CUDA Newton on original rotated force law",
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
