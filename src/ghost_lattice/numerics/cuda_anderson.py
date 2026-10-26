"""Original Anderson equilibrium iteration, resident FP64 CUDA operators.

Same original SCR states/forces, mAA=8, 600 iteration bound and force tolerance.
Elastic corrections are solved with residual-checked CG instead of CPU LU.
QR plus small CPU SVD implements the original rank-aware least-squares solve.
"""

import time
import numpy as np
import torch as t
from .cuda_linear import norm, cg
from .cuda_material import gpu, cpu


def step(self, E, lam, tol=1e-09, **kwargs):
    if self.R is not None:
        raise ValueError("This retry is restricted to real reciprocal merging bonds")
    start = time.time()
    m = self.m
    u = self.u.clone()
    affine = gpu((m.X @ np.asarray(E).T).ravel())
    boundary = t.zeros_like(u)
    boundary[self.fixed] = (lam - self._lam) * affine[self.fixed]
    rhs = -self.force(self.B @ boundary)[self.free]
    predictor, pit = cg(lambda x: self.K @ x, rhs, self.M, rtol=1e-10, maxiter=500)
    u[self.fixed] += boundary[self.fixed]
    u[self.free] += predictor
    x = u[self.free].clone()
    hdx = []
    hdf = []
    xp = gp = None
    best = float("inf")
    linear = []
    for iteration in range(600):
        u[self.free] = x
        _, state, f = self.evaluate(u, False)
        r = f[self.free]
        nr = norm(r)
        scale = max(norm(f[self.fixed]), 1e-30)
        if nr < tol * scale:
            break
        correction, its = cg(lambda v: self.K @ v, -r, self.M, rtol=1e-10, maxiter=500)
        linear.append(its)
        g = correction
        if xp is not None:
            hdx.append(x - xp)
            hdf.append(g - gp)
            if len(hdx) > 8:
                hdx.pop(0)
                hdf.pop(0)
        if nr > 50 * best:
            hdx = []
            hdf = []
        best = min(best, nr)
        xp = x.clone()
        gp = g.clone()
        if hdx:
            DX = t.column_stack(hdx)
            DF = t.column_stack(hdf)
            Q, R = t.linalg.qr(DF, mode="reduced")
            rhs_small = cpu(Q.T @ g)
            rcond = np.finfo(float).eps * max(DF.shape)
            gamma = gpu(np.linalg.lstsq(cpu(R), rhs_small, rcond=rcond)[0])
            x = x + g - (DX + DF) @ gamma
            del DX, DF, Q, R
        else:
            x = x + g
        if iteration % 25 == 0:
            print(
                "ORIGINAL_ANDERSON",
                {
                    "lam": lam,
                    "iteration": iteration,
                    "relative_force_residual": nr / scale,
                    "seconds": time.time() - start,
                },
                flush=True,
            )
    else:
        raise RuntimeError(
            f"Original Anderson exhausted 600 iterations at {lam}; relative force {nr / scale}"
        )
    self.u = u
    self.dp, self.alpha, self.ep = state[:3]
    self._lam = lam
    m.u = cpu(u)
    m.dLp = cpu(self.dp)
    m.alpha = cpu(self.alpha)
    m.epsp = cpu(self.ep)
    m.lam = lam
    m._newton_log = dict(
        solver="Original Anderson mAA8 on unchanged merging force law, FP64 CUDA; elastic CG1e-10",
        iterations=iteration,
        relative_force_residual=nr / scale,
        linear_iterations=linear,
        elastic_predictor_iterations=pit,
        existing_material_fallback_particles=state[4],
        seconds=time.time() - start,
    )
    return iteration
