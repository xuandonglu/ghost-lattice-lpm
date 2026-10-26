"""Same P2 Newton equations with recycled LU preconditioning of tangent solves.

GMRES uses the actual current tangent and checks its true linear residual.
Nonlinear equilibrium acceptance is unchanged. Direct factorization is used
whenever the recycled preconditioner becomes ineffective.
"""

import numpy as np
import scipy.sparse.linalg as sla
from .quadratic_elements import FEM_P2


class FEM_P2_Cached(FEM_P2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tangent_lu = self._elastic_lu
        self.linear_iterations = []
        self.refactorizations = 0

    def solve_tangent(self, K, rhs):
        A = K[self.free][:, self.free].tocsr()
        if len(rhs) < 3000:
            return sla.spsolve(A.tocsc(), rhs)
        M = sla.LinearOperator(A.shape, matvec=self._tangent_lu.solve)
        counts = []
        du, info = sla.gmres(
            A,
            rhs,
            M=M,
            rtol=1e-11,
            atol=1e-25,
            restart=30,
            maxiter=3,
            callback=counts.append,
            callback_type="pr_norm",
        )
        rel = np.linalg.norm(A @ du - rhs) / max(np.linalg.norm(rhs), 1e-30)
        self.linear_iterations.append(len(counts))
        if info or rel > 5e-10 or len(counts) > 25:
            self._tangent_lu = sla.splu(A.tocsc())
            self.refactorizations += 1
            du = self._tangent_lu.solve(rhs)
        return du

    def step(self, E, lam, material, tol=1e-09, maxiter=35):
        points = np.pad(self.mesh.p, ((0, 0), (0, self.dim - self.geometry_dim)))
        prescribed = (points @ E.T).ravel() * lam
        u = self.u.copy()
        delta = prescribed[self.fixed] - u[self.fixed]
        u[self.free] += self._elastic_lu.solve(-(self._elastic_fc @ delta))
        u[self.fixed] = prescribed[self.fixed]

        def state(v, tangent=True):
            strain = (self.B @ v).reshape(-1, self.nstrain)
            return (strain, material(strain, self.ep, self.alpha, tangent))

        for it in range(maxiter):
            strain, (stress, ep, alpha, Ct) = state(u)
            f = self.force(stress)
            scale = max(np.linalg.norm(f[self.fixed]), 1e-15)
            rel = np.linalg.norm(f[self.free]) / scale
            if rel < tol:
                break
            K = self.stiffness(Ct)
            du = self.solve_tangent(K, -f[self.free])
            old = np.linalg.norm(f[self.free])
            fac = 1.0
            for ls in range(14):
                trial = u.copy()
                trial[self.free] += fac * du
                _, ss = state(trial, False)
                fn = self.force(ss[0])
                if np.linalg.norm(fn[self.free]) <= old * (1 - 0.0001 * fac):
                    break
                fac *= 0.5
            u = trial
        else:
            raise RuntimeError(
                f"Global cached P2 Newton failed at {lam}: residual {rel}"
            )
        self.u = u
        self.strain = strain
        self.stress = stress
        self.ep = ep
        self.alpha = alpha
        self.lam = lam
        return {
            "iterations": it,
            "relative_residual": float(rel),
            "tangent_refactorizations": self.refactorizations,
        }
