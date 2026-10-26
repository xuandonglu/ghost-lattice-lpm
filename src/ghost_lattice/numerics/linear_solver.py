import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


class MultigridInverse:

    def __init__(self, A, X, dim, periodic_z=False):
        import pyamg

        self.A = A.tocsr()
        n = len(X)
        modes = []
        for j in range(dim):
            v = np.zeros((n, dim))
            v[:, j] = 1.0
            modes.append(v.ravel())
        centered = X - X.mean(axis=0)
        for j in range(dim):
            for k in range(j + 1, dim):
                if periodic_z and k == 2:
                    continue
                v = np.zeros((n, dim))
                v[:, j] = -centered[:, k]
                v[:, k] = centered[:, j]
                modes.append(v.ravel())
        B = np.column_stack(modes)
        B /= np.linalg.norm(B, axis=0)
        self.ml = pyamg.smoothed_aggregation_solver(
            A.tobsr(blocksize=(dim, dim)),
            B=B,
            symmetry="symmetric",
            smooth="energy",
            max_coarse=300,
        )
        self.M = self.ml.aspreconditioner(cycle="V")
        self.solve_iterations = []
        print("AMG_READY", A.shape, A.nnz, "levels", len(self.ml.levels), flush=True)

    def solve(self, rhs):
        self.solve_iterations.append(1)
        x = self.M @ rhs
        if not np.isfinite(x).all():
            raise RuntimeError("Nonfinite multigrid correction")
        return x


def inverse(A, m):
    if m.N > 50000:
        nodes = np.flatnonzero(m.free.reshape(m.N, -1).all(axis=1))
        return MultigridInverse(
            A,
            m.X[nodes],
            m.X.shape[1],
            periodic_z=getattr(m, "lz", 1000000.0) < 100000.0,
        )
    return sla.splu(A.tocsc())
