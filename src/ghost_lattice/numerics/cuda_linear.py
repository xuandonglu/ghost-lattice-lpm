"""FP64 CUDA sparse products and residual-checked Krylov solvers."""

import numpy as np
import torch as t
from .cuda_material import gpu, cpu


def sparse(A):
    A = A.tocsr()
    A.sort_indices()
    return t.sparse_csr_tensor(
        t.as_tensor(A.indptr, device="cuda"),
        t.as_tensor(A.indices, device="cuda"),
        gpu(A.data),
        size=A.shape,
        device="cuda",
    )


def norm(x):
    return t.linalg.vector_norm(x).item()


def cg(A, b, M=None, rtol=1e-10, maxiter=400):
    x = t.zeros_like(b)
    r = b.clone()
    bn = norm(b)
    if bn < 1e-30:
        return (x, 0)
    z = r.clone() if M is None else M(r)
    p = z.clone()
    rz = t.dot(r, z)
    for k in range(maxiter):
        ap = A(p)
        den = t.dot(p, ap)
        if den.item() <= 0:
            raise RuntimeError("CUDA CG encountered a non-positive tangent")
        al = rz / den
        x += al * p
        r -= al * ap
        if norm(r) < rtol * bn:
            true = b - A(x)
            if norm(true) < max(rtol * bn, 1e-28):
                return (x, k + 1)
            r = true
        z = r.clone() if M is None else M(r)
        new = t.dot(r, z)
        p = z + new / rz * p
        rz = new
    raise RuntimeError(f"CUDA CG did not converge: {norm(b - A(x)) / bn}, {maxiter}")


def gmres(A, b, M=None, rtol=1e-08, restart=35, cycles=12):
    """Right-preconditioned restarted GMRES, modified GS with reorthogonalization."""
    x = t.zeros_like(b)
    bn = norm(b)
    count = 0
    if bn < 1e-30:
        return (x, 0)
    for outer in range(cycles):
        r = b - A(x)
        beta = norm(r)
        if beta < rtol * bn:
            return (x, count)
        V = t.zeros((len(b), restart + 1), dtype=t.float64, device="cuda")
        Z = t.empty((len(b), restart), dtype=t.float64, device="cuda")
        H = np.zeros((restart + 1, restart))
        g = np.zeros(restart + 1)
        g[0] = beta
        cs = np.zeros(restart)
        sn = np.zeros(restart)
        V[:, 0] = r / beta
        for j in range(restart):
            Z[:, j] = V[:, j] if M is None else M(V[:, j])
            w = A(Z[:, j])
            basis = V[:, : j + 1]
            h = basis.T @ w
            w -= basis @ h
            h2 = basis.T @ w
            w -= basis @ h2
            h += h2
            hn = norm(w)
            H[: j + 1, j] = cpu(h)
            H[j + 1, j] = hn
            if hn > 1e-30:
                V[:, j + 1] = w / hn
            for k in range(j):
                H[k, j], H[k + 1, j] = (
                    cs[k] * H[k, j] + sn[k] * H[k + 1, j],
                    -sn[k] * H[k, j] + cs[k] * H[k + 1, j],
                )
            hyp = np.hypot(H[j, j], H[j + 1, j])
            cs[j] = H[j, j] / hyp
            sn[j] = H[j + 1, j] / hyp
            H[j, j] = hyp
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            count += 1
            if abs(g[j + 1]) < rtol * bn or hn < 1e-30:
                break
        y = np.linalg.solve(H[: j + 1, : j + 1], g[: j + 1])
        x += Z[:, : j + 1] @ gpu(y)
        del V, Z
        if norm(b - A(x)) < rtol * bn:
            return (x, count)
    raise RuntimeError(f"CUDA GMRES did not converge: {norm(b - A(x)) / bn}, {count}")


class AMG:
    """Existing Galerkin hierarchy with symmetric block-Jacobi CUDA V-cycles."""

    def __init__(self, ml, block=2):
        self.levels = []
        self.block = block
        for k, level in enumerate(ml.levels):
            block = getattr(level.A, "blocksize", (block if k == 0 else 1,))[0]
            A = level.A.tocsr()
            entry = {"A": sparse(A), "block": block}
            if k == len(ml.levels) - 1:
                entry["chol"] = t.linalg.cholesky(gpu(A.toarray()))
            else:
                diag = np.stack(
                    [
                        A[i::block, j::block].diagonal()
                        for i in range(block)
                        for j in range(block)
                    ],
                    axis=1,
                ).reshape(-1, block, block)
                entry["dinv"] = gpu(np.linalg.inv(diag))
                entry["P"] = sparse(level.P)
                entry["R"] = sparse(level.R)

                def jac(v):
                    return t.einsum(
                        "nij,nj->ni", entry["dinv"], v.reshape(-1, block)
                    ).ravel()

                v = gpu(np.random.default_rng(k + 17).normal(size=A.shape[0]))
                v /= t.linalg.vector_norm(v)
                for _ in range(15):
                    v = jac(entry["A"] @ v)
                    v /= t.linalg.vector_norm(v)
                rho = norm(jac(entry["A"] @ v))
                entry["omega"] = 1.25 / rho
            self.levels.append(entry)

    def __call__(self, b):
        return self.cycle(0, b)

    def cycle(self, k, b):
        L = self.levels[k]
        if "chol" in L:
            return t.cholesky_solve(b[:, None], L["chol"])[:, 0]

        def correction(r):
            return (
                L["omega"]
                * t.einsum("nij,nj->ni", L["dinv"], r.reshape(-1, L["block"])).ravel()
            )

        x = correction(b)
        for _ in range(2):
            x += correction(b - L["A"] @ x)
        x += L["P"] @ self.cycle(k + 1, L["R"] @ (b - L["A"] @ x))
        for _ in range(3):
            x += correction(b - L["A"] @ x)
        return x
