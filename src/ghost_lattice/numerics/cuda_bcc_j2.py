"""FP64 CUDA evaluation of the original 3D SCR update; original global solve."""

import numpy as np
import torch as t
import scipy.sparse as sp
from .cuda_material import gpu, cpu
from .cuda_linear import sparse


class LocalSCR:

    def __init__(self, m):
        if m.plast != "j2" or m.ret != "scr":
            raise ValueError("Original 3D SCR required")
        self.m = m
        self.original = m.constitutive
        self.calls = 0
        self.max_checked_difference = 0.0
        self.checked_calls = 0
        self.Sd = sparse(m.Sd)
        self.Eown = sparse(m.Eown)
        self.Kn = gpu(m.Kn)
        self.L = gpu(m.L0)
        self.I = t.as_tensor(m.i.astype(np.int64), device="cuda")
        self.nn = gpu(m.nn6) * gpu([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        rows = (6 * m.i[:, None] + np.arange(6)[None]).ravel()
        cols = np.repeat(np.arange(m.nb), 6)
        data = (m.wst / m.V * m.L0)[:, None] * m.nn6
        self.P = sparse(
            sp.csr_matrix((data.ravel(), (rows, cols)), shape=(6 * m.N, m.nb))
        )
        self.C = gpu(m.Ci)
        self.sy = gpu(m.syp)

    def update(self, dL):
        m = self.m
        ex = gpu(dL - m.dLp)
        S = (
            self.P @ (2 * self.Kn * ex + 2 * m.T * (self.Eown @ (self.Sd @ ex)))
        ).reshape(m.N, 6)
        sd = S.clone()
        sd[:, :3] -= S[:, :3].mean(1)[:, None]

        def Q(a, b):
            return 1.5 * (
                (a[:, :3] - a[:, :3].mean(1)[:, None])
                * (b[:, :3] - b[:, :3].mean(1)[:, None])
            ).sum(1) + 3 * (a[:, 3:] * b[:, 3:]).sum(1)

        seq = t.sqrt(t.clamp(Q(S, S), min=0.0))
        nt = 1.5 * sd / t.clamp(seq[:, None], min=1e-300)
        d = t.einsum("nij,nj->ni", self.C, nt)
        Y = self.sy + m.H * gpu(m.alpha)
        A = Q(d, d) - m.H**2
        B = Q(S, d) + Y * m.H
        Cc = Q(S, S) - Y * Y
        disc = B * B - A * Cc
        den = B + t.sqrt(t.clamp(disc, min=0.0))
        safe = t.where(t.abs(den) > 1e-300, den, t.ones_like(den))
        dl = t.where((A > 0) & (disc >= 0), Cc / safe, B / t.clamp(A, min=1e-300))
        active = seq > Y
        dl = t.where(active, t.clamp(dl, min=0.0), t.zeros_like(dl))
        de = dl[:, None] * nt
        result = (
            m.dLp + cpu(self.L * (de[self.I] * self.nn).sum(1)),
            m.alpha + cpu(dl),
            m.epsp + cpu(de),
            cpu(active),
        )
        self.calls += 1
        if self.calls <= 20 or self.calls % 100 == 0:
            original = self.original(dL)
            error = max(
                (
                    float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-14))
                    for a, b in zip(original[:3], result[:3])
                )
            )
            if error > 2e-10 or not np.array_equal(original[3], result[3]):
                raise RuntimeError(f"3D local SCR GPU equivalence failed: {error}")
            self.max_checked_difference = max(self.max_checked_difference, error)
            self.checked_calls += 1
        return result


def attach(m):
    solver = LocalSCR(m)
    m.constitutive = solver.update
    m._local_gpu = solver
    return m
