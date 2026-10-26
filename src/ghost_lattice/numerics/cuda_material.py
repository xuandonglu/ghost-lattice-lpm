"""FP64 CUDA versions of the existing independent FEM material equations.

No change to time step, constitutive parameters, activation or tolerances.
This hybrid interface returns NumPy arrays for the existing FEM global solver.
"""

import numpy as np
import torch as t

t.set_num_threads(1)
try:
    t.set_num_interop_threads(1)
except RuntimeError:
    pass


def gpu(x):
    return t.as_tensor(np.asarray(x), dtype=t.float64, device="cuda")


def cpu(x):
    return None if x is None else x.detach().cpu().numpy()


def solve(A, b):
    result = t.linalg.solve_ex(A, b, check_errors=False)
    if bool((result.info != 0).any()):
        raise RuntimeError("CUDA batched material system singular")
    return result.result


class Material:

    def __init__(self, C, sy=None, P=None, g0=None, H=0.02, c=None, exponent=10.0):
        self.C = gpu(C)
        self.sy = None if sy is None else gpu(sy)
        self.P = None if P is None else gpu(P)
        self.g0 = None if g0 is None else gpu(g0)
        self.H = H
        self.c = c
        self.exponent = exponent

    def __call__(self, strain, ep, alpha, tangent=True, extra=0.0):
        n, m = strain.shape
        C = self.C.expand(n, m, m)
        H = self.H
        e, p, a = (gpu(strain), gpu(ep), gpu(alpha))
        tr = t.einsum("nij,nj->ni", C, e - p)
        stress = tr.clone()
        p1 = p.clone()
        a1 = a.clone()
        ct = C.clone() if tangent else None
        if self.P is not None:
            P = self.P.expand(n, *self.P.shape[-2:])
            old = self.g0.expand(n) + H * a + gpu(extra)
            active = (t.abs(t.einsum("nai,ni->na", P, tr)) / old[:, None]).amax(1) > 0.2
        else:
            A = t.diag(
                gpu([1.0, 1.0, 3.0] if m == 3 else [1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
            )
            A[0, 1] = A[1, 0] = -0.5
            if m == 6:
                A[0, 2] = A[2, 0] = A[1, 2] = A[2, 1] = -0.5
            q = t.sqrt(t.clamp(t.einsum("ni,ij,nj->n", tr, A, tr), min=0.0))
            old = self.sy.expand(n) + H * a
            active = q > old + 1e-13 * t.clamp(t.abs(old), min=1e-12)
        ids = t.nonzero(active).flatten()
        if not ids.numel():
            return tuple((cpu(x) for x in (stress, p1, a1, ct)))
        cc = C[ids]
        tr0 = tr[ids]
        s = tr0.clone()
        base = old[ids]
        if self.P is not None:
            pp = P[ids]
            cp = t.einsum("nij,naj->nia", cc, pp)
            g = base.clone()
            c, ex = (self.c, self.exponent)

            def equations(s, g):
                tau = t.einsum("nai,ni->na", pp, s)
                r = t.abs(tau) / g[:, None]
                gamma = c * t.sign(tau) * r**ex
                q = c * ex * r ** (ex - 1) / g[:, None]
                res = t.cat(
                    (
                        s - tr0 + t.einsum("nia,na->ni", cp, gamma),
                        (g - base - H * t.abs(gamma).sum(1))[:, None],
                    ),
                    dim=1,
                )
                J = t.zeros((len(ids), m + 1, m + 1), dtype=t.float64, device="cuda")
                J[:, :m, :m] = t.eye(m, dtype=t.float64, device="cuda")[
                    None
                ] + t.einsum("nia,na,naj->nij", cp, q, pp)
                J[:, :m, m] = -ex * t.einsum("nia,na->ni", cp, gamma) / g[:, None]
                J[:, m, :m] = -H * t.einsum("na,na,nai->ni", t.sign(gamma), q, pp)
                J[:, m, m] = 1 + H * ex * t.abs(gamma).sum(1) / g
                return (res, J, gamma)

            for iteration in range(60):
                res, J, gamma = equations(s, g)
                norm = (t.abs(res) / base[:, None]).amax(1)
                if norm.max().item() < 2e-11:
                    break
                dx = solve(J, -res[:, :, None])[:, :, 0]
                fac = t.ones_like(g)
                for ls in range(16):
                    sn = s + fac[:, None] * dx[:, :m]
                    gn = g + fac * dx[:, m]
                    rr, _, _ = equations(sn, t.maximum(gn, base * 0.01))
                    newnorm = (t.abs(rr) / base[:, None]).amax(1)
                    good = (gn > 0) & ((newnorm < norm) | (norm < 2e-11))
                    if bool(good.all()):
                        break
                    fac = t.where(good, fac, fac * 0.5)
                s, g = (sn, gn)
            else:
                raise RuntimeError("GPU FEM CP local Newton failed")
            dep = t.einsum("na,nai->ni", gamma, pp)
            da = t.abs(gamma).sum(1)
        else:
            dl = t.zeros(len(ids), dtype=t.float64, device="cuda")
            eye = t.eye(m, dtype=t.float64, device="cuda")[None]
            for iteration in range(45):
                aq = s @ A
                qk = t.sqrt(t.clamp((s * aq).sum(1), min=1e-50))
                flow = aq / qk[:, None]
                hess = (
                    A[None] / qk[:, None, None]
                    - flow[:, :, None] * flow[:, None, :] / qk[:, None, None]
                )
                cf = t.einsum("nij,nj->ni", cc, flow)
                res = t.cat(
                    (s - tr0 + dl[:, None] * cf, (qk - base - H * dl)[:, None]), dim=1
                )
                J = t.zeros((len(ids), m + 1, m + 1), dtype=t.float64, device="cuda")
                J[:, :m, :m] = eye + dl[:, None, None] * t.einsum(
                    "nij,njk->nik", cc, hess
                )
                J[:, :m, m] = cf
                J[:, m, :m] = flow
                J[:, m, m] = -H
                if (
                    t.abs(res) / t.clamp(base[:, None], min=1e-12)
                ).max().item() < 2e-11:
                    break
                dx = solve(J, -res[:, :, None])[:, :, 0]
                fac = t.where(
                    dx[:, -1] < -dl,
                    t.minimum(
                        t.ones_like(dl), 0.95 * dl / t.clamp(-dx[:, -1], min=1e-30)
                    ),
                    t.ones_like(dl),
                )
                s = s + fac[:, None] * dx[:, :m]
                dl = dl + fac * dx[:, m]
            else:
                raise RuntimeError("GPU FEM J2 local Newton failed")
            dep = dl[:, None] * flow
            da = dl
        stress[ids] = s
        p1[ids] += dep
        a1[ids] += da
        if tangent:
            rhs = t.zeros((len(ids), m + 1, m), dtype=t.float64, device="cuda")
            rhs[:, :m] = cc
            ct[ids] = solve(J, rhs)[:, :m]
        return tuple((cpu(x) for x in (stress, p1, a1, ct)))


class Dispatcher:
    """Drop-in module patch; cache constant material data on the GPU."""

    def __init__(self):
        self.cache = {}

    def cp(
        self, strain, ep, alpha, C, P, g0, H, c, exponent=10.0, extra=0.0, tangent=True
    ):
        key = ("cp", id(C), id(P), float(H), float(c), float(exponent))
        if key not in self.cache:
            self.cache[key] = Material(C, P=P, g0=g0, H=H, c=c, exponent=exponent)
        return self.cache[key](strain, ep, alpha, tangent, extra)

    def j2(self, strain, ep, alpha, C, sy, H, tangent=True):
        key = ("j2", id(C), float(H))
        if key not in self.cache:
            self.cache[key] = Material(C, sy=sy, H=H)
        return self.cache[key](strain, ep, alpha, tangent)
