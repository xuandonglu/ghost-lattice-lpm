"""FP64 local solve of the existing3D implicit12-slip equations.

Seven stress/resistance unknowns replace twelve slip unknowns algebraically.
Geometry, force law, material and the original global equilibrium are retained.
"""

import numpy as np
import torch as t
from .cuda_material import gpu, cpu


class LocalCP:

    def __init__(self, m):
        assert m.X.shape[1] == 3 and m.plast == "cp"
        self.m = m
        self.calls = 0
        self.checks = []
        vw = gpu([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.P = gpu(m.Pcp) * vw
        self.C = gpu(m.Ci) / vw
        self.CP = t.einsum("nij,naj->nia", self.C, self.P)

    def update(self, dL):
        m = self.m
        dLe = dL - m.dLp
        trial = m.pstress(2 * m.Kn * dLe + 2 * m.T * (m.Eown @ (m.Sd @ dLe)))
        tr = gpu(trial)
        gold = gpu(m.cp["g0"] * m.syp / m.syp.mean() + m.H * m.alpha)
        tau = t.einsum("nai,ni->na", self.P, tr)
        active = t.nonzero((t.abs(tau) / gold[:, None]).amax(1) > 0.2).flatten()
        dg = t.zeros((m.N, m.Pcp.shape[1]), dtype=t.float64, device="cuda")
        if active.numel():
            p = self.P[active]
            cp = self.CP[active]
            tr = tr[active]
            old = gold[active]
            s = tr.clone()
            g = old.clone()
            H = m.H
            c = m.cp["c"]
            n = m.cp.get("n", 10.0)

            def equations(s, g):
                tau = t.einsum("nai,ni->na", p, s)
                r = t.abs(tau) / g[:, None]
                gamma = c * t.sign(tau) * r**n
                q = c * n * r ** (n - 1) / g[:, None]
                rr = t.cat(
                    (
                        s - tr + t.einsum("nia,na->ni", cp, gamma),
                        (g - old - H * t.abs(gamma).sum(1))[:, None],
                    ),
                    dim=1,
                )
                J = t.zeros((len(active), 7, 7), dtype=t.float64, device="cuda")
                J[:, :6, :6] = t.eye(6, dtype=t.float64, device="cuda")[
                    None
                ] + t.einsum("nia,na,naj->nij", cp, q, p)
                J[:, :6, 6] = -n * t.einsum("nia,na->ni", cp, gamma) / g[:, None]
                J[:, 6, :6] = -H * t.einsum("na,na,nai->ni", t.sign(gamma), q, p)
                J[:, 6, 6] = 1 + H * n * t.abs(gamma).sum(1) / g
                return (rr, J, gamma)

            for iteration in range(80):
                rr, J, gamma = equations(s, g)
                norm = (t.abs(rr) / old[:, None]).amax(1)
                if norm.max().item() < 2e-13:
                    break
                dx = t.linalg.solve_ex(J, -rr[:, :, None], check_errors=False).result[
                    :, :, 0
                ]
                fac = t.ones(len(active), dtype=t.float64, device="cuda")
                for ls in range(18):
                    sn = s + fac[:, None] * dx[:, :6]
                    gn = g + fac * dx[:, 6]
                    rn, _, _ = equations(sn, t.maximum(gn, old * 0.01))
                    newnorm = (t.abs(rn) / old[:, None]).amax(1)
                    good = (gn > 0) & ((newnorm < norm) | (norm < 2e-13))
                    if bool(good.all()):
                        break
                    fac = t.where(good, fac, fac * 0.5)
                s = sn
                g = gn
            else:
                raise RuntimeError(
                    "Original3D CP local equations did not converge on GPU"
                )
            actual_s = tr - t.einsum("nia,na->ni", cp, gamma)
            actual_g = old + H * t.abs(gamma).sum(1)
            actual_tau = t.einsum("nai,ni->na", p, actual_s)
            rr = (
                gamma
                - c * t.sign(actual_tau) * (t.abs(actual_tau) / actual_g[:, None]) ** n
            )
            if t.max(t.abs(rr)).item() > 2e-10 * max(c, 1e-30):
                raise RuntimeError("Original12-slip residual failed on GPU")
            dg[active] = gamma
        dg = cpu(dg)
        m._last_dg = dg
        dep = np.einsum("na,nai->ni", dg, m.Pcp)
        dal = np.abs(dg).sum(1)
        answer = (
            m.dLp + m.proj(dep),
            m.alpha + dal,
            m.epsp + dep,
            dal > 0.001 * m.cp["c"],
        )
        self.calls += 1
        if self.calls <= 3 or self.calls % 100 == 0:
            from .implicit_slip import constitutive_cp

            expected = constitutive_cp(m, dL)
            differences = [
                float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-14))
                for a, b in zip(expected[:3], answer[:3])
            ]
            if max(differences) > 2e-07:
                raise RuntimeError(
                    f"Independent original3D CP CPU/GPU mismatch {differences}"
                )
            self.checks.append(dict(call=self.calls, differences=differences))
            m._last_dg = dg
        return answer


def attach(m):
    obj = LocalCP(m)
    m.constitutive_cp = obj.update
    m._local_gpu = obj
    return m
