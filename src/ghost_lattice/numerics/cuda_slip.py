"""Reusable, unchanged GPUSlip kernel from the verified actual-case GPU pilot.

Only the local slip solve runs on CUDA. Lattice, MLS, GFC, forces, elastic
preconditioner and the original Anderson equilibrium iteration stay unchanged.
"""

import numpy as np
import torch as t
from .plane_lpm import KN, TT


def gpu_array(x):
    return t.as_tensor(np.asarray(x), dtype=t.float64, device="cuda")


def cpu_array(x):
    return x.detach().cpu().numpy()


class GPUSlip:

    def __init__(self, m):
        self.m = m
        self.Ptensor = gpu_array(m.Pcp)
        vw = t.tensor([1.0, 1.0, 2.0], dtype=t.float64, device="cuda")
        self.P = self.Ptensor * vw
        self.C = gpu_array(m.Ci) / vw
        self.CP = t.einsum("nij,naj->nia", self.C, self.P)

    def update(self, dL):
        m = self.m
        dLe = dL - m.dLp
        D = m.Sd @ dLe
        stress = m.pstress(2 * KN * dLe + 2 * TT * (m.Eown @ D))
        tr = gpu_array(stress)
        old = gpu_array(m.cp["g0"] * m.syp / m.syp.mean() + m.H * m.alpha + m.gG)
        chi = getattr(m, "chi", None)
        back = 0.0 if chi is None else gpu_array(chi)
        tau = t.einsum("nai,ni->na", self.P, tr) - back
        ids = t.nonzero((t.abs(tau) / old[:, None]).amax(1) > 0.2).flatten()
        dg = t.zeros((m.N, m.Pcp.shape[1]), dtype=t.float64, device="cuda")
        if ids.numel():
            p = self.P[ids]
            cp = self.CP[ids]
            trial = tr[ids]
            gold = old[ids]
            bc = 0.0 if chi is None else back[ids]
            s = trial.clone()
            g = gold.clone()
            c = m.cp["c"]
            exponent = m.cp.get("n", 10.0)
            H = m.H

            def equations(s, g):
                tau = t.einsum("nai,ni->na", p, s) - bc
                r = t.abs(tau) / g[:, None]
                gamma = c * t.sign(tau) * r**exponent
                q = c * exponent * r ** (exponent - 1) / g[:, None]
                residual = t.cat(
                    (
                        s - trial + t.einsum("nia,na->ni", cp, gamma),
                        (g - gold - H * t.abs(gamma).sum(1))[:, None],
                    ),
                    dim=1,
                )
                J = t.zeros((len(ids), 4, 4), dtype=t.float64, device="cuda")
                J[:, :3, :3] = t.eye(3, dtype=t.float64, device="cuda")[
                    None
                ] + t.einsum("nia,na,naj->nij", cp, q, p)
                J[:, :3, 3] = -exponent * t.einsum("nia,na->ni", cp, gamma) / g[:, None]
                J[:, 3, :3] = -H * t.einsum("na,na,nai->ni", t.sign(gamma), q, p)
                J[:, 3, 3] = 1 + H * exponent * t.abs(gamma).sum(1) / g
                return (residual, J, gamma)

            for iteration in range(80):
                residual, J, gamma = equations(s, g)
                norm = t.amax(t.abs(residual) / gold[:, None], dim=1)
                if norm.max().item() < 2e-13:
                    break
                dx = t.linalg.solve_ex(
                    J, -residual[:, :, None], check_errors=False
                ).result[:, :, 0]
                fac = t.ones(len(ids), dtype=t.float64, device="cuda")
                for ls in range(18):
                    sn = s + fac[:, None] * dx[:, :3]
                    gn = g + fac * dx[:, 3]
                    rr, _, _ = equations(sn, t.maximum(gn, gold * 0.01))
                    newnorm = t.amax(t.abs(rr) / gold[:, None], dim=1)
                    good = (gn > 0) & ((newnorm < norm) | (norm < 2e-13))
                    if bool(good.all()):
                        break
                    fac = t.where(good, fac, fac * 0.5)
                s = sn
                g = gn
            else:
                raise RuntimeError("GPU local CP did not converge")
            actual_s = trial - t.einsum("nia,na->ni", cp, gamma)
            actual_g = gold + H * t.abs(gamma).sum(1)
            actual_tau = t.einsum("nai,ni->na", p, actual_s) - bc
            slip_res = (
                gamma
                - c
                * t.sign(actual_tau)
                * (t.abs(actual_tau) / actual_g[:, None]) ** exponent
            )
            if t.max(t.abs(slip_res)).item() > 2e-10 * max(c, 1e-30):
                raise RuntimeError("GPU original-slip residual failed")
            dg[ids] = gamma
        dg = cpu_array(dg)
        dep = np.einsum("na,nai->ni", dg, m.Pcp)
        dal = np.abs(dg).sum(1)
        ddLp = m.L0 * (
            dep[m.I, 0] * m.nnL[:, 0]
            + dep[m.I, 1] * m.nnL[:, 1]
            + 2 * dep[m.I, 2] * m.nnL[:, 2]
        )
        m._last_dg = dg
        return (m.dLp + ddLp, m.alpha + dal, m.epsp + dep, dal > 0.001 * m.cp["c"])
