"""Equivalent implicit LPM slip equations in stress/resistance coordinates.

This changes the local nonlinear solver, not the material, lattice geometry,
interface treatment or force law. No FEM field enters the calculation.
"""

import types
import numpy as np


def slip(trial, Ctensor, Ptensor, old, H, c, exponent=10.0, chi=None):
    count, m = trial.shape
    ns = Ptensor.shape[1]
    vw = np.ones(m)
    vw[2 if m == 3 else 3 :] = 2.0
    P = Ptensor * vw
    C = Ctensor / vw
    tau0 = np.einsum("nai,ni->na", P, trial)
    if chi is not None:
        tau0 -= chi
    active = np.flatnonzero(np.max(np.abs(tau0) / old[:, None], axis=1) > 0.2)
    result = np.zeros((count, ns))
    for start in range(0, len(active), 32768):
        ids = active[start : start + 32768]
        p = P[ids]
        cc = C[ids]
        tr = trial[ids]
        gold = old[ids]
        back = 0.0 if chi is None else chi[ids]
        cp = np.einsum("nij,naj->nia", cc, p)
        s = tr.copy()
        g = gold.copy()

        def equations(s, g):
            tau = np.einsum("nai,ni->na", p, s) - back
            r = np.abs(tau) / g[:, None]
            gamma = c * np.sign(tau) * r**exponent
            q = c * exponent * r ** (exponent - 1) / g[:, None]
            residual = np.column_stack(
                [
                    s - tr + np.einsum("nia,na->ni", cp, gamma),
                    g - gold - H * np.abs(gamma).sum(1),
                ]
            )
            J = np.zeros((len(ids), m + 1, m + 1))
            J[:, :m, :m] = np.eye(m)[None] + np.einsum("nia,na,naj->nij", cp, q, p)
            J[:, :m, m] = -exponent * np.einsum("nia,na->ni", cp, gamma) / g[:, None]
            J[:, m, :m] = -H * np.einsum("na,na,nai->ni", np.sign(gamma), q, p)
            J[:, m, m] = 1 + H * exponent * np.abs(gamma).sum(1) / g
            return (residual, J, gamma)

        for iteration in range(80):
            residual, J, gamma = equations(s, g)
            norm = np.max(np.abs(residual) / gold[:, None], axis=1)
            if norm.max() < 2e-13:
                break
            dx = np.linalg.solve(J, -residual[..., None])[..., 0]
            fac = np.ones(len(ids))
            for ls in range(18):
                sn = s + fac[:, None] * dx[:, :m]
                gn = g + fac * dx[:, m]
                rr, _, _ = equations(sn, np.maximum(gn, gold * 0.01))
                newnorm = np.max(np.abs(rr) / gold[:, None], axis=1)
                good = (gn > 0) & ((newnorm < norm) | (norm < 2e-13))
                if good.all():
                    break
                fac[~good] *= 0.5
            s = sn
            g = gn
        else:
            raise RuntimeError(f"Reduced local slip equations failed: {norm.max()}")
        actual_s = tr - np.einsum("nia,na->ni", cp, gamma)
        actual_g = gold + H * np.abs(gamma).sum(1)
        actual_tau = np.einsum("nai,ni->na", p, actual_s) - back
        original_residual = (
            gamma
            - c
            * np.sign(actual_tau)
            * (np.abs(actual_tau) / actual_g[:, None]) ** exponent
        )
        if np.max(np.abs(original_residual)) > 2e-10 * max(c, 1e-30):
            raise RuntimeError(
                "Original slip-equation residual failed after reduced solve"
            )
        result[ids] = gamma
    return result


def constitutive_cp(self, dL):
    dim = self.X.shape[1]
    dLe = dL - self.dLp
    D = self.Sd @ dLe
    if dim == 2:
        from .plane_lpm import KN, TT

        stress = self.pstress(2 * KN * dLe + 2 * TT * (self.Eown @ D))
    else:
        stress = self.pstress(2 * self.Kn * dLe + 2 * self.T * (self.Eown @ D))
    c = self.cp["c"]
    old = self.cp["g0"] * self.syp / self.syp.mean() + self.H * self.alpha
    if dim == 2:
        old = old + self.gG
    dg = slip(
        stress,
        self.Ci,
        self.Pcp,
        old,
        self.H,
        c,
        self.cp.get("n", 10.0),
        getattr(self, "chi", None),
    )
    self._last_dg = dg
    dep = np.einsum("na,nai->ni", dg, self.Pcp)
    dal = np.abs(dg).sum(1)
    if dim == 2:
        ddLp = self.L0 * (
            dep[self.I, 0] * self.nnL[:, 0]
            + dep[self.I, 1] * self.nnL[:, 1]
            + 2 * dep[self.I, 2] * self.nnL[:, 2]
        )
    else:
        ddLp = self.proj(dep)
    return (self.dLp + ddLp, self.alpha + dal, self.epsp + dep, dal > 0.001 * c)


def attach(model):
    assert model.plast == "cp"
    model.constitutive_cp = types.MethodType(constitutive_cp, model)
    return model
