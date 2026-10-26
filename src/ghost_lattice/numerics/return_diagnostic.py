import numpy as np
from .plane_lpm import KN, TT, Cmat


def cmat_particles(m):
    """C_i (N, 3, 3), Voigt [xx, yy, xy] stress per tensor-component plastic strain [xx, yy, xy] (own bonds only)"""
    N = m.N
    Ci = np.zeros((N, 3, 3))
    for k, e in enumerate(np.eye(3)):
        ddLp = m.L0 * (e[0] * m.nnL[:, 0] + e[1] * m.nnL[:, 1] + 2 * e[2] * m.nnL[:, 2])
        D = m.Sd @ ddLp
        F = 2 * KN * ddLp + 2 * TT * (m.Eown @ D)
        Ci[:, :, k] = m.pstress(F)
    return Ci


def kappa(m, sig):
    """sig: Voigt stress direction (uniform state); returns kappa_i"""
    s = np.asarray(sig, float)
    mm = (s[0] + s[1]) / 3
    dev = np.array([s[0] - mm, s[1] - mm, -mm, s[2]])
    seq = np.sqrt(1.5 * (dev[0] ** 2 + dev[1] ** 2 + dev[2] ** 2 + 2 * dev[3] ** 2))
    de = 1.5 / seq * np.array([dev[0], dev[1], dev[3]])
    Ci = cmat_particles(m)
    dsig = Ci @ de
    dsig0 = Cmat(1.0, 0.3) @ (de * np.array([1, 1, 2]))
    n = 1.5 / seq * dev

    def proj(ds):
        m_ = (ds[..., 0] + ds[..., 1]) / 3
        return (
            n[0] * (ds[..., 0] - m_)
            + n[1] * (ds[..., 1] - m_)
            + n[2] * -m_
            + 2 * n[3] * ds[..., 2]
        )

    return (proj(dsig) / proj(dsig0), proj(dsig0))
