import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

S3 = np.sqrt(3.0)
A_FE = 2.8665e-10
FE = dict(C11=236.0, C12=140.0, C44=116.0)
SH1 = (
    np.array([(i, j, k) for i in (1, -1) for j in (1, -1) for k in (1, -1)], float)
    * 0.5
)
SH2 = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], float
)
BV = np.vstack([SH1, SH2])
BSH = np.array([0] * 8 + [1] * 6)
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def params(C11=FE["C11"], C12=FE["C12"], C44=FE["C44"], a=1.0, mapping="paper"):
    r = S3 * a / 4
    Kn1 = S3 * r * C44
    Kn2 = r * (C11 - C12) / S3
    T = S3 * r * (C12 - C44 if mapping == "paper" else C12 + C44) / 14.0
    return dict(
        C11=C11,
        C12=C12,
        C44=C44,
        a=a,
        r=r,
        Kn=np.array([Kn1, Kn2]),
        T=T,
        V=a**3 / 2,
        L=np.array([S3 * a / 2, a]),
        mapping=mapping,
    )


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class BCCModel:
    """BCC LPM model, directed bonds (i->j and j->i), optional periodicity along z (specimen frame)."""

    def __init__(self, X, R, p, Lz=None):
        self.X = X
        self.N = N = len(X)
        self.p = p
        self.R = R
        self.Lz = Lz
        L1, L2 = p["L"]
        if Lz is not None:
            assert Lz > 2.05 * L2, "periodic length too short (double images)"
            Xs = X - X.min(0)
            Xs[:, 2] = np.mod(X[:, 2], Lz)
            Xs[Xs[:, 2] >= Lz, 2] -= Lz
            Xs[Xs[:, 2] < 0, 2] = 0.0
            box = np.array([Xs[:, 0].max() * 3 + 10, Xs[:, 1].max() * 3 + 10, Lz])
            tree = cKDTree(Xs, boxsize=box)
        else:
            Xs = X
            tree = cKDTree(Xs)
        pairs = tree.query_pairs(1.01 * L2, output_type="ndarray")
        d = X[pairs[:, 0]] - X[pairs[:, 1]]
        if Lz is not None:
            d[:, 2] -= Lz * np.round(d[:, 2] / Lz)
        dist = np.linalg.norm(d, axis=1)
        sh = np.where(dist < 1.01 * L1, 0, 1)
        self.I = np.concatenate([pairs[:, 0], pairs[:, 1]])
        self.J = np.concatenate([pairs[:, 1], pairs[:, 0]])
        self.d = np.vstack([d, -d])
        self.sh = np.concatenate([sh, sh])
        self.L0 = np.concatenate([dist, dist])
        self.n = self.d / self.L0[:, None]
        self.npair = len(pairs)
        nb = self.nb = 2 * len(pairs)
        self.pair = np.concatenate([np.arange(len(pairs)), np.arange(len(pairs))])
        self.Kn = p["Kn"][self.sh]
        I, J = (self.I, self.J)
        rows = np.repeat(np.arange(nb), 6)
        cols = np.column_stack(
            [3 * I, 3 * I + 1, 3 * I + 2, 3 * J, 3 * J + 1, 3 * J + 2]
        ).ravel()
        vals = np.column_stack([self.n, -self.n]).ravel()
        self.B = sp.csr_matrix((vals, (rows, cols)), shape=(nb, 3 * N))
        self.S = sp.csr_matrix(
            (np.ones(nb), (2 * I + self.sh, np.arange(nb))), shape=(2 * N, nb)
        )
        self.Aown = sp.csr_matrix(
            (
                self.n.ravel(),
                (
                    np.column_stack([3 * I, 3 * I + 1, 3 * I + 2]).ravel(),
                    np.repeat(np.arange(nb), 3),
                ),
            ),
            shape=(3 * N, nb),
        )
        self.nbcount = np.bincount(I, minlength=N)
        self.nn6 = np.column_stack([self.n[:, i] * self.n[:, j] for i, j in VOIGT])
        self.alive = np.ones(self.npair, bool)

    def K(self, alive=None):
        alive = self.alive if alive is None else alive
        m = alive[self.pair].astype(float)
        Bm = sp.diags(m) @ self.B
        SB = (self.S @ Bm).tocsr()
        return (Bm.T @ sp.diags(self.Kn) @ Bm + self.p["T"] * (SB.T @ SB)).tocsr()

    def bond_forces(self, dL, alive=None):
        alive = self.alive if alive is None else alive
        m = alive[self.pair].astype(float)
        dL = dL * m
        D = self.S @ dL
        F = (
            2 * self.Kn * dL
            + self.p["T"] * (D[2 * self.I + self.sh] + D[2 * self.J + self.sh])
        ) * m
        return F

    def particle_stress(self, F):
        c = 0.5 / self.p["V"] * self.L0 * F
        return np.column_stack(
            [
                np.bincount(self.I, weights=c * self.nn6[:, k], minlength=self.N)
                for k in range(6)
            ]
        )

    def energy(self, dL, alive=None):
        alive = self.alive if alive is None else alive
        m = alive[self.pair].astype(float)
        dL = dL * m
        D = self.S @ dL
        return 0.5 * np.sum(self.Kn * dL**2) + 0.5 * self.p["T"] * np.sum(D**2)


def miller_frame(plane, front):
    """specimen frame: y = crack-plane normal, z = crack front, x = y cross z (propagation)."""
    ey = unit(plane)
    ez = unit(front)
    assert abs(ey @ ez) < 1e-12
    ex = np.cross(ey, ez)
    return np.vstack([ex, ey, ez])
