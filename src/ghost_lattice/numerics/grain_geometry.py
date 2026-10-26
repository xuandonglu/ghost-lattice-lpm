import numpy as np
from scipy.spatial import cKDTree
from .bcc_lattice import miller_frame
from .bcc_sites import sites

R_A = miller_frame((1, 0, 0), (0, 1, 0))


def rotz(deg):
    t = np.radians(deg)
    c, s = (np.cos(t), np.sin(t))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _periodic_tree(X, Lz):
    Xs = X - X.min(0)
    Xs[:, 2] = np.mod(X[:, 2], Lz)
    Xs[Xs[:, 2] >= Lz, 2] -= Lz
    Xs[Xs[:, 2] < 0, 2] = 0.0
    box = np.array([Xs[:, 0].max() * 3 + 10, Xs[:, 1].max() * 3 + 10, Lz])
    return cKDTree(Xs, boxsize=box)


def merge_overlaps(X, gid, rmerge, Lz, maxit=20):
    """Chen-Jiao-Liu 2015 replacing procedure: pairs closer than rmerge -> one particle at the midpoint (gid -1)."""
    X = X.copy()
    gid = gid.copy()
    nmerge = 0
    for it in range(maxit):
        tree = _periodic_tree(X, Lz)
        pr = tree.query_pairs(rmerge, output_type="ndarray")
        if len(pr) == 0:
            break
        used = np.zeros(len(X), bool)
        keep = np.ones(len(X), bool)
        for i, j in pr:
            if used[i] or used[j]:
                continue
            d = X[j] - X[i]
            d[2] -= Lz * np.round(d[2] / Lz)
            X[i] = X[i] + 0.5 * d
            X[i, 2] = np.mod(X[i, 2], Lz)
            gid[i] = -1
            keep[j] = False
            used[i] = used[j] = True
            nmerge += 1
        X = X[keep]
        gid = gid[keep]
    return (X, gid, nmerge)


def build_cloud(frames, assign, lo, hi, Lz, rmerge=0.52, shifts=None):
    """frames: list of Rm per grain; assign(X) -> grain index array; shifts: specimen-frame rigid translation of
    each grain lattice (GB-structure variation).  Returns X, gid, nmerge."""
    Xs, G = ([], [])
    for g, Rm in enumerate(frames):
        sh = (
            None
            if shifts is None or shifts[g] is None
            else Rm.T @ np.asarray(shifts[g], float)
        )
        Xg = sites(Rm, (lo[0], lo[1], 0.0), (hi[0], hi[1], Lz), shift=sh)
        k = assign(Xg) == g
        Xs.append(Xg[k])
        G.append(np.full(k.sum(), g))
    X = np.vstack(Xs)
    gid = np.concatenate(G)
    return merge_overlaps(X, gid, rmerge, Lz)
