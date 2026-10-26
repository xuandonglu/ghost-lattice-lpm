import numpy as np

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def material(kind):
    if kind == "iso":
        C11, C12 = (236.0, 140.0)
        return dict(C11=C11, C12=C12, C44=0.5 * (C11 - C12))
    if kind == "fe":
        return dict(C11=236.0, C12=140.0, C44=116.0)
    raise ValueError(kind)


def cubic_C(C11, C12, C44):
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    np.fill_diagonal(C[:3, :3], C11)
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    return C


def rotate_C(C, Q):
    """Voigt stiffness rotated by Q (specimen = Q @ crystal)."""
    c = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            for p, q in {(i, j), (j, i)}:
                for r, s in {(k, l), (l, k)}:
                    c[p, q, r, s] = C[a, b]
    cr = np.einsum("ia,jb,kc,ld,abcd->ijkl", Q, Q, Q, Q, c)
    out = np.zeros((6, 6))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            out[a, b] = cr[i, j, k, l]
    return out
