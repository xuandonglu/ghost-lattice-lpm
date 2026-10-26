"""ebsd_grains.py -- EBSD (Oxford .ctf text export) -> grain map for the 2D LPM models of Paper A.
  read_ctf(path)                      -> dict(x, y, step, nx, ny, phase, euler (deg, Bunge), bc, ...) on the regular grid
  segment(euler, phase, nx, ny, tol)  -> grain id raster (ny, nx): union-find over 4-neighbour pairs whose cubic
                                         misorientation < tol (deg); unindexed pixels (phase 0) are filled from neighbours
  merge_small(G, amin)                -> grains smaller than amin pixels merged into their longest-shared neighbour
  schmid_fcc(euler, axis)             -> max Schmid factor of the 12 {111}<110> systems for uniaxial load along `axis`
                                         (sample frame), per orientation
Pure numpy/scipy; the data files are read as text, never executed."""

import numpy as np
from scipy import ndimage


def read_ctf(path):
    with open(path, "r", errors="replace") as f:
        hdr = {}
        for line in f:
            if line.startswith("Phase\tX\tY") or line.startswith("Phase X Y"):
                cols = line.split()
                break
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                hdr[p[0]] = p[1:]
        data = np.loadtxt(f, dtype=np.float64)
    c = {k: i for i, k in enumerate(cols)}
    nx, ny = (int(hdr["XCells"][0]), int(hdr["YCells"][0]))
    step = float(hdr["XStep"][0])
    out = dict(
        nx=nx,
        ny=ny,
        step=step,
        header=hdr,
        x=data[:, c["X"]].reshape(ny, nx),
        y=data[:, c["Y"]].reshape(ny, nx),
        phase=data[:, c["Phase"]].astype(int).reshape(ny, nx),
        euler=np.stack(
            [data[:, c["Euler1"]], data[:, c["Euler2"]], data[:, c["Euler3"]]], 1
        ).reshape(ny, nx, 3),
    )
    for k in ("BC", "BS", "MAD", "Bands", "Error"):
        if k in c:
            out[k.lower()] = data[:, c[k]].reshape(ny, nx)
    return out


def euler_to_quat(e_deg):
    """Bunge (phi1, Phi, phi2) in degrees -> unit quaternions (..., 4) [w, x, y, z] (passive, crystal -> sample)"""
    e = np.radians(e_deg)
    p1, P, p2 = (e[..., 0], e[..., 1], e[..., 2])
    c, s = (np.cos(P / 2), np.sin(P / 2))
    q = np.stack(
        [
            c * np.cos((p1 + p2) / 2),
            s * np.cos((p1 - p2) / 2),
            s * np.sin((p1 - p2) / 2),
            c * np.sin((p1 + p2) / 2),
        ],
        -1,
    )
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _cubic_syms():
    """the 24 proper rotations of the cubic group as quaternions"""
    s = 1 / np.sqrt(2)
    q = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [s, s, 0, 0],
        [s, 0, s, 0],
        [s, 0, 0, s],
        [s, -s, 0, 0],
        [s, 0, -s, 0],
        [s, 0, 0, -s],
        [0, s, s, 0],
        [0, s, -s, 0],
        [0, s, 0, s],
        [0, s, 0, -s],
        [0, 0, s, s],
        [0, 0, s, -s],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, 0.5, -0.5, -0.5],
    ]
    return np.array(q, float)


SYM = _cubic_syms()


def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        -1,
    )


def misorientation_deg(qa, qb):
    """cubic misorientation angle (deg) between orientation arrays qa, qb (n, 4): min over the 24 symmetry operators of
    the angle of qa^-1 qb s (the crystal-symmetry equivalent); |w| of the product gives the angle.
    """
    qinv = qa * np.array([1, -1, -1, -1])
    d = qmul(qinv, qb)
    ws = np.abs(qmul(d[:, None, :], SYM[None, :, :])[..., 0])
    return np.degrees(2 * np.arccos(np.clip(ws.max(1), 0, 1)))


def segment(euler, phase, tol=5.0):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ny, nx = phase.shape
    q = euler_to_quat(euler).reshape(-1, 4)
    good = phase.ravel() > 0
    idx = np.arange(nx * ny).reshape(ny, nx)
    ra, rb = ([], [])
    for a, b in (
        (idx[:, :-1].ravel(), idx[:, 1:].ravel()),
        (idx[:-1, :].ravel(), idx[1:, :].ravel()),
    ):
        ok = good[a] & good[b]
        a, b = (a[ok], b[ok])
        same = np.zeros(len(a), bool)
        for s in range(0, len(a), 500000):
            same[s : s + 500000] = (
                misorientation_deg(q[a[s : s + 500000]], q[b[s : s + 500000]]) < tol
            )
        ra.append(a[same])
        rb.append(b[same])
    ra, rb = (np.concatenate(ra), np.concatenate(rb))
    A = coo_matrix((np.ones(len(ra)), (ra, rb)), shape=(nx * ny, nx * ny)).tocsr()
    _, lab = connected_components(A, directed=False)
    lab = np.unique(lab, return_inverse=True)[1].reshape(ny, nx)
    lab[~good.reshape(ny, nx)] = -1
    return fill_unindexed(lab)


def fill_unindexed(G):
    """pixels with id -1 take the id of the nearest indexed pixel"""
    bad = G < 0
    if not bad.any():
        return G
    _, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
    return G[iy, ix]


def _boundary_pairs(G):
    a = np.r_[G[:, :-1].ravel(), G[:-1, :].ravel()]
    b = np.r_[G[:, 1:].ravel(), G[1:, :].ravel()]
    k = a != b
    return (np.r_[a[k], b[k]], np.r_[b[k], a[k]])


def _merge_into_neighbours(G, flagged):
    """relabel every flagged grain to the unflagged neighbour with the longest shared boundary (vectorised)"""
    u, v = _boundary_pairs(G)
    ng = int(G.max()) + 1
    k = flagged[u] & ~flagged[v]
    if not k.any():
        return (G, False)
    key = u[k].astype(np.int64) * ng + v[k]
    keys, cnt = np.unique(key, return_counts=True)
    uu, vv = (keys // ng, keys % ng)
    order = np.lexsort((-cnt, uu))
    uu, vv = (uu[order], vv[order])
    first = np.r_[True, uu[1:] != uu[:-1]]
    mapping = np.arange(ng)
    mapping[uu[first]] = vv[first]
    return (mapping[G], True)


def merge_small(G, amin):
    """merge grains with fewer than amin pixels into the neighbour sharing the longest boundary; relabel 0..n-1"""
    G = np.unique(G, return_inverse=True)[1].reshape(G.shape)
    for _ in range(50):
        cnt = np.bincount(G.ravel())
        flagged = cnt < amin
        if not flagged.any():
            break
        G, changed = _merge_into_neighbours(G, flagged)
        G = np.unique(G, return_inverse=True)[1].reshape(G.shape)
        if not changed:
            break
    return G


def _inside_distance(G):
    """distance (pixels) of every pixel to its own grain's boundary (one transform for the whole map)"""
    b = np.zeros_like(G, bool)
    b[:, :-1] |= G[:, :-1] != G[:, 1:]
    b[:, 1:] |= G[:, :-1] != G[:, 1:]
    b[:-1, :] |= G[:-1, :] != G[1:, :]
    b[1:, :] |= G[:-1, :] != G[1:, :]
    return ndimage.distance_transform_edt(~b) + 1.0


def grain_mean_quat(G, euler):
    """approximate orientation per grain: the orientation at the grain's most central pixel (maximum inscribed distance)"""
    ng = int(G.max()) + 1
    d = _inside_distance(G)
    pos = ndimage.maximum_position(d, labels=G, index=np.arange(ng))
    return np.array([euler[p] for p in pos])


def schmid_fcc(euler_deg, axis=(1.0, 0.0, 0.0)):
    """max Schmid factor over the 12 {111}<110> slip systems for uniaxial stress along `axis` (sample frame)"""
    n = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)
    dirs = np.array(
        [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]], float
    )
    plane, d = ([], [])
    for nn in n:
        for dd in dirs:
            if abs(nn @ dd) < 1e-09:
                plane.append(nn / np.sqrt(3))
                d.append(dd)
    plane, d = (np.array(plane), np.array(d))
    assert len(d) == 12
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    e = np.radians(np.atleast_2d(euler_deg))
    out = np.zeros(len(e))
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    for k, (p1, P, p2) in enumerate(e):
        c1, s1, c, s, c2, s2 = (
            np.cos(p1),
            np.sin(p1),
            np.cos(P),
            np.sin(P),
            np.cos(p2),
            np.sin(p2),
        )
        g = np.array(
            [
                [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
                [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
                [s1 * s, -c1 * s, c],
            ]
        )
        a = g @ ax
        out[k] = np.max(np.abs(plane @ a * (d @ a)))
    return out


def fcc_inplane_schmid(euler_deg, flip_y=False):
    """in-plane part (xx, yy, xy) of the symmetric Schmid tensors sym(s x m) of the 12 {111}<110> slip systems in the
    sample frame (x right, y down the map rows, z into the sample; Bunge g: sample -> crystal) -> (n, 12, 3).  Plane stress:
    the resolved shear stress of system a is sigma : P_a with the in-plane components only, and its in-plane plastic strain
    increment is dgamma_a P_a (the out-of-plane components do not enter the in-plane equilibrium).  flip_y: the other
    common sample-frame convention (y up, z out of the sample = 180 deg about x), which flips the sign of P_xy.
    For tension along x, max_a |P_a,xx| is the Schmid factor of schmid_fcc(euler, (1, 0, 0)).
    """
    n = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)
    dirs = np.array(
        [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]], float
    )
    plane, d = ([], [])
    for nn in n:
        for dd in dirs:
            if abs(nn @ dd) < 1e-09:
                plane.append(nn / np.sqrt(3))
                d.append(dd / np.sqrt(2))
    plane, d = (np.array(plane), np.array(d))
    e = np.radians(np.atleast_2d(euler_deg))
    out = np.zeros((len(e), 12, 3))
    for k, (p1, P, p2) in enumerate(e):
        c1, s1, c, s, c2, s2 = (
            np.cos(p1),
            np.sin(p1),
            np.cos(P),
            np.sin(P),
            np.cos(p2),
            np.sin(p2),
        )
        g = np.array(
            [
                [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
                [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
                [s1 * s, -c1 * s, c],
            ]
        )
        ss, mm = (d @ g, plane @ g)
        out[k, :, 0] = ss[:, 0] * mm[:, 0]
        out[k, :, 1] = ss[:, 1] * mm[:, 1]
        out[k, :, 2] = 0.5 * (ss[:, 0] * mm[:, 1] + ss[:, 1] * mm[:, 0])
    if flip_y:
        out[:, :, 2] *= -1.0
    return out


def fcc_systems_sample(euler_deg):
    """slip directions s and plane normals m (unit) of the 12 {111}<110> systems in the sample frame -> (n, 12, 3) x 2"""
    nn = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)
    dirs = np.array(
        [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]], float
    )
    plane, d = ([], [])
    for n_ in nn:
        for dd in dirs:
            if abs(n_ @ dd) < 1e-09:
                plane.append(n_ / np.sqrt(3))
                d.append(dd / np.sqrt(2))
    plane, d = (np.array(plane), np.array(d))
    e = np.radians(np.atleast_2d(euler_deg))
    S, M = (np.zeros((len(e), 12, 3)), np.zeros((len(e), 12, 3)))
    for k, (p1, P, p2) in enumerate(e):
        c1, s1, c, s_, c2, s2 = (
            np.cos(p1),
            np.sin(p1),
            np.cos(P),
            np.sin(P),
            np.cos(p2),
            np.sin(p2),
        )
        g = np.array(
            [
                [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s_],
                [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s_],
                [s1 * s_, -c1 * s_, c],
            ]
        )
        S[k], M[k] = (d @ g, plane @ g)
    return (S, M)


def mprime_max(euler_a, euler_b):
    """Luster-Morris slip-transmission factor m' = |n_a.n_b| |s_a.s_b|, maximised over the 12 x 12 system pairs of two FCC
    grains (1 = perfectly aligned systems, easy transmission; small = blocked)"""
    Sa, Ma = fcc_systems_sample(euler_a)
    Sb, Mb = fcc_systems_sample(euler_b)
    mp = np.abs(np.einsum("nai,nbi->nab", Ma, Mb)) * np.abs(
        np.einsum("nai,nbi->nab", Sa, Sb)
    )
    return mp.reshape(len(mp), -1).max(1)


def merge_thin(G, rmin):
    """merge grains whose maximum inscribed radius (pixels) is below rmin -- lamellae (e.g. thin annealing twins) narrower
    than ~2 rmin pixels cannot hold a 2D bond star -- into the neighbour sharing the longest boundary; relabel 0..n-1
    """
    G = np.unique(G, return_inverse=True)[1].reshape(G.shape)
    for _ in range(50):
        ng = int(G.max()) + 1
        rmax = np.asarray(
            ndimage.maximum(_inside_distance(G), labels=G, index=np.arange(ng))
        )
        flagged = rmax < rmin
        if not flagged.any():
            break
        G, changed = _merge_into_neighbours(G, flagged)
        G = np.unique(G, return_inverse=True)[1].reshape(G.shape)
        if not changed:
            break
    return G
