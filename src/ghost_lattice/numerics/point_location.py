import numpy as np
from scipy.spatial import cKDTree
from matplotlib.tri import Triangulation


def locate(fe, q):
    if fe["p"].shape[1] == 2:
        k = Triangulation(fe["p"][:, 0], fe["p"][:, 1], fe["t"]).get_trifinder()(
            q[:, 0], q[:, 1]
        )
        if (k < 0).any():
            raise ValueError("Particle lies outside the FEM mesh")
        return k
    vertices = fe["p"][fe["t"]]
    origin = vertices[:, 0]
    inverse = np.linalg.inv(vertices[:, 1:] - origin[:, None])
    tree = cKDTree(vertices.mean(1))
    result = np.full(len(q), -1, int)
    for start in range(0, len(q), 200):
        x = q[start : start + 200]
        ids = np.full(len(x), -1, int)
        for count in [32, 128, 512, 2048]:
            ii = np.flatnonzero(ids < 0)
            if not len(ii):
                break
            _, cand = tree.query(x[ii], k=min(count, len(vertices)))
            b = np.einsum("nki,nkij->nkj", x[ii, None] - origin[cand], inverse[cand])
            ok = (b.min(2) >= -1e-09) & (b.sum(2) <= 1 + 1e-09)
            found = ok.any(1)
            ids[ii[found]] = cand[np.flatnonzero(found), ok[found].argmax(1)]
        if (ids < 0).any():
            raise ValueError("Cannot locate 3D particle in actual FEM tetrahedra")
        result[start : start + len(x)] = ids
    return result
