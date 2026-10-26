"""Grain-conforming mesh of the free interior of an affine Dirichlet band.

The fully prescribed outer band is analytically eliminated. Its inner square/
cube boundary is fitted exactly, removing motion of the imposed/free interface
as a side effect of unstructured spatial refinement.
"""

import numpy as np
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
from .finite_elements import Mesh, _clip


def mesh(seeds, size, band, level=0):
    seeds = np.asarray(seeds)
    dim = seeds.shape[1]
    points = []
    lookup = {}
    cells = []
    grains = []

    def add(x):
        key = tuple(np.round(x / size, 11))
        if key not in lookup:
            lookup[key] = len(points)
            points.append(np.asarray(x))
        return lookup[key]

    for g, s in enumerate(seeds):
        others = np.delete(seeds, g, axis=0)
        a = others - s
        b = (np.sum(others**2, axis=1) - np.sum(s * s)) / 2
        if dim == 2:
            lo = band
            hi = size - band
            poly = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])
            for aa, bb in zip(a, b):
                poly = _clip(poly, aa, bb)
                if len(poly) < 3:
                    break
            if len(poly) < 3:
                continue
            center = add(poly.mean(0))
            ids = [add(x) for x in poly]
            for i, j in zip(ids, ids[1:] + ids[:1]):
                cells.append([center, i, j])
                grains.append(g)
        else:
            aa = np.vstack([a, -np.eye(3), np.eye(3)])
            bb = np.r_[b, np.full(3, -band), np.full(3, size - band)]
            norms = np.linalg.norm(aa, axis=1)
            result = linprog(
                [0, 0, 0, -1],
                A_ub=np.column_stack([aa, norms]),
                b_ub=bb,
                bounds=[(None, None)] * 3 + [(0, None)],
                method="highs",
                options={"threads": 1},
            )
            if not result.success or result.x[3] < 1e-10:
                continue
            centerpoint = result.x[:3]
            verts = HalfspaceIntersection(
                np.column_stack([aa, -bb]), centerpoint
            ).intersections
            _, keep = np.unique(np.round(verts / size, 11), axis=0, return_index=True)
            verts = verts[np.sort(keep)]
            center = add(centerpoint)
            for normal, offset in zip(aa, bb):
                face = verts[
                    np.abs(verts @ normal - offset)
                    < 1e-08 * size * max(np.linalg.norm(normal), 1.0)
                ]
                if len(face) < 3:
                    continue
                fc = face.mean(0)
                fci = add(fc)
                nn = normal / np.linalg.norm(normal)
                e1 = face[0] - fc
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(nn, e1)
                ids = [
                    add(v)
                    for v in face[
                        np.argsort(np.arctan2((face - fc) @ e2, (face - fc) @ e1))
                    ]
                ]
                for i, j in zip(ids, ids[1:] + ids[:1]):
                    cells.append([center, fci, i, j])
                    grains.append(g)
    m = Mesh(np.asarray(points), np.asarray(cells, int), np.asarray(grains, int))
    return m.refine(level)


def verify(m, size, band):
    import math

    dim = m.p.shape[1]
    X = m.p[m.t]
    affine = np.concatenate([np.ones((len(X), dim + 1, 1)), X], axis=2)
    volume = np.abs(np.linalg.det(affine)) / math.factorial(dim)
    error = abs(volume.sum() - (size - 2 * band) ** dim) / (size - 2 * band) ** dim
    assert error < 1e-10 and np.all(volume > 1e-16)
    faces = np.concatenate(
        [np.sort(np.delete(m.t, k, axis=1), axis=1) for k in range(dim + 1)]
    )
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    assert np.all((counts == 1) | (counts == 2))
    exterior = m.p[unique[counts == 1]]
    fits = np.any(
        np.all(np.abs(exterior - band) < 1e-08, axis=1)
        | np.all(np.abs(exterior - (size - band)) < 1e-08, axis=1),
        axis=1,
    )
    assert fits.all(), "Nonconforming internal face or unfitted Dirichlet boundary"
    return {
        "relative_volume_error": float(error),
        "boundary_facets": int(len(exterior)),
        "internal_facets": int(np.sum(counts == 2)),
    }
