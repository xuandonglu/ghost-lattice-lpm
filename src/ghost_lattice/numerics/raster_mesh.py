"""Exact pixel-interface PSLG mesh; no smoothing of measured grain geometry."""

from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import label
from .finite_elements import Mesh
from .quadratic_elements import quadrature
from .fitted_mesh import verify


def build(G, px, band=4.0, maxarea=1.0, retain_band=False):
    import triangle

    G = np.asarray(G)
    W = G.shape[0] * px
    bandcut = int(round(band / px))
    if abs(bandcut * px - band) > 1e-09:
        raise ValueError("Dirichlet band must coincide with raster pixel lines")
    ncut = 0 if retain_band else bandcut
    gc = G[ncut : G.shape[0] - ncut, ncut : G.shape[1] - ncut]
    ny, nx = gc.shape
    vertices = []
    lookup = {}
    segments = []
    seen = set()

    def vertex(x, y):
        key = (int(x), int(y))
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(key)
        return lookup[key]

    def segment(a, b):
        ids = tuple(sorted([vertex(*a), vertex(*b)]))
        if ids not in seen:
            seen.add(ids)
            segments.append(ids)

    yy, xx = np.nonzero(gc[:, :-1] != gc[:, 1:])
    for y, x in zip(yy, xx):
        segment((x + 1, y), (x + 1, y + 1))
    yy, xx = np.nonzero(gc[:-1] != gc[1:])
    for y, x in zip(yy, xx):
        segment((x, y + 1), (x + 1, y + 1))
    for x in range(nx):
        segment((x, 0), (x + 1, 0))
        segment((x, ny), (x + 1, ny))
    for y in range(ny):
        segment((0, y), (0, y + 1))
        segment((nx, y), (nx, y + 1))
    if retain_band:
        for x in [bandcut, nx - bandcut]:
            for y in range(ny):
                segment((x, y), (x, y + 1))
        for y in [bandcut, ny - bandcut]:
            for x in range(nx):
                segment((x, y), (x + 1, y))
    regions = []
    yy, xx = np.mgrid[:ny, :nx]
    zone = np.zeros_like(gc)
    if retain_band:
        zone = (xx >= bandcut).astype(int) + (xx >= nx - bandcut).astype(int)
        zone += 3 * ((yy >= bandcut).astype(int) + (yy >= ny - bandcut).astype(int))
    for g in np.unique(gc):
        for region in np.unique(zone[gc == g]):
            components, n = label((gc == g) & (zone == region))
            for c in range(1, n + 1):
                pos = np.argwhere(components == c)[0]
                regions.append(
                    [
                        (pos[1] + ncut + 0.5) * px,
                        (pos[0] + ncut + 0.5) * px,
                        int(g) + 1,
                        -1,
                    ]
                )
    v = (np.asarray(vertices, float) + ncut) * px
    data = {
        "vertices": v,
        "segments": np.asarray(segments, np.int32),
        "regions": np.asarray(regions),
    }
    out = triangle.triangulate(data, f"pq20a{maxarea:.14g}AQ")
    attrs = np.rint(out["triangle_attributes"][:, 0]).astype(int) - 1
    if (attrs < 0).any():
        raise AssertionError("Unassigned measured grain region")
    m = Mesh(out["vertices"], out["triangles"], attrs)
    geometry = verify(m, W, 0.0 if retain_band else band)
    p = m.p[m.t]
    vol = 0.5 * np.abs(np.linalg.det(p[:, 1:] - p[:, 0, None]))
    expected = np.bincount(gc.ravel(), minlength=int(G.max()) + 1) * px**2
    actual = np.bincount(attrs, weights=vol, minlength=len(expected))
    rel = float(np.abs(actual - expected).sum() / expected.sum())
    assert rel < 1e-10, "Changed grain areas"
    q = np.einsum("qa,tad->tqd", quadrature()[0], p)
    ix = np.clip((q[:, :, 0] / px).astype(int), 0, G.shape[1] - 1)
    iy = np.clip((q[:, :, 1] / px).astype(int), 0, G.shape[0] - 1)
    assert np.all(
        G[iy, ix] == attrs[:, None]
    ), "Quadrature point crosses a measured interface"
    if retain_band:
        for axis in range(2):
            for edge in [band, W - band]:
                assert not np.any(
                    (p[:, :, axis].min(1) < edge - 1e-08)
                    & (p[:, :, axis].max(1) > edge + 1e-08)
                ), "Element crosses fitted affine-band edge"
    return (
        m,
        {
            **geometry,
            "grain_area_relative_L1_difference": rel,
            "nodes": len(m.p),
            "elements": len(m.t),
            "input_interface_segments": len(segments),
            "connected_regions": len(regions),
            "geometry": (
                "original pixel boundaries exactly retained; full domain and fitted affine-band edges"
                if retain_band
                else "original pixel boundaries exactly retained; exact inner affine-band boundary"
            ),
            "prescribed_outer_band_states_retained": retain_band,
        },
    )
