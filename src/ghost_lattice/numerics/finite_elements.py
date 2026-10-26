"""Small-strain Galerkin FEM, independent of the LPM force/connection operators.

P1 triangles/tetrahedra; exact constant-strain integration. Engineering shear
strains and physical shear stresses are work-conjugate. Plastic material states
are stored at element integration points. Local implicit updates supply their
exact algorithmic tangents; global equilibrium uses Newton and line search.
"""

from dataclasses import dataclass
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import HalfspaceIntersection


@dataclass
class Mesh:
    p: np.ndarray
    t: np.ndarray
    grain: np.ndarray

    def refine(self, times=1):
        m = self
        for _ in range(times):
            p, t = (m.p, m.t)
            pairs = [
                (i, j) for i in range(t.shape[1]) for j in range(i + 1, t.shape[1])
            ]
            edges = np.concatenate([np.sort(t[:, [i, j]], axis=1) for i, j in pairs])
            unique, inverse = np.unique(edges, axis=0, return_inverse=True)
            mids = (len(p) + inverse).reshape(len(pairs), len(t)).T
            newp = np.vstack([p, p[unique].mean(axis=1)])
            if t.shape[1] == 3:
                a, b, c = mids.T
                newt = np.concatenate(
                    [
                        np.column_stack([t[:, 0], a, b]),
                        np.column_stack([a, t[:, 1], c]),
                        np.column_stack([b, c, t[:, 2]]),
                        np.column_stack([a, c, b]),
                    ]
                )
                ng = np.tile(m.grain, 4)
            else:
                a, b, c, d, e, f = mids.T
                newt = np.concatenate(
                    [
                        np.column_stack([t[:, 0], a, b, c]),
                        np.column_stack([a, t[:, 1], d, e]),
                        np.column_stack([b, d, t[:, 2], f]),
                        np.column_stack([c, e, f, t[:, 3]]),
                        np.column_stack([a, f, b, c]),
                        np.column_stack([a, f, c, e]),
                        np.column_stack([a, f, e, d]),
                        np.column_stack([a, f, d, b]),
                    ]
                )
                ng = np.tile(m.grain, 8)
            m = Mesh(newp, newt, ng)
        return m


def tri_box(n=8, size=1.0, labels=None):
    xx, yy = np.meshgrid(np.linspace(0, size, n + 1), np.linspace(0, size, n + 1))
    p = np.column_stack([xx.ravel(), yy.ravel()])
    v = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = (
        v[:-1, :-1].ravel(),
        v[:-1, 1:].ravel(),
        v[1:, 1:].ravel(),
        v[1:, :-1].ravel(),
    )
    t = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    g = (
        np.zeros(len(t), int)
        if labels is None
        else np.tile(np.asarray(labels).ravel(), 2)
    )
    return Mesh(p, t, g)


def _clip(poly, a, b, tol=1e-11):
    out = []
    for x, y in zip(poly, np.roll(poly, -1, axis=0)):
        fx, fy = (x @ a - b, y @ a - b)
        if fx <= tol:
            out.append(x)
        if fx < -tol and fy > tol or (fx > tol and fy < -tol):
            out.append(x + (y - x) * fx / (fx - fy))
    return np.asarray(out)


def voronoi_mesh(seeds, size, refine=0):
    """Grain-conforming mesh; shared grain faces use exactly shared vertices."""
    seeds = np.asarray(seeds)
    dim = seeds.shape[1]
    points = []
    lookup = {}
    cells = []
    grain = []

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
            poly = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]])
            for aa, bb in zip(a, b):
                poly = _clip(poly, aa, bb)
                if len(poly) == 0:
                    raise ValueError("Empty Voronoi cell")
            center = add(poly.mean(axis=0))
            ids = [add(v) for v in poly]
            for i, j in zip(ids, ids[1:] + ids[:1]):
                cells.append([center, i, j])
                grain.append(g)
        else:
            aa = np.vstack([a, -np.eye(3), np.eye(3)])
            bb = np.r_[b, np.zeros(3), np.full(3, size)]
            verts = HalfspaceIntersection(np.column_stack([aa, -bb]), s).intersections
            _, keep = np.unique(np.round(verts / size, 11), axis=0, return_index=True)
            verts = verts[np.sort(keep)]
            center = add(s)
            for normal, offset in zip(aa, bb):
                mask = np.abs(verts @ normal - offset) < 1e-08 * size * max(
                    np.linalg.norm(normal), 1.0
                )
                face = verts[mask]
                if len(face) < 3:
                    continue
                fc = face.mean(axis=0)
                fci = add(fc)
                nn = normal / np.linalg.norm(normal)
                e1 = face[0] - fc
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(nn, e1)
                ang = np.arctan2((face - fc) @ e2, (face - fc) @ e1)
                ids = [add(v) for v in face[np.argsort(ang)]]
                for i, j in zip(ids, ids[1:] + ids[:1]):
                    cells.append([center, fci, i, j])
                    grain.append(g)
    m = Mesh(np.asarray(points), np.asarray(cells, int), np.asarray(grain, int))
    return m.refine(refine)


def elastic(E=1.0, nu=0.3, dim=2):
    if dim == 2:
        return (
            E
            / (1 - nu * nu)
            * np.array([[1.0, nu, 0], [nu, 1.0, 0], [0, 0, (1 - nu) / 2]])
        )
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    c = np.diag([2 * mu] * 3 + [mu] * 3)
    c[:3, :3] += lam
    return c


def rotate_plane(C, theta):
    """Rotate plane-stress energy from crystal axes to sample axes."""
    co, si = (np.cos(theta), np.sin(theta))
    R = np.array([[co, -si], [si, co]])
    T = np.empty((3, 3))
    for j in range(3):
        e = np.zeros((2, 2))
        if j < 2:
            e[j, j] = 1.0
        else:
            e[0, 1] = e[1, 0] = 0.5
        q = R.T @ e @ R
        T[:, j] = [q[0, 0], q[1, 1], 2 * q[0, 1]]
    return T.T @ C @ T


def _batch_c(C, n):
    return np.broadcast_to(np.asarray(C), (n,) + np.asarray(C).shape[-2:])


def j2_update(strain, ep, alpha, C, sy, H, tangent=True):
    """Implicit associated J2 return, plane stress (3) or 3D (6).

    Solves stress and plastic multiplier simultaneously. Plane-stress plastic
    flow is the reduced 3D associated flow, with sigma_zz identically zero.
    """
    n, m = strain.shape
    C = _batch_c(C, n)
    tr = np.einsum("nij,nj->ni", C, strain - ep)
    A = np.diag([1.0, 1.0, 3.0]) if m == 3 else np.diag([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
    A[:2, :2] += np.array([[0.0, -0.5], [-0.5, 0.0]])
    if m == 6:
        A[0, 2] = A[2, 0] = A[1, 2] = A[2, 1] = -0.5
    q = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", tr, A, tr), 0.0))
    Y = np.broadcast_to(sy, (n,)) + H * alpha
    active = q > Y + 1e-13 * np.maximum(np.abs(Y), 1e-12)
    stress = tr.copy()
    dep = np.zeros_like(ep)
    da = np.zeros(n)
    ct = C.copy() if tangent else None
    if not active.any():
        return (stress, ep.copy(), alpha.copy(), ct)
    ids = np.flatnonzero(active)
    cc = C[ids]
    s = tr[ids].copy()
    tr0 = s.copy()
    y = Y[ids]
    dl = np.zeros(len(ids))
    eye = np.eye(m)[None]
    for it in range(45):
        aq = s @ A
        qk = np.sqrt(np.maximum(np.einsum("ni,ni->n", s, aq), 1e-50))
        flow = aq / qk[:, None]
        hess = (
            A[None] / qk[:, None, None]
            - flow[:, :, None] * flow[:, None, :] / qk[:, None, None]
        )
        cf = np.einsum("nij,nj->ni", cc, flow)
        r = np.column_stack([s - tr0 + dl[:, None] * cf, qk - y - H * dl])
        J = np.zeros((len(ids), m + 1, m + 1))
        J[:, :m, :m] = eye + dl[:, None, None] * np.einsum("nij,njk->nik", cc, hess)
        J[:, :m, m] = cf
        J[:, m, :m] = flow
        J[:, m, m] = -H
        err = np.max(np.abs(r) / np.maximum(y[:, None], 1e-12))
        if err < 2e-11:
            break
        dx = np.linalg.solve(J, -r[..., None])[..., 0]
        step = np.ones(len(ids))
        bad = dx[:, -1] < -dl
        step[bad] = np.minimum(1.0, 0.95 * dl[bad] / np.maximum(-dx[bad, -1], 1e-30))
        s += step[:, None] * dx[:, :m]
        dl += step * dx[:, m]
    else:
        raise RuntimeError(f"J2 local Newton failed: {err}")
    stress[ids] = s
    dep[ids] = dl[:, None] * flow
    da[ids] = dl
    if tangent:
        rhs = np.zeros((len(ids), m + 1, m))
        rhs[:, :m] = cc
        ct[ids] = np.linalg.solve(J, rhs)[:, :m]
    return (stress, ep + dep, alpha + da, ct)


def cp_update(
    strain, ep, alpha, C, P, g0, H, c, exponent=10.0, extra=0.0, tangent=True
):
    """Implicit power-law slip solved in stress/resistance coordinates.

    P contains engineering plastic-strain increments per unit slip; tau=P*sigma.
    This is a four/seven-unknown material solve independent of slip-system count.
    """
    n, m = strain.shape
    C = _batch_c(C, n)
    P = np.broadcast_to(P, (n,) + P.shape[-2:])
    tr = np.einsum("nij,nj->ni", C, strain - ep)
    old = np.broadcast_to(g0, (n,)) + H * alpha + np.broadcast_to(extra, (n,))
    t0 = np.einsum("nai,ni->na", P, tr)
    active = np.max(np.abs(t0) / old[:, None], axis=1) > 0.2
    stress = tr.copy()
    ep1 = ep.copy()
    a1 = alpha.copy()
    ct = C.copy() if tangent else None
    if not active.any():
        return (stress, ep1, a1, ct)
    ids = np.flatnonzero(active)
    pp = P[ids]
    cc = C[ids]
    tr0 = tr[ids]
    s = tr0.copy()
    g = old[ids].copy()
    cp = np.einsum("nij,naj->nia", cc, pp)

    def equations(s, g):
        tau = np.einsum("nai,ni->na", pp, s)
        ratio = np.abs(tau) / g[:, None]
        gamma = c * np.sign(tau) * ratio**exponent
        q = c * exponent * ratio ** (exponent - 1) / g[:, None]
        r = np.column_stack(
            [
                s - tr0 + np.einsum("nia,na->ni", cp, gamma),
                g - old[ids] - H * np.abs(gamma).sum(1),
            ]
        )
        J = np.zeros((len(ids), m + 1, m + 1))
        J[:, :m, :m] = np.eye(m)[None] + np.einsum("nia,na,naj->nij", cp, q, pp)
        J[:, :m, m] = -exponent * np.einsum("nia,na->ni", cp, gamma) / g[:, None]
        J[:, m, :m] = -H * np.einsum("na,na,nai->ni", np.sign(gamma), q, pp)
        J[:, m, m] = 1 + H * exponent * np.abs(gamma).sum(1) / g
        return (r, J, gamma)

    for it in range(60):
        r, J, gamma = equations(s, g)
        norm = np.max(np.abs(r) / old[ids, None], axis=1)
        if norm.max() < 2e-11:
            break
        dx = np.linalg.solve(J, -r[..., None])[..., 0]
        fac = np.ones(len(ids))
        for ls in range(16):
            sn = s + fac[:, None] * dx[:, :m]
            gn = g + fac * dx[:, m]
            rr, _, _ = equations(sn, np.maximum(gn, old[ids] * 0.01))
            newnorm = np.max(np.abs(rr) / old[ids, None], axis=1)
            good = (gn > 0) & ((newnorm < norm) | (norm < 2e-11))
            if good.all():
                break
            fac[~good] *= 0.5
        s = sn
        g = gn
    else:
        raise RuntimeError(f"CP local Newton failed: {norm.max()}")
    stress[ids] = s
    ep1[ids] += np.einsum("na,nai->ni", gamma, pp)
    a1[ids] += np.abs(gamma).sum(1)
    if tangent:
        rhs = np.zeros((len(ids), m + 1, m))
        rhs[:, :m] = cc
        ct[ids] = np.linalg.solve(J, rhs)[:, :m]
    return (stress, ep1, a1, ct)


class FEM:

    def __init__(self, mesh, C, band=0.0, fixed=None, field_dim=None, thickness=1.0):
        self.mesh = mesh
        self.geometry_dim = mesh.p.shape[1]
        self.dim = field_dim or self.geometry_dim
        d = self.dim
        gd = self.geometry_dim
        self.nstrain = 3 if d == 2 else 6
        m = self.nstrain
        X = mesh.p[mesh.t]
        nt, nv, _ = X.shape
        affine = np.concatenate([np.ones((nt, nv, 1)), X], axis=2)
        inv = np.linalg.inv(affine)
        self.volume = np.abs(np.linalg.det(affine)) / math.factorial(gd) * thickness
        if np.any(self.volume < 1e-16):
            raise ValueError("Degenerate element")
        grad = inv[:, 1:, :].transpose(0, 2, 1)
        B = np.zeros((nt, m, nv * d))
        for a in range(nv):
            for k in range(gd):
                B[:, k, a * d + k] = grad[:, a, k]
            pairs = [(0, 1)] if d == 2 else [(1, 2), (0, 2), (0, 1)]
            for k, (i, j) in enumerate(pairs, start=d):
                if j < gd:
                    B[:, k, a * d + i] = grad[:, a, j]
                if i < gd:
                    B[:, k, a * d + j] = grad[:, a, i]
        dofs = (mesh.t[:, :, None] * d + np.arange(d)).reshape(nt, nv * d)
        rows = np.broadcast_to(
            np.arange(nt)[:, None, None] * m + np.arange(m)[None, :, None], B.shape
        )
        cols = np.broadcast_to(dofs[:, None, :], B.shape)
        self.B = sp.coo_matrix(
            (B.ravel(), (rows.ravel(), cols.ravel())), shape=(nt * m, len(mesh.p) * d)
        ).tocsr()
        self.center = X.mean(axis=1)
        self.C = _batch_c(C, nt)
        hi = mesh.p.max(axis=0)
        lo = mesh.p.min(axis=0)
        if fixed is None:
            fixed = np.any(
                (mesh.p <= lo + band + 1e-10) | (mesh.p >= hi - band - 1e-10), axis=1
            )
        self.fixed_nodes = np.asarray(fixed, bool)
        self.fixed = np.flatnonzero(np.repeat(self.fixed_nodes, d))
        self.free = np.flatnonzero(~np.repeat(self.fixed_nodes, d))
        self.u = np.zeros(len(mesh.p) * d)
        self.ep = np.zeros((nt, m))
        self.alpha = np.zeros(nt)
        self.stress = np.zeros((nt, m))
        self.strain = np.zeros((nt, m))
        self.lam = 0.0
        self._rows = np.repeat(np.arange(nt * m), m)
        self._cols = np.repeat(np.arange(nt), m * m) * m + np.tile(np.arange(m), nt * m)
        self.Kelastic = self.stiffness(self.C)
        self._elastic_lu = sla.splu(self.Kelastic[self.free][:, self.free])
        self._elastic_fc = self.Kelastic[self.free][:, self.fixed]

    def stiffness(self, C):
        weighted = C * self.volume[:, None, None]
        D = sp.csr_matrix(
            (weighted.ravel(), (self._rows, self._cols)),
            shape=(len(C) * self.nstrain,) * 2,
        )
        return (self.B.T @ D @ self.B).tocsc()

    def force(self, stress):
        return self.B.T @ (stress * self.volume[:, None]).ravel()

    def step(self, E, lam, material, tol=1e-09, maxiter=35):
        points = np.pad(self.mesh.p, ((0, 0), (0, self.dim - self.geometry_dim)))
        prescribed = (points @ E.T).ravel() * lam
        u = self.u.copy()
        delta = prescribed[self.fixed] - u[self.fixed]
        u[self.free] += self._elastic_lu.solve(-(self._elastic_fc @ delta))
        u[self.fixed] = prescribed[self.fixed]

        def state(v, tangent=True):
            strain = (self.B @ v).reshape(-1, self.nstrain)
            return (strain, material(strain, self.ep, self.alpha, tangent))

        for it in range(maxiter):
            strain, (stress, ep, alpha, Ct) = state(u)
            f = self.force(stress)
            scale = max(np.linalg.norm(f[self.fixed]), 1e-15)
            rel = np.linalg.norm(f[self.free]) / scale
            if rel < tol:
                break
            K = self.stiffness(Ct)
            du = sla.spsolve(K[self.free][:, self.free], -f[self.free])
            old = np.linalg.norm(f[self.free])
            fac = 1.0
            for ls in range(14):
                trial = u.copy()
                trial[self.free] += fac * du
                _, ss = state(trial, False)
                fn = self.force(ss[0])
                if np.linalg.norm(fn[self.free]) <= old * (1 - 0.0001 * fac):
                    break
                fac *= 0.5
            u = trial
        else:
            raise RuntimeError(f"Global FEM Newton failed at {lam}: residual {rel}")
        self.u = u
        self.strain = strain
        self.stress = stress
        self.ep = ep
        self.alpha = alpha
        self.lam = lam
        return {"iterations": it, "relative_residual": float(rel)}

    def save(self, path, **extra):
        np.savez_compressed(
            path,
            p=self.mesh.p,
            t=self.mesh.t,
            grain=self.mesh.grain,
            X=self.center,
            volume=self.volume,
            u=self.u,
            strain=self.strain,
            stress=self.stress,
            epsp=self.ep,
            alpha=self.alpha,
            **extra,
        )
