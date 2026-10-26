"""Generate independent continuum FEM solutions from the public case definitions."""

import json
import time
from pathlib import Path
import numpy as np
from .cases import Case, get_case
from .numerics import finite_elements as core


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _hole_mesh(level):
    nt, nr = 64 * 2**level, 12 * 2**level
    angle = np.arange(nt) * 2 * np.pi / nt
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    outer = 26 / np.max(np.abs(direction), axis=1)
    radius = 8 + np.linspace(0, 1, nr + 1)[:, None] * (outer[None] - 8)
    p = (32 + radius[:, :, None] * direction[None]).reshape(-1, 2)
    a = (np.arange(nr)[:, None] * nt + np.arange(nt)[None]).ravel()
    b = (np.arange(nr)[:, None] * nt + (np.arange(nt)[None] + 1) % nt).ravel()
    t = np.concatenate(
        [np.stack([a, a + nt, b + nt], axis=1), np.stack([a, b + nt, b], axis=1)]
    )
    return core.Mesh(p, t, np.zeros(len(t), int))


def _setup(case, level, data_root):
    from .numerics.cached_quadratic_solver import FEM_P2_Cached
    from .numerics import fitted_mesh

    measured = case.geometry == "measured"
    spatial3 = case.geometry in ("columnar", "equiaxed")
    dim = 3 if spatial3 else 2
    if spatial3:
        from .numerics import columnar_lpm, equiaxed_lpm

        generator = (
            equiaxed_lpm.microstructure
            if case.geometry == "equiaxed"
            else columnar_lpm.microstructure
        )
        seeds, frames, _ = generator(case.size, case.grains, case.seed)
        young = 48 * (3 * 140 + 2 * 48) / (140 + 48)
        poisson = 140 / (2 * (140 + 48))
    else:
        from .numerics.plane_lpm import microstructure

        seeds, theta, _ = microstructure(case.size, case.grains, case.seed)
        young, poisson = 1.0, 0.3
    C = core.elastic(young, poisson, dim)
    strength = 0.0015 * young
    hardening = (0.01 if measured else 0.02) * young
    column = C[:, 0]
    if dim == 2:
        ey = strength / np.sqrt(column[0] ** 2 - column[0] * column[1] + column[1] ** 2)
    else:
        ey = strength / np.sqrt(
            1.5 * np.sum((column[:3] - column[:3].mean()) ** 2)
            + 3 * np.sum(column[3:] ** 2)
        )
    E = np.diag([1.0, 0.0, 0.0]) if dim == 3 else np.diag([1.0, 0.0])
    factor = 1 + case.yield_contrast * (
        2 * np.random.default_rng(case.seed + 1000).uniform(0, 1, case.grains) - 1
    )
    sy = strength * factor
    ng = case.grains
    if measured:
        from .numerics.raster_mesh import build
        from .numerics.raster_geometry import Raster
        from .numerics import orientations

        folder = Path(data_root or "outputs") / "inputs"
        prefix = "CG" if case.measured_map == "coarse" else "FG"
        with np.load(
            folder / f"hea_{prefix}_grains_W128t.npz", allow_pickle=False
        ) as z:
            data = {k: z[k] for k in z.files}
        with np.load(folder / f"hea_{prefix}_prep.npz", allow_pickle=False) as z:
            prep = {k: z[k] for k in ("G", "step", "ori")}
        # Physical micrometres here; the original reference used 0.5 um units.
        px = float(data["px"]) * float(data["a_um"])
        msh, geometry = build(data["G"], px, case.band, 0.25 / 4**level)
        model = FEM_P2_Cached(msh, C)
        raster = Raster(data["G"], px)
        ng = int(data["G"].max()) + 1
        median = float(np.median(data["sf"]))
        sy = strength * median / data["sf"]
        ey = strength
        E = np.diag([1.0, -0.3])
    elif case.geometry == "columnar":
        msh = core.voronoi_mesh(seeds, case.size, level)
        model = core.FEM(msh, C, band=case.band, field_dim=3, thickness=3)
        geometry = {"type": "grain_conforming_P1_generalized_plane_strain"}
    else:
        msh = (
            _hole_mesh(level)
            if case.name == "hole"
            else fitted_mesh.mesh(seeds, case.size, case.band, level)
        )
        if case.law == "elastic":
            base = np.array([[1.5, 0.25, 0.0], [0.25, 0.9, 0.0], [0.0, 0.0, 0.35]])
            Cg = np.array([core.rotate_plane(base, angle) for angle in theta])
            E, ey = np.diag([0.001, 0.0]), 1.0
        else:
            Cg = np.broadcast_to(
                C, (ng, dim * (dim + 1) // 2, dim * (dim + 1) // 2)
            ).copy()
        model = FEM_P2_Cached(msh, Cg[msh.grain], field_dim=dim)
        geometry = (
            {"type": "square_with_hole"}
            if case.name == "hole"
            else fitted_mesh.verify(msh, case.size, case.band)
        )
    grain = model.qgrain if hasattr(model, "qgrain") else msh.grain
    extra = np.zeros(len(grain))
    weight = np.zeros(len(grain))
    P = np.empty((ng, 0, 3 if dim == 2 else 6))
    g0 = np.full(len(grain), 0.5 * strength)
    if case.law not in ("j2", "elastic"):
        if measured:
            ori = prep["ori"][data["grain_orig"]]
            P = orientations.fcc_inplane_schmid(ori).copy()
            P[:, :, 2] *= 2
            distance = raster.dist_to_gb(model.center)
            g0[:] = median * strength
            if case.law == "static":
                g0 *= 1 + 0.25 * np.exp(-distance)
            if case.law == "hallpetch":
                area = np.bincount(prep["G"].ravel()) * float(prep["step"]) ** 2
                ratio = 8.82 / 5.68 if prefix == "CG" else 2.75 / 1.81
                diameter = ratio * 2 * np.sqrt(area[data["grain_orig"]] / np.pi)
                g0 *= ((174 + 371 / np.sqrt(diameter)) / 306)[grain]
            if case.law in ("slip_all", "slip_sigma3"):
                weight = np.exp(-distance)
                if case.law == "slip_sigma3":
                    from .numerics.twin_boundaries import sigma3_dev

                    q = orientations.euler_to_quat(ori)
                    other = raster.other_grain(model.center)
                    valid = other != grain
                    flags = np.zeros(len(grain), bool)
                    flags[valid] = sigma3_dev(
                        q[grain[valid]], q[other[valid]]
                    ) <= 15 / np.sqrt(3)
                    weight *= 1 - flags * np.exp(-distance)
        elif dim == 2:
            from .models import _cp_projections

            P = _cp_projections(theta).copy()
            P[:, :, 2] *= 2
        else:
            sv, mv = columnar_lpm.bcc_slip()
            projections = []
            for frame in frames:
                s, m = sv @ frame.T, mv @ frame.T
                tensor = 0.5 * (
                    s[:, :, None] * m[:, None, :] + m[:, :, None] * s[:, None, :]
                )
                projections.append(
                    np.array(
                        [
                            tensor[:, i, j] * (1 if i == j else 2)
                            for i, j in columnar_lpm.VO
                        ]
                    ).T
                )
            P = np.array(projections)

    def material(strain, ep, alpha, tangent=True):
        if case.law == "elastic":
            return (
                np.einsum("nij,nj->ni", model.C, strain),
                ep.copy(),
                alpha.copy(),
                model.C if tangent else None,
            )
        if case.law == "j2":
            return core.j2_update(strain, ep, alpha, C, sy[grain], hardening, tangent)
        return core.cp_update(
            strain,
            ep,
            alpha,
            C,
            P[grain],
            g0,
            hardening,
            case.increment * ey,
            extra=extra,
            tangent=tangent,
        )

    def update():
        if case.law in ("slip_all", "slip_sigma3"):
            extra[:] = (
                (3 if case.law == "slip_all" else 2) * hardening * weight * model.alpha
            )

    metadata = dict(
        material_C=(
            Cg
            if not measured and case.geometry != "columnar"
            else np.broadcast_to(C, (ng,) + C.shape)
        ),
        yield_by_grain=sy,
        slip_P=P,
        H=hardening,
        SY=strength,
        ey=ey,
        dt=case.increment,
        cp_g0=median * strength if measured else 0.5 * strength,
        constitutive=(
            "elastic" if case.law == "elastic" else "j2" if case.law == "j2" else "cp"
        ),
        coordinate_scale=1.0,
        case_law=case.law,
    )
    return model, E, ey, material, update, metadata, geometry


def run(name, output, level=0, data_root=None, seed=None):
    """Generate one mesh level and loading history; refinement is explicit."""
    case = name if isinstance(name, Case) else get_case(name, seed)
    if level < 0:
        raise ValueError("level must be nonnegative")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty FEM output directory")
    output.mkdir(parents=True, exist_ok=True)
    history, snapshots, displacements = [], [], []
    started = time.perf_counter()
    stage = "mesh_construction"
    try:
        model, E, ey, material, update, metadata, geometry = _setup(
            case, level, data_root
        )
        _write(
            output / "configuration.json",
            {
                "case": case.as_dict(),
                "level": level,
                "geometry": geometry,
                "mesh_convergence": "Not assessed by a single mesh-level run",
            },
        )
        loads = case.loads()
        for step, load in enumerate(loads, 1):
            stage = "equilibrium"
            info = model.step(E, float(load) * ey, material)
            if (
                not np.isfinite(info["relative_residual"])
                or info["relative_residual"] > 1.05e-9
            ):
                raise RuntimeError("FEM equilibrium acceptance failed")
            history.append({"step": step, "load": float(load), **info})
            displacements.append(model.u.copy())
            update()
            if any(abs(load - target) < 1e-10 for target in case.checkpoints):
                stage = "field_export"
                filename = f"step_{step:04d}.npz"
                model.save(
                    output / filename,
                    **metadata,
                    loads=loads[:step],
                    u_history=np.asarray(displacements),
                )
                snapshots.append({"step": step, "load": float(load), "path": filename})
            _write(
                output / "progress.json",
                {
                    "status": "running",
                    "accepted_steps": step,
                    "total_steps": len(loads),
                },
            )
    except Exception as error:
        _write(
            output / "result.json",
            {
                "status": "failed",
                "stage": stage,
                "error": str(error),
                "history": history,
                "snapshots": snapshots,
            },
        )
        raise
    result = {
        "status": "completed",
        "steps": len(history),
        "history": history,
        "snapshots": snapshots,
        "seconds_including_setup": time.perf_counter() - started,
        "mesh_convergence": "Not assessed by a single mesh-level run",
    }
    _write(output / "result.json", result)
    return result
