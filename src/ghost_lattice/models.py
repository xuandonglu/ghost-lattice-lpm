"""Assemble the published physical models through a portable public interface."""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
from .cases import Case


@dataclass
class BuiltModel:
    model: object
    load_tensor: np.ndarray
    strain_scale: float
    length_per_lattice_unit: float
    update_resistance: object
    diagnostics: dict


def repository_root():
    return Path(__file__).resolve().parents[2]


def _cp_projections(theta):
    angle = theta[:, None] + np.radians([0.0, 60.0, 120.0])[None]
    sx, sy = np.cos(angle), np.sin(angle)
    mx, my = -sy, sx
    return np.stack([sx * mx, sy * my, 0.5 * (sx * my + sy * mx)], axis=2)


def build(case: Case, method="proposed", data_root=None, assembly=None):
    """Build one case. No simulation, queue or file write occurs on import.

    ``assembly`` is available only for the matched assembly-control case.
    All dimensions in Case are physical dimensions or stated model-length units.
    """
    from .numerics import plane_lpm as plane
    from .numerics.rotated_plane import CompactRotated
    from .numerics.sparse_assembly import MeasuredCP, Equiaxed

    if method not in ("proposed", "merged"):
        raise ValueError("method must be proposed or merged")
    if assembly is not None and case.name != "assembly_control":
        raise ValueError("Assembly variants belong to assembly_control only")
    kind = "ghost" if method == "proposed" else "j04"
    spacing = case.spacing
    update = lambda: None
    diagnostics = {"case": case.as_dict(), "method": method}
    if case.geometry in ("columnar", "equiaxed"):
        from .numerics import columnar_lpm as columnar
        from .numerics import equiaxed_lpm as equiaxed

        material = columnar.gc.material("iso")
        mu, la = material["C44"], material["C12"]
        young = mu * (3 * la + 2 * mu) / (la + mu)
        strength, hardening = 0.0015 * young, 0.02 * young
        c = columnar.gc.cubic_C(material["C11"], material["C12"], material["C44"])[:, 0]
        ey = strength / np.sqrt(
            1.5 * np.sum((c[:3] - c[:3].mean()) ** 2) + 3 * np.sum(c[3:] ** 2)
        )
        generator = (
            equiaxed.microstructure
            if case.geometry == "equiaxed"
            else columnar.microstructure
        )
        seeds, frames, shifts = generator(case.size, case.grains, case.seed)
        factor = 1 + case.yield_contrast * (
            2 * np.random.default_rng(case.seed + 1000).uniform(0, 1, case.grains) - 1
        )
        cls = Equiaxed if case.geometry == "equiaxed" else columnar.Plast3D
        if case.geometry == "columnar":
            columnar.LZ = 3.0 / spacing
        model = cls(
            kind,
            case.size / spacing,
            seeds / spacing,
            frames,
            shifts,
            mat=material,
            band=case.band / spacing,
            sy=strength * factor,
            H=hardening,
            ret="scr",
            coupling="energy" if method == "proposed" else "force",
            plast=case.law,
            cp=dict(g0=0.5 * strength, n=10.0, c=case.increment * ey, frames=frames),
        )
        if method == "proposed":
            model.enable_gfc()
        return BuiltModel(
            model, np.diag([1.0, 0.0, 0.0]), ey, spacing, update, diagnostics
        )

    if case.geometry == "measured":
        from .numerics.raster_geometry import Raster
        from .numerics import orientations

        folder = Path(data_root or "outputs") / "inputs"
        prefix = "CG" if case.measured_map == "coarse" else "FG"
        with np.load(
            folder / f"hea_{prefix}_grains_W128t.npz", allow_pickle=False
        ) as z:
            grain_map = {k: z[k] for k in z.files}
        with np.load(folder / f"hea_{prefix}_prep.npz", allow_pickle=False) as z:
            prep = {k: z[k] for k in ("ori", "G", "step")}
        base_spacing = float(grain_map["a_um"])
        scale = base_spacing / spacing
        raster = Raster(grain_map["G"], float(grain_map["px"]) * scale)
        theta = np.radians(np.mod(grain_map["phi1"], 60.0))
        shift = np.random.default_rng(7).uniform(0, 1, (len(theta), 2))
        median = float(np.median(grain_map["sf"]))
        ey = 0.0015
        kwargs = dict(
            kind=kind, band=case.band / spacing, H=0.01, ret="scr", raster=raster
        )
        if case.law == "j2":
            model = CompactRotated(
                case.size / spacing,
                raster.centroids,
                theta,
                shift,
                sy=0.0015 * median / grain_map["sf"],
                **kwargs,
            )
        else:
            ori = prep["ori"][grain_map["grain_orig"]]
            model = MeasuredCP(
                case.size / spacing,
                raster.centroids,
                theta,
                shift,
                sy=0.0015,
                cp=dict(
                    g0=median * 0.0015,
                    n=10.0,
                    c=case.increment * ey,
                    P=orientations.fcc_inplane_schmid(ori),
                ),
                **kwargs,
            )
        model.raster = raster
        model.dgb = raster.dist_to_gb(model.X)
        distance = model.dgb * spacing
        if case.law in ("static", "hallpetch"):
            before = float(model.syp.mean())
            if case.law == "static":
                factor = 1 + 0.25 * np.exp(-distance)
            else:
                area = np.bincount(prep["G"].ravel()) * float(prep["step"]) ** 2
                diameter = (
                    (8.82 / 5.68) * 2 * np.sqrt(area[grain_map["grain_orig"]] / np.pi)
                )
                factor = ((174 + 371 / np.sqrt(diameter)) / 306)[model.grain]
            model.syp *= factor
            model.cp["g0"] *= float(model.syp.mean()) / before
        if case.law in ("slip_all", "slip_sigma3"):
            weight = np.exp(-distance)
            if case.law == "slip_sigma3":
                from .numerics.twin_boundaries import sigma3_dev

                quaternions = orientations.euler_to_quat(ori)
                other = raster.other_grain(model.X)
                valid = other != model.grain
                flags = np.zeros(model.N, bool)
                flags[valid] = sigma3_dev(
                    quaternions[model.grain[valid]], quaternions[other[valid]]
                ) <= 15 / np.sqrt(3)
                weight *= 1 - flags * np.exp(-distance)
            coefficient = 3 if case.law == "slip_all" else 2

            def update():
                model.gG = coefficient * model.H * weight * model.alpha

        if method == "proposed":
            model.enable_gfc()
        return BuiltModel(model, np.diag([1.0, -0.3]), ey, spacing, update, diagnostics)

    seeds, theta, shift = plane.microstructure(case.size, case.grains, case.seed)
    if case.law == "elastic":
        from .numerics import anisotropic_plane as anisotropic

        material = np.array([[1.5, 0.25, 0.0], [0.25, 0.9, 0.0], [0.0, 0.0, 0.35]])
        grain_material = np.array(
            [anisotropic.rotate_plane(material, angle) for angle in theta]
        )
        # The inherited elastic constructor uses its module's Poly2D factory.
        factory = anisotropic.Poly2D

        def scaled_factory(*args, **kwargs):
            kwargs["band"] = case.band / spacing
            return factory(*args, **kwargs)

        anisotropic.Poly2D = scaled_factory
        try:
            model = anisotropic.AnisotropicCoupling(
                case.size / spacing,
                theta,
                kind,
                grain_material,
                method == "proposed",
                seeds=seeds / spacing,
                shifts=shift,
            )
        finally:
            anisotropic.Poly2D = factory
        return BuiltModel(
            model, np.diag([0.001, 0.0]), 1.0, spacing, update, diagnostics
        )
    c = plane.Cmat()[:, 0]
    ey = 0.0015 / np.sqrt(c[0] ** 2 - c[0] * c[1] + c[1] ** 2)
    sy = 0.0015 * (
        1
        + case.yield_contrast
        * (2 * np.random.default_rng(case.seed + 1000).uniform(0, 1, case.grains) - 1)
    )
    kwargs = dict(kind=kind, band=case.band / spacing, sy=sy, H=0.02, ret="scr")
    if case.name == "hole":
        kwargs["hole"] = (32 / spacing, 32 / spacing, 8 / spacing)
    if case.law == "cp":
        model = MeasuredCP(
            case.size / spacing,
            seeds / spacing,
            theta,
            shift,
            cp=dict(
                g0=0.00075, n=10.0, c=case.increment * ey, P=_cp_projections(theta)
            ),
            **kwargs,
        )
    elif assembly is not None:
        model = plane.Poly2D(
            case.size,
            seeds,
            theta,
            shift,
            coupling="force" if assembly == "owner_force" else "energy",
            **kwargs,
        )
    else:
        model = CompactRotated(
            case.size / spacing, seeds / spacing, theta, shift, **kwargs
        )
    if method == "proposed" and assembly in (None, "energy_gfc"):
        model.enable_gfc()
    return BuiltModel(model, np.diag([1.0, 0.0]), ey, spacing, update, diagnostics)
