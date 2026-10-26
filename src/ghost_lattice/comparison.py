"""Compare freshly generated particle fields with independent FEM fields."""

import json
from pathlib import Path
import numpy as np


def compare(
    particle_file,
    fem_file,
    quantity="alpha",
    load=None,
    reverse=False,
    output=None,
    chunk_size=2048,
):
    from .numerics.point_location import locate
    from .numerics.sample_history import sample
    from .numerics.sample_kinematics import sample as kinematics

    with np.load(particle_file, allow_pickle=False) as field:
        mask = field["win"] if "win" in field else np.ones(len(field["X"]), bool)
        points = field["X"][mask]
        grain = field["grain"][mask]
        actual = field[quantity][mask]
    if not len(points):
        raise ValueError("The particle observation window is empty")
    reference = np.empty_like(actual)
    with np.load(fem_file, allow_pickle=False) as fe:
        if (
            quantity == "alpha"
            and "case_law" in fe
            and str(fe["case_law"])
            in ("static", "hallpetch", "slip_all", "slip_sigma3")
        ):
            raise ValueError(
                "Use ges for measured boundary-resistance laws; their history has spatial/evolving resistance"
            )
        points = points[:, : fe["p"].shape[1]]
        checkpoint_load = float(fe["loads"][-1]) if load is None else load
        if load is not None and not np.isclose(load, float(fe["loads"][-1])):
            raise ValueError("Select the FEM snapshot at the requested checkpoint load")
        for start in range(0, len(points), chunk_size):
            part = slice(start, min(start + chunk_size, len(points)))
            q = points[part]
            if "p2cells" in fe:
                value = (
                    kinematics(fe, q)
                    if quantity in ("ges", "strain")
                    else sample(fe, q, load=checkpoint_load, reverse=reverse)
                )
                ids = value["grain"]
                reference[part] = value[quantity]
            else:
                element = locate(fe, q)
                ids = fe["grain"][element]
                reference[part] = fe[quantity][element]
            if len(np.unique(fe["grain"])) > 1 and not np.array_equal(ids, grain[part]):
                raise ValueError("Particle/FEM grain labels differ at the query points")
    denominator = float(np.abs(reference).sum())
    if denominator == 0:
        raise ValueError("Relative L1 error is undefined for a zero reference field")
    result = dict(
        particle_file=str(particle_file),
        fem_file=str(fem_file),
        quantity=quantity,
        load=checkpoint_load,
        points=len(points),
        relative_L1=float(np.abs(reference - actual).sum() / denominator),
        particle_max=float(actual.max()),
        FEM_max=float(reference.max()),
        FEM_convergence="Compare separate reference mesh levels before claiming mesh convergence",
    )
    if quantity == "strain":
        result["relative_L2"] = float(
            np.linalg.norm(reference - actual) / np.linalg.norm(reference)
        )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
