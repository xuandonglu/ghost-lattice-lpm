"""Evaluate the local return diagnostic on the adopted representative geometry."""

import argparse
import json
from pathlib import Path
import numpy as np
from ghost_lattice.numerics import plane_lpm
from ghost_lattice.numerics.return_diagnostic import kappa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/return_consistency.npz")
    )
    args = parser.parse_args()
    seeds, theta, shift = plane_lpm.microstructure(64, 16, 1)
    model = plane_lpm.Poly2D(64, seeds, theta, shift, kind="j04", band=6, ret="scr")
    ratio, continuum_decrement = kappa(model, plane_lpm.Cmat()[:, 0])
    threshold = (3 * plane_lpm.GSH + 0.02) / continuum_decrement
    boundary = model.inner & (model.dgb < 1.5)
    info = {
        "scope": "Local directional diagnostic on the adopted geometry; no equilibrium history was solved",
        "N": model.N,
        "boundary_particles": int(boundary.sum()),
        "threshold_for_negative_return_factor": float(threshold),
        "boundary_fraction_above_threshold": float(
            np.mean(ratio[boundary] > threshold)
        ),
        "boundary_maximum_ratio": float(ratio[boundary].max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=model.X,
        grain=model.grain,
        dgb=model.dgb,
        boundary=boundary,
        kappa=ratio,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
