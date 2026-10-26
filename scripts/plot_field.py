"""Plot raw saved particle values; plotting does not change field-error metrics."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantity", choices=["alpha", "ges", "exx"], default="alpha")
    parser.add_argument(
        "--coordinate-scale",
        type=float,
        default=1.0,
        help="Optional display-only coordinate scaling; generated output uses 1",
    )
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as z:
        X = z["X"] * args.coordinate_scale
        mask = z["win"] if "win" in z else np.ones(len(X), bool)
        quantity = args.quantity
        if quantity == "exx":
            value = z["strain"][:, 0]
        elif quantity == "ges" and "ges" not in z:
            strain = z["strain"]
            value = np.hypot((strain[:, 0] - strain[:, 1]) / 2, strain[:, 2] / 2)
        else:
            value = z[quantity]
    plt.rcParams.update(
        {"font.family": "DejaVu Serif", "font.size": 10, "axes.linewidth": 0.65}
    )
    if X.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        artist = ax.scatter(
            *X[mask].T,
            c=100 * value[mask],
            s=1,
            cmap="cividis",
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set(xlabel="x", ylabel="y")
    else:
        fig = plt.figure(figsize=(4, 3.5))
        ax = fig.add_subplot(projection="3d")
        artist = ax.scatter(
            *X[mask].T,
            c=100 * value[mask],
            s=1,
            cmap="cividis",
            linewidths=0,
            rasterized=True,
        )
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.set_box_aspect(np.ptp(X[mask], axis=0))
    labels = {
        "alpha": "Accumulated plasticity / slip (%)",
        "ges": "Effective shear strain (%)",
        "exx": "Axial strain (%)",
    }
    fig.colorbar(artist, ax=ax, label=labels[quantity], fraction=0.04, pad=0.04)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
