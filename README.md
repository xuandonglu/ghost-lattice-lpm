# Ghost-lattice coupling for polycrystal LPM

Code for *Consistent grain-boundary coupling for lattice particle models of
polycrystal plasticity*.

The implementation describes quasi-static, small-strain J2 and crystal
plasticity on independently oriented grain lattices. It provides ghost-lattice
coupling and particle-merging comparisons in two and three dimensions.

## What the code generates

- Seeded grain geometries, lattice orientations, particle positions and bond stars.
- Material assignments, prescribed loading histories and observation windows.
- Particle displacement, strain and plasticity fields at selected load checkpoints.
- Independent continuum FEM meshes, loading histories and reference fields.
- Native-point field comparisons, local return diagnostics and field plots.

Numerical inputs are defined in `configs/` and `src/ghost_lattice/cases.py`.
Simulation results, reference solutions and images are generated locally under
`outputs/`; they are excluded from version control. Measured grain geometries
come from external EBSD observations. Download and preprocessing scripts are
provided for those observations and the associated HR-DIC fields.

## Installation

Use Python 3.10 or newer in a virtual environment. From the repository root:

```bash
python -m pip install -e .
python -m ghost_lattice list
python -m unittest discover -s tests -v
```

Optional dependencies:

```bash
python -m pip install -e ".[experiment,mesh]"
# Optional GPU execution; install a PyTorch build appropriate for your CUDA runtime.
python -m pip install -e ".[cuda]"
```

## Generate a numerical dataset

The `demo` case is a small CPU example. The commands below generate particle
fields for both constructions and an independent quadratic FEM solution:

```bash
python -m ghost_lattice run demo --method proposed --output outputs/demo_proposed
python -m ghost_lattice run demo --method merged --output outputs/demo_merged
python -m ghost_lattice fem demo --level 1 --output outputs/demo_fem
python -m ghost_lattice compare outputs/demo_proposed/step_0030.npz outputs/demo_fem/step_0030.npz --quantity alpha --output outputs/demo_comparison.json
python scripts/plot_field.py outputs/demo_proposed/step_0030.npz --output outputs/demo_alpha.png --quantity alpha
```

Use `list` to inspect the available cases. Replace `demo` with the selected
case name to generate its configured geometry and loading history. A complete
Case JSON can be supplied with `--config configs/<case>.json` to either `run`
or `fem`. FEM refinement levels are selected explicitly; compare successive
levels to assess reference-field convergence. Completion of one level establishes
solver convergence at that level.

## Prepare measured microstructures

The external source is the CoCrFeNi EBSD/HR-DIC dataset associated with
*Grain size dependence of microscopic strain distribution in a high entropy
alloy at the onset of plastic deformation*,
[Zenodo 14532401](https://doi.org/10.5281/zenodo.14532401), licensed CC BY 4.0.
Source attribution is retained in [NOTICE.md](NOTICE.md).

```bash
python scripts/download_experiment.py --list
python scripts/download_experiment.py --output outputs/experimental_raw
python -m ghost_lattice prepare-experiment CG --raw outputs/experimental_raw --output outputs/inputs
python -m ghost_lattice prepare-experiment FG --raw outputs/experimental_raw --output outputs/inputs
python -m ghost_lattice run coarse_cp --method proposed --backend cuda --data-root outputs --output outputs/coarse_cp_proposed
python -m ghost_lattice fem coarse_cp --level 0 --data-root outputs --output outputs/coarse_cp_fem
python -m ghost_lattice compare outputs/coarse_cp_proposed/step_0064.npz outputs/coarse_cp_fem/step_0064.npz --quantity ges --output outputs/coarse_cp_comparison.json
```

Preprocessing reconstructs grain segmentation, orientations, registration,
the adopted 64-micrometre windows and the HR-DIC cleaning mask. Particle
calculations then generate new simulated fields on the measured geometry.
The original experimental observations are downloaded from their published
source.

## Code guide

| Location | Purpose |
| --- | --- |
| `src/ghost_lattice/cases.py`, `configs/` | Case definitions and numerical inputs |
| `src/ghost_lattice/models.py` | Particle geometry, materials and coupling assembly |
| `src/ghost_lattice/solver.py` | Accepted loading histories and field export |
| `src/ghost_lattice/fem.py` | Independent continuum reference generation |
| `src/ghost_lattice/experiment.py` | External EBSD/HR-DIC preprocessing |
| `src/ghost_lattice/comparison.py` | Raw-field comparisons at particle positions |
| `src/ghost_lattice/numerics/` | Numerical kernels and optional CUDA operations |
| `scripts/` | Downloads, local diagnostics and plotting |
| `tests/` | Small numerical and generation checks |

See [implementation notes](docs/implementation.md) and
[generation details](docs/generation.md) for conventions and outputs.

## Copyright

No software distribution license has been selected. Copyright is retained by
the respective copyright holders. See [NOTICE.md](NOTICE.md) for the software
statement and the separate experimental-data attribution.
