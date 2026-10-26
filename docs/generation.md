# Generating inputs and results

## Synthetic numerical cases

`cases.py` defines domain dimensions, number of grains, lattice spacing,
yield contrast, loading increments, observation windows and random seeds.
`models.build` generates a Voronoi microstructure and independently oriented
particle lattices, then assembles the selected coupling. `solver.run` solves
the prescribed history and exports checkpoint fields. A fixed seed reproduces
the selected random geometry and material assignment.

`fem.run` generates a separate grain-conforming continuum mesh with the same
geometry, material assignment and loading. Planar and equiaxed cases use
quadratic elements. The columnar reference uses linear elements with three
displacement components and generalized plane strain. The perforated plate uses
an annular mesh fitted to the circular hole and square loading boundary.
Measured-map meshes follow the pixel interfaces and require `triangle`.

Each run writes `configuration.json`, `progress.json`, `result.json` and
`step_NNNN.npz` checkpoint files into the requested output directory. Output
directories must be empty. The result distinguishes model/mesh construction,
equilibrium and field-export failures and records accepted steps. A partial
history does not establish completion of the prescribed loading path.

Particle snapshots contain physical coordinates `X`, displacements `u`, grain
labels and the observation-window mask `win`. Plastic cases contain accumulated
plasticity `alpha` and plastic strain `epsp`. Planar snapshots also contain
engineering strain and effective shear strain `ges`. Elastic snapshots contain
strain and stress. Measured coordinates and displacements are in micrometres;
idealized cases use their configured length unit.

FEM snapshots contain mesh geometry, quadrature fields and displacement
histories. `comparison.compare` samples quadratic FEM histories at particle
positions for accumulated plasticity, and samples displacement gradients for
effective shear strain. The columnar P1 fields are constant within each
element. The comparison checks grain labels and uses particle observation
windows. For a return-leg accumulated-plasticity comparison, pass `--reverse`.
For measured CP boundary-resistance laws, compare `ges`; the generic history
sampler implements the base CP law, while these laws have spatial/evolving
resistance. The recorded checkpoint kinematics already include that resistance.

`--level` controls explicit FEM refinement. Assess mesh convergence by comparing
successive reference fields at common positions. Solver acceptance alone does
not establish spatial convergence. Running the scripts generates a fresh set
of results; this code-only distribution contains no archived paper results.

## External experimental observations

`scripts/download_experiment.py` downloads only the eight selected source files
listed in `configs/experimental_sources.json` and checks their published MD5
values. It downloads no software and executes no downloaded content.

`prepare-experiment` reconstructs the adopted processing: 8-degree EBSD grain
segmentation, merging regions smaller than ten pixels, a central-pixel grain
orientation, FCC Schmid factors, affine DIC-to-EBSD registration, a centered
64-micrometre square and the original small/thin-grain merging. The retained
`a_um=0.5` field is a coordinate normalization for the processed raster. The
actual particle spacing remains the physical `spacing` in the Case definition
and is applied by `models.build`.

The historical crop recipe evaluates inscribed radii with a per-grain Euclidean
distance transform, separately from the later shared-boundary approximation
in the general orientation utilities. The CG recipe explicitly assigns the
15-pixel border fragment containing pixel (33,127) to the neighboring region
containing (37,123). This compact preprocessing instruction reproduces the
adopted CG grain map. The resulting CG and FG windows contain 131 and 272
regions, respectively. The processing report records this instruction.

The spreadsheet reader removes the published index row and column. HR-DIC
cleaning masks values above five times the 99.9th percentile and expands the
mask using a 5-by-5 stencil. Raw and cleaned strain arrays are both exported.
The generated processing JSON records the source DOI, license and choices.
The experimental observations supply geometry and qualitative strain context;
the numerical solver generates the simulated fields separately.
