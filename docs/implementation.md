# Implementation

`cases.py` and `configs/` define inputs. `models.py` assembles particle models,
`solver.py` advances their loading histories, `fem.py` generates independent
continuum references and `comparison.py` evaluates fresh raw fields.
`experiment.py` prepares external measured microstructures.

The numerical kernels are organized by task: `plane_lpm`, `columnar_lpm` and
`equiaxed_lpm` contain particle constitutive and force operations;
`rotated_plane` and `sparse_assembly` provide equivalent sparse assembly;
`grain_geometry`, `ghost_bcc` and `raster_geometry` generate the geometry;
`orientations` and `twin_boundaries` process measured crystal information.
`finite_elements`, `quadratic_elements`, `fitted_mesh` and `raster_mesh` implement
the independent continuum models. `cuda_*` modules provide optional GPU paths.

The proposed construction uses linear MLS, energy-derived reaction transfer
and elastic-strain ghost-force correction (GFC). Particle merging uses its
force assembly. Both J2 constructions use star-consistent radial return (SCR).
CP solves the implicit slip equations with the actual local stiffness.

GFC is applied to the assembled force after differentiation of the interaction
energy. Its particle-specific correction can have a nonzero resultant; it does
not enforce an exact global momentum identity. For a deficient 2-D MLS stencil,
the recorded safeguard chooses the nearest existing support with unit weight.
It preserves a constant displacement, with a first moment equal to that
support's offset. Construction diagnostics record such fallbacks.

Accepted particle steps are independently checked using the original CPU force
assembly with relative residual at most 1.05e-9. Optional CUDA paths include
periodic CPU/GPU material checks. FEM uses independent continuum constitutive
equations and reports its own equilibrium residual. The small tests check
affine reproduction, correction, independent elasticity and data-generation
workflows. Manuscript-scale jobs are explicit CLI operations.
