"""Small physical checks; no manuscript-scale solve or external data download."""

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ghost_lattice.cases import get_case
from ghost_lattice.models import build
from ghost_lattice.numerics.mls_safeguard import MLSGuard
from ghost_lattice.numerics.plane_lpm import mls_weights
from ghost_lattice.numerics.finite_elements import FEM, elastic, tri_box
from ghost_lattice.solver import run


class NumericalChecks(unittest.TestCase):
    def test_mls_reproduces_affine_fields(self):
        points = np.random.default_rng(17).normal(size=(12, 2))
        weights = mls_weights(points)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=2e-14)
        np.testing.assert_allclose(weights @ points, [0.0, 0.0], atol=2e-14)

    def test_guard_reports_loss_of_first_moment_on_fallback(self):
        guard = MLSGuard()
        points = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        weights = guard(points)
        np.testing.assert_array_equal(weights, [1.0, 0.0, 0.0])
        self.assertEqual(guard.report()["fallback_count"], 1)
        np.testing.assert_array_equal(weights @ points, points[0])

    def test_corrected_uniform_state_2d(self):
        model = build(get_case("demo"), "proposed").model
        strain = np.diag([1e-6, -3e-7])
        u = (model.X @ strain.T).ravel()
        force, _ = model.forces(model.B @ u, np.zeros(model.nb), np.zeros((model.N, 3)))
        residual = np.linalg.norm(force[model.free]) / np.linalg.norm(
            force[model.fixed]
        )
        self.assertLess(residual, 1e-10)

    def test_corrected_uniform_state_3d(self):
        case = replace(
            get_case("equiaxed_homogeneous"), size=8, grains=2, band=2, window_margin=3
        )
        model = build(case, "proposed").model
        u = (model.X @ np.diag([1e-6, 0.0, 0.0])).ravel()
        force, _ = model.forces(model.B @ u, np.zeros(model.nb), np.zeros((model.N, 6)))
        residual = np.linalg.norm(force[model.free]) / np.linalg.norm(
            force[model.fixed]
        )
        self.assertLess(residual, 1e-10)

    def test_independent_fem_affine_elastic_solution(self):
        mesh = tri_box(n=4, size=1.0)
        C = elastic()
        model = FEM(mesh, C)

        def material(strain, ep, alpha, tangent=True):
            return (
                np.einsum("ij,nj->ni", C, strain),
                ep.copy(),
                alpha.copy(),
                np.broadcast_to(C, (len(strain), 3, 3)) if tangent else None,
            )

        E = np.diag([1e-4, -3e-5])
        info = model.step(E, 1.0, material)
        self.assertLess(info["relative_residual"], 1e-9)
        np.testing.assert_allclose(model.u.reshape(-1, 2), mesh.p @ E.T, atol=1e-13)

    def test_public_cpu_run_produces_accepted_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            run("demo", folder, method="proposed", backend="cpu")
            import json

            result = json.loads((Path(folder) / "result.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["steps"], 30)
            self.assertTrue(result["snapshots"])
            self.assertLessEqual(
                max(x["independent_CPU_force_residual"] for x in result["history"]),
                1.05e-9,
            )


if __name__ == "__main__":
    unittest.main()
