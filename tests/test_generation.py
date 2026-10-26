"""Check portable generation with small models and synthetic observations."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ghost_lattice.cases import get_case
from ghost_lattice.fem import run as run_fem
from ghost_lattice.solver import run as run_particles
from ghost_lattice.comparison import compare
from ghost_lattice.experiment import clean_strain, crop_grains, read_strain_sheet


class GenerationChecks(unittest.TestCase):
    def test_particle_fem_comparison_workflow(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            case = replace(get_case("demo"), increment=0.5, checkpoints=(3.0,))
            run_particles(case, root / "particles")
            result = run_fem(case, root / "fem", level=1)
            self.assertEqual(result["steps"], 6)
            error = compare(
                root / "particles/step_0006.npz", root / "fem/step_0006.npz"
            )
            self.assertGreater(error["points"], 10)
            self.assertTrue(np.isfinite(error["relative_L1"]))

    def test_independent_cp_reference_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            case = replace(
                get_case("planar_cp"),
                size=12,
                grains=3,
                band=2,
                window_margin=3,
                increment=0.5,
                checkpoints=(2.0,),
            )
            result = run_fem(case, folder)
            self.assertEqual(result["steps"], 4)
            with np.load(Path(folder) / "step_0004.npz") as field:
                self.assertGreater(float(field["alpha"].max()), 0)
                self.assertTrue(np.all(np.isfinite(field["stress"])))

    def test_experimental_cleaning_preserves_regular_values(self):
        field = np.ones((100, 100))
        field[10, 10] = 1000
        cleaned = clean_strain(field)
        self.assertTrue(np.isnan(cleaned[8:13, 8:13]).all())
        self.assertEqual(cleaned[0, 0], 1)

    def test_crop_preserves_orientation_mapping(self):
        grains = np.zeros((100, 100), int)
        grains[:, 50:] = 1
        prep = dict(
            G=grains,
            step=1,
            sf=np.array([0.4, 0.5]),
            ori=np.array([[10, 20, 30], [40, 50, 60]]),
        )
        crop = crop_grains(
            prep, np.array([[1, 0], [0, 1], [0, 0]]), (100, 100), crop_um=64
        )
        np.testing.assert_array_equal(crop["phi1"], [10, 40])
        np.testing.assert_array_equal(crop["grain_orig"], [0, 1])
        self.assertEqual(crop["G"].shape, (64, 64))

    def test_spreadsheet_index_row_and_column(self):
        try:
            import openpyxl
        except ImportError:
            self.skipTest("Install the experiment extra to test XLSX input")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.xlsx"
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.append([None, 0, 1])
            sheet.append([0, 0.1, 0.2])
            sheet.append([1, 0.3, None])
            workbook.save(path)
            workbook.close()
            np.testing.assert_allclose(
                read_strain_sheet(path), [[0.1, 0.2], [0.3, np.nan]], equal_nan=True
            )


if __name__ == "__main__":
    unittest.main()
