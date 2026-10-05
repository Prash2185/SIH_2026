"""
BharatOpt Core - MPS and LP Format Roundtrip Unit Tests (FR-IO1)
"""

import os
import tempfile
import unittest

from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense
from bharatopt.io.mps_reader import MPSReader, LPFormatReader
from bharatopt.solvers.simplex import RevisedSimplexSolver, SolverStatus


class TestFileIO(unittest.TestCase):

    def test_mps_write_and_read_roundtrip(self):
        """Verify MPS export, re-import, and solution equality."""
        model = CanonicalModel(name="TestMPS_Roundtrip", sense=Sense.MAXIMIZE)
        x1 = model.add_var("x1", lb=0.0, ub=10.0, obj=5.0)
        x2 = model.add_var("x2", lb=0.0, ub=8.0, obj=4.0)
        model.add_constraint({x1: 2.0, x2: 1.0}, ConstraintSense.LE, 14.0, name="c1")
        model.add_constraint({x1: 1.0, x2: 2.0}, ConstraintSense.LE, 12.0, name="c2")

        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as f:
            temp_path = f.name

        try:
            MPSReader.write(model, temp_path)
            loaded_model = MPSReader.read(temp_path)

            self.assertEqual(loaded_model.num_vars, 2)
            self.assertEqual(loaded_model.num_ineq, 2)

            solver = RevisedSimplexSolver()
            res_orig = solver.solve(model)
            res_loaded = solver.solve(loaded_model)

            self.assertEqual(res_orig.status, SolverStatus.OPTIMAL)
            self.assertEqual(res_loaded.status, SolverStatus.OPTIMAL)
            self.assertAlmostEqual(res_orig.obj_val, res_loaded.obj_val, places=6)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_lp_format_write(self):
        """Verify CPLEX LP export."""
        model = CanonicalModel(name="TestLP_Write", sense=Sense.MAXIMIZE)
        x1 = model.add_var("x1", lb=0.0, ub=1.0, obj=10.0, var_type=VarType.BINARY)
        x2 = model.add_var("x2", lb=0.0, ub=5.0, obj=20.0, var_type=VarType.INTEGER)
        model.add_constraint({x1: 1.0, x2: 2.0}, ConstraintSense.LE, 7.0, name="knapsack")

        with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
            temp_path = f.name

        try:
            LPFormatReader.write(model, temp_path)
            with open(temp_path, 'r') as f:
                content = f.read()
            self.assertIn("Maximize", content)
            self.assertIn("knapsack:", content)
            self.assertIn("Binaries", content)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
