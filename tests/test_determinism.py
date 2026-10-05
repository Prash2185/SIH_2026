"""
BharatOpt Core - Determinism & Reproducibility Unit Test (FR-DET1, §9.3)
Verifies that 100 repeated runs in Tier T1 generate identical SHA-256 plan hashes.
"""

import hashlib
import json
import unittest
import numpy as np

from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense
from bharatopt.run_tiers.tiers import RunTier, TierOrchestrator
from bharatopt.applications.mrpl_level1 import build_mrpl_level1_model


class TestDeterminismT1(unittest.TestCase):

    def test_100_run_identical_plan_hashes(self):
        """FR-DET1: 100 T1 runs on one platform must give identical plan hashes."""
        # Setup synthetic model
        crudes = [
            {"name": "ArabLight", "cost_per_bbl": 75.0, "sulfur_wt_pct": 1.2, "api_gravity": 33.0, "density_kg_l": 0.86, "initial_stock": 50.0, "max_storage": 300.0, "yields": {"MS": 0.3, "HSD": 0.5}},
            {"name": "ArabHeavy", "cost_per_bbl": 65.0, "sulfur_wt_pct": 2.8, "api_gravity": 27.0, "density_kg_l": 0.89, "initial_stock": 50.0, "max_storage": 300.0, "yields": {"MS": 0.2, "HSD": 0.4}},
            {"name": "BonnyLight", "cost_per_bbl": 82.0, "sulfur_wt_pct": 0.3, "api_gravity": 35.5, "density_kg_l": 0.84, "initial_stock": 20.0, "max_storage": 300.0, "is_spot": True, "cargo_size": 100.0, "fixed_cargo_cost": 5000.0, "yields": {"MS": 0.35, "HSD": 0.52}}
        ]
        cdus = [{"name": "CDU1", "min_capacity": 50.0, "max_capacity": 200.0, "max_sulfur_wt_pct": 1.5, "min_api_gravity": 30.0, "op_cost_per_bbl": 2.0}]
        prices = {"MS": 110.0, "HSD": 95.0}
        demands = {"MS": (20.0, 150.0), "HSD": (30.0, 200.0)}

        model = build_mrpl_level1_model(crudes, cdus, prices, demands)
        orchestrator = TierOrchestrator()

        reference_hash = None
        reference_obj = None

        # Execute 100 runs in Tier T1
        for run_idx in range(100):
            sol, ver_rep, replay = orchestrator.solve(model, tier=RunTier.T1, seed=42)

            # Construct plan fingerprint hash
            plan_data = {
                "obj": round(sol.obj_val, 7),
                "x": [round(val, 7) for val in sol.x]
            }
            plan_hash = hashlib.sha256(json.dumps(plan_data, sort_keys=True).encode('utf-8')).hexdigest()

            if reference_hash is None:
                reference_hash = plan_hash
                reference_obj = sol.obj_val
            else:
                self.assertEqual(plan_hash, reference_hash, f"Run {run_idx} produced different plan hash!")
                self.assertAlmostEqual(sol.obj_val, reference_obj, places=7)

        print(f"  [+] Determinism Test Passed: 100/100 identical T1 hashes ({reference_hash[:16]}...)")


if __name__ == "__main__":
    unittest.main()
