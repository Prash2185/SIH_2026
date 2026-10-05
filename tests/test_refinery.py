"""
BharatOpt Core - MRPL Refinery Applications Unit Test (FR-APP1, FR-APP2, §8.1 - §8.2)
Tests:
1. Level 1 Monthly Crude Selection & Blending Solve + Verifier Check
2. Sensitivity Attribution (Duals & Reduced Costs) + Breakeven Pricing
3. Contrastive 'Why Not X?' Analysis on Unselected Spot Crudes
4. Level 2 Tank & CDU Multi-Period Scheduling Solve
5. Forward Simulation Verifier Validation (Tank Balances, Settling Times, Mutex)
"""

import unittest
import numpy as np

from bharatopt.core.model import Sense, VarType, ConstraintSense
from bharatopt.applications.mrpl_level1 import build_mrpl_level1_model
from bharatopt.applications.mrpl_level2 import build_mrpl_level2_model
from bharatopt.solvers.simplex import RevisedSimplexSolver, SolverStatus
from bharatopt.solvers.milp import BranchAndCutSolver
from bharatopt.verifier.verifier import IndependentVerifier
from bharatopt.verifier.forward_sim import ForwardSimulationVerifier
from bharatopt.explanation.bundle import ExplainabilityEngine


class TestMRPLRefineryModels(unittest.TestCase):

    def setUp(self):
        # Realistic MRPL-calibrated synthetic data
        self.crudes = [
            {
                "name": "Arab_Light", "cost_per_bbl": 78.0, "sulfur_wt_pct": 1.78, "api_gravity": 33.4,
                "density_kg_l": 0.858, "initial_stock": 100.0, "max_storage": 600.0,
                "yields": {"LPG": 0.03, "MS": 0.22, "HSD": 0.42, "ATF": 0.12, "FO": 0.15, "Bitumen": 0.06}
            },
            {
                "name": "Arab_Heavy", "cost_per_bbl": 68.0, "sulfur_wt_pct": 2.85, "api_gravity": 27.9,
                "density_kg_l": 0.887, "initial_stock": 80.0, "max_storage": 600.0,
                "yields": {"LPG": 0.02, "MS": 0.16, "HSD": 0.35, "ATF": 0.08, "FO": 0.25, "Bitumen": 0.14}
            },
            {
                "name": "Bonny_Light", "cost_per_bbl": 84.0, "sulfur_wt_pct": 0.14, "api_gravity": 35.3,
                "density_kg_l": 0.848, "initial_stock": 40.0, "max_storage": 600.0,
                "is_spot": True, "cargo_size": 200.0, "fixed_cargo_cost": 8000.0,
                "yields": {"LPG": 0.04, "MS": 0.26, "HSD": 0.48, "ATF": 0.14, "FO": 0.06, "Bitumen": 0.02}
            },
            {
                "name": "Maya_HighSulfur", "cost_per_bbl": 62.0, "sulfur_wt_pct": 3.40, "api_gravity": 21.8,
                "density_kg_l": 0.923, "initial_stock": 0.0, "max_storage": 600.0,
                "high_tan": True, "is_spot": True, "cargo_size": 150.0, "fixed_cargo_cost": 12000.0,
                "yields": {"LPG": 0.01, "MS": 0.12, "HSD": 0.30, "ATF": 0.05, "FO": 0.32, "Bitumen": 0.20}
            }
        ]

        self.cdus = [
            {"name": "CDU_Phase1", "min_capacity": 50.0, "max_capacity": 180.0, "max_sulfur_wt_pct": 2.0, "min_api_gravity": 28.0, "op_cost_per_bbl": 2.2, "tan_metallurgy": False},
            {"name": "CDU_Phase2", "min_capacity": 60.0, "max_capacity": 220.0, "max_sulfur_wt_pct": 2.5, "min_api_gravity": 26.0, "op_cost_per_bbl": 2.0, "tan_metallurgy": True}
        ]

        self.prices = {
            "LPG": 105.0, "MS": 115.0, "HSD": 98.0, "ATF": 112.0, "FO": 55.0, "Bitumen": 60.0
        }

        self.demands = {
            "LPG": (5.0, 50.0), "MS": (30.0, 150.0), "HSD": (60.0, 250.0),
            "ATF": (15.0, 80.0), "FO": (20.0, 120.0), "Bitumen": (10.0, 60.0)
        }

    def test_mrpl_level1_solve_and_verification(self):
        """Test MRPL Level 1 Monthly Crude Planning solve + Verifier Gate."""
        model = build_mrpl_level1_model(self.crudes, self.cdus, self.prices, self.demands)

        solver = BranchAndCutSolver()
        sol = solver.solve(model)
        self.assertIn(sol.status, (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE))
        self.assertGreater(sol.obj_val, 0.0)

        # Verifier Gate Check
        verifier = IndependentVerifier()
        v_rep = verifier.verify_solution(model, sol.x, sol.obj_val, sol.best_dual_bound)
        self.assertTrue(v_rep.passed, f"Verifier failed: {[c.message for c in v_rep.checks if not c.passed]}")
        self.assertLessEqual(v_rep.max_primal_violation, 1e-6)

    def test_mrpl_level1_explainability_and_why_not_x(self):
        """Test Explainability, Breakeven calculations, and contrastive 'Why Not X?'."""
        model = build_mrpl_level1_model(self.crudes, self.cdus, self.prices, self.demands)
        solver = RevisedSimplexSolver()
        sol = solver.solve(model)
        self.assertEqual(sol.status, SolverStatus.OPTIMAL)

        expl = ExplainabilityEngine()
        sens = expl.analyze_sensitivity(model, sol.x, sol.y, sol.reduced_costs)
        self.assertGreater(len(sens.reduced_costs), 0)

        # Contrastive 'Why Not X?' on unselected/spot variable
        why_not = expl.why_not_variable(model, sol.obj_val, "spot_buy_Bonny_Light", min_forced_val=1.0)
        self.assertIsNotNone(why_not.hypothesis)
        self.assertIsNotNone(why_not.explanation_summary)

    def test_mrpl_level2_scheduling_forward_sim(self):
        """Test MRPL Level 2 multi-period scheduling with Forward Simulation Verifier."""
        tanks = [
            {"id": "TK_01", "capacity": 150.0, "settling_time_periods": 2, "max_inflow_rate": 60.0, "initial_inventory": {"Arab_Light": 80.0}},
            {"id": "TK_02", "capacity": 150.0, "settling_time_periods": 2, "max_inflow_rate": 60.0, "initial_inventory": {"Arab_Heavy": 70.0}}
        ]
        cdus = [{"id": "CDU_01", "max_rate": 35.0, "min_throughput": 0.0}]
        receipts = [{"period": 1, "grade": "Arab_Light", "volume": 40.0}]

        model = build_mrpl_level2_model(
            tanks=tanks, cdus=cdus, crude_grades=["Arab_Light", "Arab_Heavy"],
            num_periods=5, vessel_receipts=receipts
        )

        solver = BranchAndCutSolver(max_nodes=500)
        sol = solver.solve(model)
        self.assertIn(sol.status, (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE))

        # Forward Simulation Verifier
        sim_sol = {
            "inventory": {},
            "inflow": {},
            "outflow": {},
            "receiving": {},
            "feeding": {}
        }
        # Populate solution dictionary from variable values
        for j, name in enumerate(model.var_names):
            val = sol.x[j]
            if name.startswith("inv_"):
                # inv_TK_01_Arab_Light_t0
                parts = name.split("_")
                t_id = f"{parts[1]}_{parts[2]}"
                grade = f"{parts[3]}_{parts[4]}"
                period = int(parts[5][1:])
                sim_sol["inventory"][(t_id, grade, period)] = val
            elif name.startswith("inflow_"):
                parts = name.split("_")
                t_id = f"{parts[1]}_{parts[2]}"
                grade = f"{parts[3]}_{parts[4]}"
                period = int(parts[5][1:])
                sim_sol["inflow"][(t_id, grade, period)] = val
            elif name.startswith("outflow_"):
                parts = name.split("_")
                t_id = f"{parts[1]}_{parts[2]}"
                grade = f"{parts[3]}_{parts[4]}"
                cdu_id = f"{parts[5]}_{parts[6]}"
                period = int(parts[7][1:])
                sim_sol["outflow"][(t_id, grade, cdu_id, period)] = val
            elif name.startswith("rcv_"):
                parts = name.split("_")
                t_id = f"{parts[1]}_{parts[2]}"
                period = int(parts[3][1:])
                sim_sol["receiving"][(t_id, period)] = val
            elif name.startswith("feed_"):
                parts = name.split("_")
                t_id = f"{parts[1]}_{parts[2]}"
                cdu_id = f"{parts[3]}_{parts[4]}"
                period = int(parts[5][1:])
                sim_sol["feeding"][(t_id, cdu_id, period)] = val

        fwd_sim = ForwardSimulationVerifier()
        sim_report = fwd_sim.verify_schedule(5, tanks, cdus, ["Arab_Light", "Arab_Heavy"], sim_sol)
        self.assertTrue(sim_report.passed)


if __name__ == "__main__":
    unittest.main()
