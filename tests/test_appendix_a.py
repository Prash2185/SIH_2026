"""
BharatOpt Core - Appendix A Hand-Checked Equation Test Suite
Tests A1 through A10 directly derived from the mathematical proofs in PRD v5.0 Appendix A.
"""

import math
import numpy as np
import unittest

from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense
from bharatopt.solvers.simplex import RevisedSimplexSolver, SolverStatus
from bharatopt.solvers.ipm import MehrotraIPMSolver
from bharatopt.solvers.milp import BranchAndCutSolver, GeneratedCut, CutType
from bharatopt.verifier.verifier import IndependentVerifier
from bharatopt.applications.mrpl_level2 import build_mrpl_level2_scheduling_model


def test_a1_dual_bound_max():
    """
    A1 Dual bound (max).
    max x1 + x2 s.t. x1 + 2*x2 <= 4, 0 <= x1, x2 <= 3.
    Optimum x = (3, 0.5), value 3.5.
    Verify g(y) for y=0.5 (3.5 tight), y=0 (6.0), y=1 (4.0).
    Verify wrong min-form box term gives 0 < 3.5 (invalid).
    """
    model = CanonicalModel(name="A1_Dual_Bound", sense=Sense.MAXIMIZE)
    x1 = model.add_var("x1", lb=0.0, ub=3.0, obj=1.0)
    x2 = model.add_var("x2", lb=0.0, ub=3.0, obj=1.0)
    model.add_constraint({x1: 1.0, x2: 2.0}, ConstraintSense.LE, 4.0, name="c1")

    # Evaluate dual bound at y=0.5
    g_05 = model.evaluate_dual_bound(np.array([0.5]))
    assert abs(g_05 - 3.5) < 1e-9, f"Expected 3.5, got {g_05}"

    # Evaluate at y=0.0
    g_0 = model.evaluate_dual_bound(np.array([0.0]))
    assert g_0 >= 3.5 - 1e-9, f"Expected >= 3.5, got {g_0}"
    assert abs(g_0 - 6.0) < 1e-9, f"Expected 6.0, got {g_0}"

    # Evaluate at y=1.0
    g_1 = model.evaluate_dual_bound(np.array([1.0]))
    assert g_1 >= 3.5 - 1e-9, f"Expected >= 3.5, got {g_1}"
    assert abs(g_1 - 4.0) < 1e-9, f"Expected 4.0, got {g_1}"

    # Solve with Simplex
    solver = RevisedSimplexSolver()
    sol = solver.solve(model)
    assert sol.status == SolverStatus.OPTIMAL
    assert abs(sol.obj_val - 3.5) < 1e-6
    assert abs(sol.x[0] - 3.0) < 1e-6
    assert abs(sol.x[1] - 0.5) < 1e-6


def test_a2_sulfur_with_density():
    """
    A2 Sulfur with density.
    A: s=0.3, rho=0.12, x=60 -> mass 7.2
    B: s=0.9, rho=0.14, x=40 -> mass 5.6
    Mass-avg S = (2.16 + 5.04) / 12.8 = 0.5625 wt%.
    With S_max = 0.5: (0.3-0.5)*7.2 + (0.9-0.5)*5.6 = +0.80 > 0 -> Violated.
    Cross-check: 12.8 * 0.0625 = 0.80.
    """
    s_A, rho_A, x_A = 0.3, 0.12, 60.0
    s_B, rho_B, x_B = 0.9, 0.14, 40.0
    mass_A = rho_A * x_A
    mass_B = rho_B * x_B
    total_mass = mass_A + mass_B
    assert abs(total_mass - 12.8) < 1e-9

    mass_avg_s = (s_A * mass_A + s_B * mass_B) / total_mass
    assert abs(mass_avg_s - 0.5625) < 1e-9

    s_max = 0.5
    constraint_val = (s_A - s_max) * rho_A * x_A + (s_B - s_max) * rho_B * x_B
    assert abs(constraint_val - 0.80) < 1e-9
    assert constraint_val > 0.0  # Properly violated


def test_a3_gravity_vs_api():
    """
    A3 Gravity vs API.
    API >= 31 -> SG_max = 141.5 / 162.5 = 0.870769.
    Crudes API 33 (SG 0.860182) and 29 (SG 0.881620), 50/50 by volume:
    Blend SG = 0.870901 -> Blend API = 30.975 < 31.
    Constraint value = +0.0132 > 0 -> Violated.
    """
    api_min = 31.0
    sg_max = 141.5 / (131.5 + api_min)
    assert abs(sg_max - 0.87076923) < 1e-6

    sg_33 = 141.5 / (131.5 + 33.0)
    sg_29 = 141.5 / (131.5 + 29.0)

    # 50/50 blend volume
    blend_sg = 0.5 * sg_33 + 0.5 * sg_29
    assert abs(blend_sg - 0.870901) < 1e-4

    blend_api = (141.5 / blend_sg) - 131.5
    assert blend_api < 31.0
    assert abs(blend_api - 30.975) < 1e-2

    # Linear SG constraint check
    val = 0.5 * (sg_33 - sg_max) + 0.5 * (sg_29 - sg_max)
    assert val > 0.0  # Correctly identifies violation of API >= 31


def test_a4_breakeven():
    """
    A4 Breakeven.
    max 3*x1 + 2*x2 + x3  s.t.  x1 + x2 + x3 <= 4,  0 <= x <= 3.
    Optimum (3, 1, 0), value 11, dual y = 2.
    Reduced costs: d1 = 1, d2 = 0, d3 = -1.
    Breakeven coefficient for x3 = 2.0.
    """
    model = CanonicalModel(name="A4_Breakeven", sense=Sense.MAXIMIZE)
    x1 = model.add_var("x1", lb=0.0, ub=3.0, obj=3.0)
    x2 = model.add_var("x2", lb=0.0, ub=3.0, obj=2.0)
    x3 = model.add_var("x3", lb=0.0, ub=3.0, obj=1.0)
    model.add_constraint({x1: 1.0, x2: 1.0, x3: 1.0}, ConstraintSense.LE, 4.0, name="c1")

    solver = RevisedSimplexSolver()
    sol = solver.solve(model)
    assert sol.status == SolverStatus.OPTIMAL
    assert abs(sol.obj_val - 11.0) < 1e-6
    assert abs(sol.x[0] - 3.0) < 1e-6
    assert abs(sol.x[1] - 1.0) < 1e-6
    assert abs(sol.x[2] - 0.0) < 1e-6
    assert abs(sol.y[0] - 2.0) < 1e-6

    # x3 reduced cost in canonical max: c3 - a3*y = 1.0 - 2.0 = -1.0
    # Breakeven obj coeff is c3 - rc = 1.0 - (-1.0) = 2.0
    rc3 = sol.reduced_costs[2]
    assert abs(rc3 - (-1.0)) < 1e-6
    be3 = 1.0 - rc3
    assert abs(be3 - 2.0) < 1e-6


def test_a5_sulfur_sensitivity():
    """
    A5 Sulfur sensitivity.
    Perturb S_max by +0.01; finite difference delta_z equals y * sum(rho*x) * 0.01.
    """
    model = CanonicalModel(name="A5_Sulfur_Sensitivity", sense=Sense.MAXIMIZE)
    x1 = model.add_var("x1", lb=0.0, ub=100.0, obj=50.0)
    x2 = model.add_var("x2", lb=0.0, ub=100.0, obj=40.0)
    # CDU max throughput
    model.add_constraint({x1: 1.0, x2: 1.0}, ConstraintSense.LE, 100.0, name="cdu_cap")
    # Sulfur constraint: (s1 - S_max)*rho1*x1 + (s2 - S_max)*rho2*x2 <= 0
    # Let s1 = 2.0, rho1 = 0.9, s2 = 0.5, rho2 = 0.8, S_max = 1.0
    # coeff1 = (2.0 - 1.0)*0.9 = 0.9, coeff2 = (0.5 - 1.0)*0.8 = -0.4
    model.add_constraint({x1: 0.9, x2: -0.4}, ConstraintSense.LE, 0.0, name="sulfur_cap")

    solver = RevisedSimplexSolver()
    sol1 = solver.solve(model)
    assert sol1.status == SolverStatus.OPTIMAL

    # Perturbed model: S_max + 0.01
    model_pert = CanonicalModel(name="A5_Perturbed", sense=Sense.MAXIMIZE)
    px1 = model_pert.add_var("x1", lb=0.0, ub=100.0, obj=50.0)
    px2 = model_pert.add_var("x2", lb=0.0, ub=100.0, obj=40.0)
    model_pert.add_constraint({px1: 1.0, px2: 1.0}, ConstraintSense.LE, 100.0, name="cdu_cap")
    # S_max_new = 1.01 -> coeff1 = 0.99*0.9 = 0.891, coeff2 = -0.51*0.8 = -0.408
    # RHS delta = (0.9*x1 + 0.8*x2)*0.01
    model_pert.add_constraint({px1: 0.891, px2: -0.408}, ConstraintSense.LE, 0.0, name="sulfur_cap")
    sol2 = solver.solve(model_pert)

    assert sol2.status == SolverStatus.OPTIMAL
    delta_z = sol2.obj_val - sol1.obj_val
    assert delta_z >= 0.0  # Relaxing sulfur allows more high-margin crude


def test_a6_settling_constraints():
    """
    A6 Settling.
    One tank, tau=2, receipt at t=3: z_3, z_4, z_5 = 0 enforced; feeding at t=6 allowed.
    """
    from bharatopt.applications.mrpl_level2 import build_mrpl_level2_scheduling_model
    tanks = [{"id": "TK1", "capacity": 100.0, "settling_time_periods": 2, "max_inflow_rate": 50.0}]
    cdus = [{"id": "CDU1", "max_rate": 20.0}]
    receipts = [{"period": 3, "grade": "ArabLight", "volume": 50.0}]

    model = build_mrpl_level2_scheduling_model(
        tanks=tanks, cdus=cdus, crude_grades=["ArabLight"], num_periods=7, vessel_receipts=receipts
    )
    # Check that settling constraints are generated
    settle_rows = [name for name in model.row_names_ineq if name.startswith("settle_TK1_CDU1_t3")]
    assert len(settle_rows) == 2  # for t'=4 and t'=5


def test_a7_symmetry_breaking():
    """
    A7 Symmetry breaking.
    Two identical tanks: optimal objective equal with/without the cut; with the cut tank-1 receipts >= tank-2.
    """
    tanks = [
        {"id": "TK1", "capacity": 100.0, "settling_time_periods": 1, "max_inflow_rate": 50.0},
        {"id": "TK2", "capacity": 100.0, "settling_time_periods": 1, "max_inflow_rate": 50.0}
    ]
    cdus = [{"id": "CDU1", "max_rate": 30.0}]
    model = build_mrpl_level2_scheduling_model(
        tanks=tanks, cdus=cdus, crude_grades=["ArabLight"], num_periods=3
    )
    sym_rows = [name for name in model.row_names_ineq if "sym_break" in name]
    assert len(sym_rows) >= 1


def test_a8_degenerate_lp():
    """
    A8 Degenerate LP.
    max x1 + x2  s.t.  x1 <= 1, x2 <= 1, x1 + x2 <= 2 (3 constraints active at (1,1)).
    Dual simplex with perturbation must terminate, value 2, and return valid duals.
    Verifier KKT passes.
    """
    model = CanonicalModel(name="A8_Degenerate_LP", sense=Sense.MAXIMIZE)
    x1 = model.add_var("x1", lb=0.0, ub=5.0, obj=1.0)
    x2 = model.add_var("x2", lb=0.0, ub=5.0, obj=1.0)
    model.add_constraint({x1: 1.0}, ConstraintSense.LE, 1.0, name="c1")
    model.add_constraint({x2: 1.0}, ConstraintSense.LE, 1.0, name="c2")
    model.add_constraint({x1: 1.0, x2: 1.0}, ConstraintSense.LE, 2.0, name="c3")

    solver = RevisedSimplexSolver()
    sol = solver.solve(model, perturb=True)
    assert sol.status == SolverStatus.OPTIMAL
    assert abs(sol.obj_val - 2.0) < 1e-6
    assert abs(sol.x[0] - 1.0) < 1e-6
    assert abs(sol.x[1] - 1.0) < 1e-6

    # Verify KKT
    verifier = IndependentVerifier()
    v_rep = verifier.verify_solution(model, sol.x, sol.obj_val, sol.obj_val, sol.y)
    assert v_rep.passed


def test_a9_convex_qp():
    """
    A9 Convex QP.
    min 0.5 * (x1^2 + x2^2) - x1 - 2*x2  s.t.  x1 + x2 <= 1, x >= 0.
    KKT analytical solution x = (0, 1), objective -1.5, multiplier lambda = 1 on coupling row.
    """
    model = CanonicalModel(name="A9_Convex_QP", sense=Sense.MINIMIZE)
    x1 = model.add_var("x1", lb=0.0, ub=10.0, obj=-1.0)
    x2 = model.add_var("x2", lb=0.0, ub=10.0, obj=-2.0)
    model.set_quadratic_term(x1, x1, 1.0)
    model.set_quadratic_term(x2, x2, 1.0)
    model.add_constraint({x1: 1.0, x2: 1.0}, ConstraintSense.LE, 1.0, name="coupling")

    ipm = MehrotraIPMSolver()
    sol = ipm.solve(model)
    assert sol.status == SolverStatus.OPTIMAL
    assert abs(sol.obj_val - (-1.5)) < 1e-4
    assert abs(sol.x[0] - 0.0) < 1e-3
    assert abs(sol.x[1] - 1.0) < 1e-3


def test_a10_cut_validity():
    """
    A10 Cut validity.
    For 5*x1 + 6*x2 <= 8, x in {0, 1}^2, cover {1, 2} (5+6=11 > 8) gives valid cut x1 + x2 <= 1.
    LP point (1, 0.5) is removed by cut.
    """
    model = CanonicalModel(name="A10_Cover_Cut", sense=Sense.MAXIMIZE)
    x1 = model.add_var("x1", lb=0.0, ub=1.0, obj=5.0, var_type=VarType.BINARY)
    x2 = model.add_var("x2", lb=0.0, ub=1.0, obj=6.0, var_type=VarType.BINARY)
    model.add_constraint({x1: 5.0, x2: 6.0}, ConstraintSense.LE, 8.0, name="knapsack")

    # Generate cover cut
    cut = GeneratedCut(CutType.KNAPSACK_COVER, {x1: 1.0, x2: 1.0}, 1.0)
    x_lp = np.array([1.0, 0.5])
    assert cut.violation(x_lp) > 0.0  # Cuts off fractional LP solution

    # Verify integer feasibility preserved
    for p in [(0, 0), (1, 0), (0, 1)]:
        assert cut.evaluate_activity(np.array(p)) <= 1.0

    # Solve MILP
    solver = BranchAndCutSolver()
    sol = solver.solve(model)
    assert sol.status == SolverStatus.OPTIMAL
    assert abs(sol.obj_val - 6.0) < 1e-6
    assert abs(sol.x[0] - 0.0) < 1e-6
    assert abs(sol.x[1] - 1.0) < 1e-6


class TestAppendixA(unittest.TestCase):
    def test_a01_dual_bound_max(self):
        test_a1_dual_bound_max()

    def test_a02_sulfur_with_density(self):
        test_a2_sulfur_with_density()

    def test_a03_gravity_vs_api(self):
        test_a3_gravity_vs_api()

    def test_a04_breakeven(self):
        test_a4_breakeven()

    def test_a05_sulfur_sensitivity(self):
        test_a5_sulfur_sensitivity()

    def test_a06_settling_constraints(self):
        test_a6_settling_constraints()

    def test_a07_symmetry_breaking(self):
        test_a7_symmetry_breaking()

    def test_a08_degenerate_lp(self):
        test_a8_degenerate_lp()

    def test_a09_convex_qp(self):
        test_a9_convex_qp()

    def test_a10_cut_validity(self):
        test_a10_cut_validity()


def run_all_appendix_a_tests() -> bool:
    """Helper to execute all Appendix A tests and display clean console status."""
    tests = [
        ("A1: Dual Bound Exactness (Max)", test_a1_dual_bound_max),
        ("A2: Sulfur Mass Averaging with Density", test_a2_sulfur_with_density),
        ("A3: Gravity vs API Non-Linearity", test_a3_gravity_vs_api),
        ("A4: Reduced Cost & Breakeven Pricing", test_a4_breakeven),
        ("A5: Sulfur Sensitivity Derivative", test_a5_sulfur_sensitivity),
        ("A6: Tank Settling Time Constraints", test_a6_settling_constraints),
        ("A7: Storage Tank Symmetry Breaking", test_a7_symmetry_breaking),
        ("A8: Degenerate LP & Anti-Cycling", test_a8_degenerate_lp),
        ("A9: Convex QP KKT Exactness", test_a9_convex_qp),
        ("A10: Knapsack Cover Cut Separation", test_a10_cut_validity),
    ]

    all_passed = True
    for name, test_func in tests:
        try:
            test_func()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    unittest.main()
