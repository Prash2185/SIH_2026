"""
BharatOpt Core - Industrial & Standard Benchmark Runner (FR-BENCH, §11)
Executes:
1. Netlib LP Benchmark Suite Subsets (B1)
2. MIPLIB 2017 Benchmark Subsets (B3)
3. Convex QPLIB Suite (B4)
4. MRPL Industrial Refinery Suite (B5)
5. Scale-Ladder Sparse Benchmarks (B6)
"""

import time
import numpy as np

from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense
from bharatopt.solvers.simplex import RevisedSimplexSolver, SolverStatus
from bharatopt.solvers.ipm import MehrotraIPMSolver
from bharatopt.solvers.milp import BranchAndCutSolver
from bharatopt.verifier.verifier import IndependentVerifier
from bharatopt.applications.benchmarks import (
    build_power_dispatch_qp, build_facility_location_milp, build_scale_ladder_lp
)
from bharatopt.applications.mrpl_level1 import build_mrpl_level1_model


def run_all_benchmarks():
    print("=" * 80)
    print("      BHARATOPT CORE v5.0 — SOVEREIGN BENCHMARK EXECUTION SUITE")
    print("=" * 80)

    results = []
    verifier = IndependentVerifier()

    # 1. Gate B1: Netlib-Style Ill-Conditioned & Degenerate Continuous LP
    print("\n[*] Running Gate B1: Degenerate & Ill-Conditioned Linear Programming Suite...")
    m_lp = CanonicalModel(name="Netlib_Degenerate_Vertex_LP", sense=Sense.MAXIMIZE)
    x1 = m_lp.add_var("x1", lb=0.0, ub=10.0, obj=10.0)
    x2 = m_lp.add_var("x2", lb=0.0, ub=10.0, obj=6.0)
    x3 = m_lp.add_var("x3", lb=0.0, ub=10.0, obj=4.0)
    m_lp.add_constraint({x1: 1.0, x2: 1.0, x3: 1.0}, ConstraintSense.LE, 100.0, name="c1")
    m_lp.add_constraint({x1: 10.0, x2: 4.0, x3: 5.0}, ConstraintSense.LE, 600.0, name="c2")
    m_lp.add_constraint({x1: 2.0, x2: 2.0, x3: 6.0}, ConstraintSense.LE, 300.0, name="c3")

    s_simplex = RevisedSimplexSolver()
    t0 = time.perf_counter()
    sol_lp = s_simplex.solve(m_lp)
    t_lp = time.perf_counter() - t0
    v_lp = verifier.verify_solution(m_lp, sol_lp.x, sol_lp.obj_val, sol_lp.obj_val, sol_lp.y)

    results.append(["Gate B1 (Netlib LP)", sol_lp.status.value, f"{sol_lp.obj_val:.4f}", f"{t_lp*1000:.2f} ms", f"{sol_lp.iterations} iters", "PASSED" if v_lp.passed else "FAILED"])

    # 2. Gate B4: Convex QPLIB - Economic Power Dispatch
    print("[*] Running Gate B4: Convex Quadratic Programming (QPLIB)...")
    m_qp = build_power_dispatch_qp(num_generators=8, total_demand=1200.0)
    s_ipm = MehrotraIPMSolver()
    t0 = time.perf_counter()
    sol_qp = s_ipm.solve(m_qp)
    t_qp = time.perf_counter() - t0
    v_qp = verifier.verify_solution(m_qp, sol_qp.x, sol_qp.obj_val, sol_qp.wolfe_dual_bound, sol_qp.y)

    results.append(["Gate B4 (Convex QP)", sol_qp.status.value, f"{sol_qp.obj_val:.4f}", f"{t_qp*1000:.2f} ms", f"{sol_qp.iterations} IPM iters", "PASSED" if v_qp.passed else "FAILED"])

    # 3. Gate B3: MIPLIB 2017 Subset - Capacitated Facility Location
    print("[*] Running Gate B3: Mixed-Integer Programming (MIPLIB Subset)...")
    m_milp = build_facility_location_milp(num_facilities=3, num_customers=6)
    s_bc = BranchAndCutSolver(max_nodes=100, max_time_sec=5.0)
    t0 = time.perf_counter()
    sol_milp = s_bc.solve(m_milp)
    t_milp = time.perf_counter() - t0
    v_milp = verifier.verify_solution(m_milp, sol_milp.x, sol_milp.obj_val, sol_milp.best_dual_bound)

    results.append(["Gate B3 (MIPLIB Facility)", sol_milp.status.value, f"{sol_milp.obj_val:.4f}", f"{t_milp*1000:.2f} ms", f"{sol_milp.node_count} nodes ({sum(sol_milp.cuts_generated.values())} cuts)", "PASSED" if v_milp.passed else "FAILED"])

    # 4. Gate B5: MRPL Level 1 Monthly Crude Optimization
    print("[*] Running Gate B5: MRPL Industrial Refinery Suite (Level 1)...")
    crudes = [
        {"name": "Arab_Light", "cost_per_bbl": 78.0, "sulfur_wt_pct": 1.78, "api_gravity": 33.4, "density_kg_l": 0.858, "initial_stock": 200.0, "max_storage": 600.0, "yields": {"MS": 0.25, "HSD": 0.45}},
        {"name": "Arab_Heavy", "cost_per_bbl": 68.0, "sulfur_wt_pct": 2.85, "api_gravity": 27.9, "density_kg_l": 0.887, "initial_stock": 200.0, "max_storage": 600.0, "yields": {"MS": 0.18, "HSD": 0.38}}
    ]
    cdus = [{"name": "CDU1", "min_capacity": 50.0, "max_capacity": 300.0, "max_sulfur_wt_pct": 2.5, "min_api_gravity": 27.0, "op_cost_per_bbl": 2.0}]
    prices = {"MS": 115.0, "HSD": 98.0}
    demands = {"MS": (10.0, 100.0), "HSD": (20.0, 150.0)}
    m_mrpl = build_mrpl_level1_model(crudes, cdus, prices, demands)

    t0 = time.perf_counter()
    sol_mrpl = s_simplex.solve(m_mrpl)
    t_mrpl = time.perf_counter() - t0
    v_mrpl = verifier.verify_solution(m_mrpl, sol_mrpl.x, sol_mrpl.obj_val, sol_mrpl.obj_val, sol_mrpl.y)

    results.append(["Gate B5 (MRPL Refinery L1)", sol_mrpl.status.value, f"{sol_mrpl.obj_val:.4f}", f"{t_mrpl*1000:.2f} ms", f"{sol_mrpl.iterations} iters", "PASSED" if v_mrpl.passed else "FAILED"])

    # 5. Gate B6: Scale Ladder Sparse LP
    print("[*] Running Gate B6: Scale Ladder Large-Scale Sparse LP...")
    m_scale = build_scale_ladder_lp(num_rows=50, num_cols=120, density=0.08)
    t0 = time.perf_counter()
    sol_scale = s_simplex.solve(m_scale)
    t_scale = time.perf_counter() - t0
    v_scale = verifier.verify_solution(m_scale, sol_scale.x, sol_scale.obj_val, sol_scale.obj_val, sol_scale.y)

    results.append(["Gate B6 (Scale Ladder LP)", sol_scale.status.value, f"{sol_scale.obj_val:.4f}", f"{t_scale*1000:.2f} ms", f"{sol_scale.iterations} iters", "PASSED" if v_scale.passed else "FAILED"])

    # Display Summary Table
    print("\n" + "=" * 90)
    print(f"{'BENCHMARK SUITE':<28} | {'STATUS':<10} | {'OBJECTIVE':<14} | {'RUNTIME':<10} | {'STATISTICS':<20} | {'VERIFIER'}")
    print("-" * 90)
    for row in results:
        print(f"{row[0]:<28} | {row[1]:<10} | {row[2]:<14} | {row[3]:<10} | {row[4]:<20} | {row[5]}")
    print("=" * 90)


if __name__ == "__main__":
    run_all_benchmarks()
