"""
BharatOpt Core - Convex Quadratic Programming & Power Dispatch Example
Demonstrates:
1. Formulation of convex quadratic objective: min 0.5 * x^T Q x + c^T x s.t. A x <= b, E x = d
2. Positive semi-definiteness verification of Q
3. Primal-Dual Mehrotra Interior Point Method solving the KKT system
4. Exact Wolfe Dual Bound calculation: b^T y + d^T lam - 0.5 x^T Q x
5. Independent Verifier Gate Certification
"""

import time
import numpy as np

from bharatopt.applications.benchmarks import build_power_dispatch_qp
from bharatopt.solvers.ipm import MehrotraIPMSolver
from bharatopt.verifier.verifier import IndependentVerifier


def main():
    print("=" * 80)
    print("  BHARATOPT CORE v5.0 -- ECONOMIC POWER DISPATCH (CONVEX QP)")
    print("=" * 80)

    num_generators = 6
    demand = 1800.0
    print(f"[1] Building Convex QP for {num_generators} Generators (Demand = {demand} MW)...")
    model = build_power_dispatch_qp(num_generators=num_generators, total_demand=demand)

    print(f"    * Variables: {model.num_vars}")
    print(f"    * Quadratic terms in Q: {len(model.Q_triplets)}")
    print(f"    * Equalities (Power Balance): {model.num_eq}")

    print("\n[2] Solving via Sovereign Mehrotra IPM Solver...")
    t0 = time.perf_counter()
    ipm = MehrotraIPMSolver(max_iters=100, tol_feas=1e-8, tol_gap=1e-8)
    sol = ipm.solve(model)
    dt = time.perf_counter() - t0

    print(f"    [+] Status:              {sol.status.value}")
    print(f"    [+] Total Cost:          ${sol.obj_val:,.2f}")
    print(f"    [+] Wolfe Dual Bound:    ${sol.wolfe_dual_bound:,.2f}")
    print(f"    [+] Complementarity Gap: {sol.duality_gap:.2e}")
    print(f"    [+] IPM Iterations:      {sol.iterations}")
    print(f"    [+] Solve Time:          {dt*1000:.2f} ms")

    print("\n[3] Generator Power Outputs:")
    for j in range(model.num_vars):
        print(f"    * {model.var_names[j]:<12}: {sol.x[j]:>8.2f} MW (Limits: [{model.lb[j]:.1f}, {model.ub[j]:.1f}])")

    print("\n[4] Zero-Trust Verifier Gate:")
    verifier = IndependentVerifier()
    v_rep = verifier.verify_solution(model, sol.x, sol.obj_val, sol.wolfe_dual_bound, sol.y)
    for chk in v_rep.checks:
        symbol = "[PASS]" if chk.passed else "[FAIL]"
        print(f"    {symbol} {chk.name:<32} (resid: {chk.measured_val:.2e})")


if __name__ == "__main__":
    main()
