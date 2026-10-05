"""
BharatOpt Core - Command Line Interface (CLI)
Provides command-line commands:
  bharatopt solve <model.mps> [--tier T1] [--out result.json]
  bharatopt appendix-a
  bharatopt benchmark [--suite l1|l2|netlib|scale]
  bharatopt ui [--port 8080]
"""

import argparse
import json
import sys
import time
from typing import Optional

from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense
from bharatopt.io.mps_reader import MPSReader
from bharatopt.run_tiers.tiers import RunTier, TierOrchestrator
from bharatopt.solvers.simplex import RevisedSimplexSolver, SolverStatus
from bharatopt.solvers.ipm import MehrotraIPMSolver
from bharatopt.solvers.milp import BranchAndCutSolver
from bharatopt.verifier.verifier import IndependentVerifier
from bharatopt.explanation.bundle import ExplainabilityEngine, ExplanationBundle
from bharatopt.applications.mrpl_level1 import build_mrpl_level1_model
from bharatopt.applications.mrpl_level2 import build_mrpl_level2_model


def run_appendix_a_suite():
    """Runs all 10 hand-checked equation tests from Appendix A."""
    from tests.test_appendix_a import run_all_appendix_a_tests
    return run_all_appendix_a_tests()


def main():
    parser = argparse.ArgumentParser(
        prog="bharatopt",
        description="BharatOpt Core v5.0 - Sovereign LP / QP / MILP Optimization Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: solve
    solve_parser = subparsers.add_parser("solve", help="Solve an MPS/LP mathematical model file")
    solve_parser.add_argument("file", help="Path to MPS or LP file")
    solve_parser.add_argument("--tier", choices=["T0", "T1", "T2"], default="T1", help="Execution Tier")
    solve_parser.add_argument("--out", help="Save JSON ExplanationBundle to path")

    # Command: appendix-a
    subparsers.add_parser("appendix-a", help="Run the 10 Hand-Checked Equation Tests from Appendix A")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run industrial benchmark suite")
    bench_parser.add_argument("--suite", choices=["l1", "l2", "scale", "all"], default="all")

    # Command: audit (Gate G0)
    subparsers.add_parser("audit", help="Run Gate G0 Clean-Room Sovereignty & Provenance Audit")

    # Command: ui
    ui_parser = subparsers.add_parser("ui", help="Launch interactive Web Dashboard & Visualizer")
    ui_parser.add_argument("--port", type=int, default=8000, help="Port to host Web Dashboard")

    args = parser.parse_args()

    if args.command == "appendix-a":
        print("=" * 70)
        print("  BHARATOPT CORE v5.0 - APPENDIX A HAND-CHECKED TEST SUITE")
        print("=" * 70)
        success = run_appendix_a_suite()
        sys.exit(0 if success else 1)

    elif args.command == "audit":
        from bharatopt.verifier.clean_room_audit import run_clean_room_audit
        success = run_clean_room_audit()
        sys.exit(0 if success else 1)

    elif args.command == "benchmark":
        from benchmarks.run_benchmarks import run_all_benchmarks
        run_all_benchmarks()

    elif args.command == "solve":
        print(f"[*] Ingesting model from: {args.file}")
        t0 = time.perf_counter()
        model = MPSReader.read(args.file)
        print(f"[*] Model Loaded: {model.name} | Vars: {model.num_vars} (Integer: {len(model.integer_indices)}) | Ineq: {model.num_ineq} | Eq: {model.num_eq}")
        print(f"[*] Model SHA-256 Hash: {model.compute_hash()[:16]}...")

        orchestrator = TierOrchestrator()
        sol, ver_rep, replay = orchestrator.solve(model, tier=RunTier(args.tier))
        dt = time.perf_counter() - t0

        print("\n" + "=" * 50)
        print(f"  SOLVER STATUS:   {sol.status.value}")
        print(f"  OBJECTIVE VALUE: {sol.obj_val:.8f}")
        print(f"  BEST DUAL BOUND: {sol.best_dual_bound:.8f}")
        print(f"  RELATIVE GAP:    {sol.relative_gap * 100:.4f}%")
        print(f"  SOLVE TIME:      {dt*1000:.2f} ms")
        print(f"  VERIFIER GATE:   {'PASSED [CERTIFIED]' if ver_rep.passed else 'FAILED [WITHHELD]'}")
        print("=" * 50)

        for chk in ver_rep.checks:
            status_sym = "[x]" if chk.passed else "[!]"
            print(f"  {status_sym} {chk.name:<32} (val: {chk.measured_val:.2e}, tol: {chk.tolerance:.2e})")

        if args.out:
            expl_engine = ExplainabilityEngine()
            y_arr = getattr(sol, 'y', None)
            rc_arr = getattr(sol, 'reduced_costs', None)
            sens = expl_engine.analyze_sensitivity(model, sol.x, y_arr, rc_arr)

            bundle = ExplanationBundle(
                run_id=f"RUN_{int(time.time()*1000)}",
                model_hash=model.compute_hash(),
                status=sol.status.value,
                objective=sol.obj_val,
                dual_bound=sol.best_dual_bound,
                gap=sol.relative_gap,
                iterations=getattr(sol, 'iterations', 0),
                node_count=getattr(sol, 'node_count', 1),
                solve_time_sec=dt,
                verifier_dict=ver_rep.to_dict()
            )
            bundle.shadow_prices = sens.shadow_prices
            bundle.reduced_costs = sens.reduced_costs
            bundle.breakevens = sens.breakeven_prices

            with open(args.out, 'w') as f:
                f.write(bundle.to_json())
            print(f"\n[+] Saved ExplanationBundle to: {args.out}")

    elif args.command == "ui":
        from bharatopt.ui.server import start_dashboard_server
        start_dashboard_server(port=args.port)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
