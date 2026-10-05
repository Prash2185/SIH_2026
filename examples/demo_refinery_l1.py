"""
BharatOpt Core - MRPL Level 1 Monthly Crude Planning Example
Demonstrates:
1. Ingestion of 4 crude candidates (Arab Light, Arab Heavy, Bonny Light, Maya High-Sulfur)
2. CDUs with strict sulfur mass-balance and non-linear API gravity conversion
3. Solving via BharatOpt Sovereign Engine in Tier T1 (Deterministic)
4. Independent Verifier Gate Certificate
5. Opportunity costs (reduced costs) & Breakeven pricing for unselected crudes
6. Contrastive 'Why Not X?' Analysis for high-tan Maya crude
7. Exporting structured JSON ExplanationBundle
"""

import json
import time

from bharatopt.core.model import Sense
from bharatopt.applications.mrpl_level1 import build_mrpl_level1_model
from bharatopt.run_tiers.tiers import RunTier, TierOrchestrator
from bharatopt.explanation.bundle import ExplainabilityEngine, ExplanationBundle


def main():
    print("=" * 80)
    print("  BHARATOPT CORE v5.0 — MRPL MONTHLY CRUDE SELECTION (LEVEL 1)")
    print("=" * 80)

    # 1. Define MRPL Crude Basket Data
    crudes = [
        {
            "name": "Arab_Light", "cost_per_bbl": 78.0, "sulfur_wt_pct": 1.78, "api_gravity": 33.4,
            "density_kg_l": 0.858, "initial_stock": 250.0, "max_storage": 800.0,
            "yields": {"LPG": 0.04, "MS": 0.24, "HSD": 0.44, "ATF": 0.12, "FO": 0.10, "Bitumen": 0.06}
        },
        {
            "name": "Arab_Heavy", "cost_per_bbl": 66.0, "sulfur_wt_pct": 2.85, "api_gravity": 27.9,
            "density_kg_l": 0.887, "initial_stock": 200.0, "max_storage": 800.0,
            "yields": {"LPG": 0.02, "MS": 0.16, "HSD": 0.36, "ATF": 0.08, "FO": 0.24, "Bitumen": 0.14}
        },
        {
            "name": "Bonny_Light", "cost_per_bbl": 85.0, "sulfur_wt_pct": 0.14, "api_gravity": 35.3,
            "density_kg_l": 0.848, "initial_stock": 100.0, "max_storage": 800.0,
            "is_spot": True, "cargo_size": 200.0, "fixed_cargo_cost": 8000.0,
            "yields": {"LPG": 0.05, "MS": 0.28, "HSD": 0.48, "ATF": 0.14, "FO": 0.04, "Bitumen": 0.01}
        },
        {
            "name": "Maya_HighTAN", "cost_per_bbl": 58.0, "sulfur_wt_pct": 3.40, "api_gravity": 21.8,
            "density_kg_l": 0.923, "initial_stock": 0.0, "max_storage": 800.0,
            "high_tan": True, "is_spot": True, "cargo_size": 150.0, "fixed_cargo_cost": 15000.0,
            "yields": {"LPG": 0.01, "MS": 0.10, "HSD": 0.28, "ATF": 0.04, "FO": 0.35, "Bitumen": 0.22}
        }
    ]

    cdus = [
        {"name": "CDU_1", "min_capacity": 100.0, "max_capacity": 300.0, "max_sulfur_wt_pct": 2.2, "min_api_gravity": 28.5, "op_cost_per_bbl": 2.2, "tan_metallurgy": False},
        {"name": "CDU_2", "min_capacity": 80.0, "max_capacity": 250.0, "max_sulfur_wt_pct": 2.6, "min_api_gravity": 26.0, "op_cost_per_bbl": 2.0, "tan_metallurgy": True}
    ]

    prices = {
        "LPG": 105.0, "MS": 118.0, "HSD": 99.0, "ATF": 115.0, "FO": 52.0, "Bitumen": 62.0
    }

    demands = {
        "LPG": (10.0, 100.0), "MS": (50.0, 300.0), "HSD": (100.0, 450.0),
        "ATF": (25.0, 150.0), "FO": (30.0, 200.0), "Bitumen": (15.0, 100.0)
    }

    print("[1] Building Canonical MILP Formulation...")
    model = build_mrpl_level1_model(crudes, cdus, prices, demands, carbon_cap=600000.0)
    print(f"    • Variables: {model.num_vars} (Discrete: {len(model.integer_indices)})")
    print(f"    • Constraints: {model.num_constraints} (Ineq: {model.num_ineq}, Eq: {model.num_eq})")
    print(f"    • Model SHA-256 Hash: {model.compute_hash()}")
    from bharatopt.io.mps_reader import MPSReader
    MPSReader.write(model, "mrpl_level1.mps")
    print(f"    • Exported MPS Benchmark File: mrpl_level1.mps")

    # 2. Solve in Tier T1
    print("\n[2] Solving via BharatOpt Sovereign Engine (Tier T1)...")
    t0 = time.perf_counter()
    orchestrator = TierOrchestrator()
    sol, ver_rep, replay = orchestrator.solve(model, tier=RunTier.T1, seed=42)
    dt = time.perf_counter() - t0

    print(f"\n    [+] Status:          {sol.status.value}")
    print(f"    [+] Net Margin:      ${sol.obj_val:,.2f}")
    print(f"    [+] Best Dual Bound: ${sol.best_dual_bound:,.2f}")
    print(f"    [+] Optimality Gap:  {sol.relative_gap * 100:.4f}%")
    print(f"    [+] Compute Time:    {dt*1000:.2f} ms")

    # 3. Verifier Gate Report
    print("\n[3] Zero-Trust Verifier Gate (Layer 1 Trust):")
    for chk in ver_rep.checks:
        symbol = "[PASS]" if chk.passed else "[FAIL]"
        print(f"    {symbol} {chk.name:<32} (resid: {chk.measured_val:.2e}, tol: {chk.tolerance:.2e})")

    # 4. Sensitivity & Explainability (Layer 2 Trust)
    print("\n[4] Economic Sensitivity & Opportunity Costs (Layer 2 Trust):")
    expl = ExplainabilityEngine()
    sens = expl.analyze_sensitivity(model, sol.x, getattr(sol, 'y', None), getattr(sol, 'reduced_costs', None))

    print(f"    {'VARIABLE':<30} | {'VOLUME (kbbl)':<14} | {'OPPORTUNITY COST':<18} | {'BREAKEVEN PRICE'}")
    print("    " + "-" * 78)
    for rc in sens.reduced_costs:
        if rc['value'] > 0.01 or abs(rc['reduced_cost']) > 1e-4:
            print(f"    {rc['var_name']:<30} | {rc['value']:>12.2f}  | ${rc['reduced_cost']:>16.2f} | ${rc['breakeven_cost']:>14.2f}")

    # 5. Contrastive "Why Not X?"
    print("\n[5] Contrastive 'Why Not X?' Analysis:")
    why_not = expl.why_not_variable(model, sol.obj_val, "spot_buy_Maya_HighTAN", min_forced_val=1.0)
    print(f"    • Hypothesis: {why_not.hypothesis}")
    print(f"    • Result:     {'FEASIBLE BUT SUBOPTIMAL' if why_not.is_feasible else 'INFEASIBLE'}")
    print(f"    • Explanation: {why_not.explanation_summary}")

    # 6. Save JSON ExplanationBundle
    bundle = ExplanationBundle(
        run_id=f"MRPL_L1_{int(time.time())}",
        model_hash=model.compute_hash(),
        status=sol.status.value,
        objective=sol.obj_val,
        dual_bound=sol.best_dual_bound,
        gap=sol.relative_gap,
        iterations=sol.iterations,
        node_count=sol.node_count,
        solve_time_sec=dt,
        verifier_dict=ver_rep.to_dict()
    )
    bundle.reduced_costs = sens.reduced_costs
    bundle.breakevens = sens.breakeven_prices
    bundle.why_not_analyses = [why_not.to_dict()]

    with open("mrpl_level1_explanation_bundle.json", "w") as f:
        f.write(bundle.to_json())
    print("\n[+] Exported ExplanationBundle to: mrpl_level1_explanation_bundle.json")


if __name__ == "__main__":
    main()
