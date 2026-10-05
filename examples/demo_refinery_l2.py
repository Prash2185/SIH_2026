"""
BharatOpt Core - MRPL Level 2 Multi-Period Tank & CDU Scheduling Example
Demonstrates:
1. Dynamic tank scheduling with 2 crude grades, 2 storage tanks, and 1 CDU
2. Tank settling times (tau = 2 periods) and receiving/feeding mutual exclusion
3. Forward Simulation Verifier checking step-by-step physical mass balances
"""

import time
from bharatopt.applications.mrpl_level2 import build_mrpl_level2_model
from bharatopt.run_tiers.tiers import RunTier, TierOrchestrator
from bharatopt.verifier.forward_sim import ForwardSimulationVerifier


def main():
    print("=" * 80)
    print("  BHARATOPT CORE v5.0 -- MRPL TANK & CDU SCHEDULING (LEVEL 2)")
    print("=" * 80)

    tanks = [
        {"id": "TK_01", "capacity": 200.0, "settling_time_periods": 2, "max_inflow_rate": 80.0, "initial_inventory": {"Arab_Light": 120.0}},
        {"id": "TK_02", "capacity": 200.0, "settling_time_periods": 2, "max_inflow_rate": 80.0, "initial_inventory": {"Arab_Heavy": 100.0}}
    ]

    cdus = [
        {"id": "CDU_1", "max_rate": 40.0, "min_throughput": 0.0}
    ]

    crude_grades = ["Arab_Light", "Arab_Heavy"]
    num_periods = 4
    vessel_receipts = [
        {"period": 1, "grade": "Arab_Light", "volume": 50.0}
    ]

    print("[1] Building Multi-Period Tank/CDU Scheduling MILP Formulation...")
    model = build_mrpl_level2_model(
        tanks=tanks, cdus=cdus, crude_grades=crude_grades,
        num_periods=num_periods, vessel_receipts=vessel_receipts
    )
    print(f"    * Variables: {model.num_vars} (Discrete: {len(model.integer_indices)})")
    print(f"    * Constraints: {model.num_constraints}")

    print("\n[2] Solving via BharatOpt Sovereign Engine (Tier T1)...")
    t0 = time.perf_counter()
    orchestrator = TierOrchestrator()
    sol, ver_rep, replay = orchestrator.solve(model, tier=RunTier.T1, max_nodes=500)
    dt = time.perf_counter() - t0

    print(f"    [+] Status:          {sol.status.value}")
    print(f"    [+] Objective Value: {sol.obj_val:,.2f}")
    print(f"    [+] Solve Time:      {dt*1000:.2f} ms")

    # Forward Simulation Verifier
    print("\n[3] Forward Simulation Physical Verifier (FR-VER2):")
    sim_sol = {"inventory": {}, "inflow": {}, "outflow": {}, "receiving": {}, "feeding": {}}
    for j, name in enumerate(model.var_names):
        val = sol.x[j] if len(sol.x) > j else 0.0
        if name.startswith("inv_"):
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
    sim_report = fwd_sim.verify_schedule(num_periods, tanks, cdus, crude_grades, sim_sol)
    print(f"    [+] Forward Simulation Status: {'PASSED [CERTIFIED]' if sim_report.passed else 'FAILED'}")
    print(f"    [+] Max Inventory Drift:       {sim_report.max_inventory_drift:.2e}")
    print(f"    [+] Settling Violations:       {len(sim_report.settling_violations)}")
    print(f"    [+] Mutex Violations:          {len(sim_report.mutex_violations)}")


if __name__ == "__main__":
    main()
