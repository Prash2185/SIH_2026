"""
BharatOpt Core - MRPL Level 2 14-Day Tank & CDU Scheduling Formulation (FR-APP2, §8.2)
Implements:
1. Multi-period discrete scheduling (T periods)
2. Storage tanks with individual capacities C_i and settling times tau_i
3. Single-grade containment assignment: sum_j a_{i,j,t} <= 1
4. Receiving and feeding mutual exclusion: r_{i,t} + sum_c z_{i,c,t} <= 1
5. Rigorous settling-time quiet window (A6): z_{i,c,t'} + r_{i,t} <= 1 for t' in [t, t+tau_i]
6. Symmetry-breaking cuts for identical tanks (A7)
"""

from typing import List, Dict, Tuple, Any, Optional
from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense


def build_mrpl_level2_scheduling_model(
    tanks: List[Dict[str, Any]],
    cdus: List[Dict[str, Any]],
    crude_grades: List[str],
    num_periods: int = 14,
    vessel_receipts: Optional[List[Dict[str, Any]]] = None
) -> CanonicalModel:
    """
    Builds canonical MILP for MRPL Level 2 Tank and CDU Scheduling.
    """
    model = CanonicalModel(name="MRPL_14Day_Tank_CDU_Scheduling_L2", sense=Sense.MAXIMIZE)
    if vessel_receipts is None:
        vessel_receipts = []

    # Map vessel receipts by (period, grade)
    receipt_map: Dict[Tuple[int, str], float] = {}
    for v in vessel_receipts:
        receipt_map[(v["period"], v["grade"])] = v["volume"]

    # Variables:
    # a[i, j, t] in {0, 1}: Tank i contains crude grade j at period t
    # I[i, j, t] >= 0: Inventory of grade j in tank i at period t (kbbl)
    # in_flow[i, j, t] >= 0: Receipt flow into tank i of grade j at period t
    # out_flow[i, j, c, t] >= 0: Feeding flow from tank i of grade j to CDU c at period t
    # r[i, t] in {0, 1}: Tank i is in receiving mode at period t
    # z[i, c, t] in {0, 1}: Tank i is actively feeding CDU c at period t

    a_vars: Dict[Tuple[str, str, int], int] = {}
    I_vars: Dict[Tuple[str, str, int], int] = {}
    in_vars: Dict[Tuple[str, str, int], int] = {}
    out_vars: Dict[Tuple[str, str, str, int], int] = {}
    r_vars: Dict[Tuple[str, int], int] = {}
    z_vars: Dict[Tuple[str, str, int], int] = {}

    for t in range(num_periods):
        for tank in tanks:
            t_id = tank["id"]
            cap = tank["capacity"]

            # Receiving flag
            r_idx = model.add_var(
                name=f"rcv_{t_id}_t{t}",
                lb=0.0, ub=1.0,
                obj=-1.0,  # Small operational penalty for switching
                var_type=VarType.BINARY
            )
            r_vars[(t_id, t)] = r_idx

            # Feeding flags per CDU
            for cdu in cdus:
                c_id = cdu["id"]
                z_idx = model.add_var(
                    name=f"feed_{t_id}_{c_id}_t{t}",
                    lb=0.0, ub=1.0,
                    obj=10.0,  # Positive incentive for sustained CDU feeding
                    var_type=VarType.BINARY
                )
                z_vars[(t_id, c_id, t)] = z_idx

            # Per grade variables
            for g in crude_grades:
                a_idx = model.add_var(
                    name=f"assign_{t_id}_{g}_t{t}",
                    lb=0.0, ub=1.0,
                    obj=0.0,
                    var_type=VarType.BINARY
                )
                a_vars[(t_id, g, t)] = a_idx

                I_idx = model.add_var(
                    name=f"inv_{t_id}_{g}_t{t}",
                    lb=0.0, ub=cap,
                    obj=-0.1,  # Inventory holding cost
                    var_type=VarType.CONTINUOUS
                )
                I_vars[(t_id, g, t)] = I_idx

                in_idx = model.add_var(
                    name=f"inflow_{t_id}_{g}_t{t}",
                    lb=0.0, ub=tank.get("max_inflow_rate", 50.0),
                    obj=0.0,
                    var_type=VarType.CONTINUOUS
                )
                in_vars[(t_id, g, t)] = in_idx

                for cdu in cdus:
                    c_id = cdu["id"]
                    out_idx = model.add_var(
                        name=f"outflow_{t_id}_{g}_{c_id}_t{t}",
                        lb=0.0, ub=cdu.get("max_rate", 40.0),
                        obj=5.0,  # Value of crude processed
                        var_type=VarType.CONTINUOUS
                    )
                    out_vars[(t_id, g, c_id, t)] = out_idx

    # 2. Constraints

    for t in range(num_periods):
        # (A) Single Grade per Tank: sum_j a[i, j, t] <= 1
        for tank in tanks:
            t_id = tank["id"]
            cap = tank["capacity"]
            row_single = {a_vars[(t_id, g, t)]: 1.0 for g in crude_grades}
            model.add_constraint(row_single, ConstraintSense.LE, 1.0, name=f"single_grade_{t_id}_t{t}")

            # (B) Inventory Capacity vs Assignment: I[i, j, t] <= C_i * a[i, j, t]
            for g in crude_grades:
                row_cap = {
                    I_vars[(t_id, g, t)]: 1.0,
                    a_vars[(t_id, g, t)]: -cap
                }
                model.add_constraint(row_cap, ConstraintSense.LE, 0.0, name=f"inv_bound_{t_id}_{g}_t{t}")

            # (C) Receiving & Feeding Mutual Exclusion: r[i, t] + sum_c z[i, c, t] <= 1
            row_mutex = {r_vars[(t_id, t)]: 1.0}
            for cdu in cdus:
                row_mutex[z_vars[(t_id, cdu["id"], t)]] = 1.0
            model.add_constraint(row_mutex, ConstraintSense.LE, 1.0, name=f"mutex_rcv_feed_{t_id}_t{t}")

            # (D) Settling Time Quiet Period (A6):
            # When receipt occurs at period t (r[i, t]=1), feeding z[i, c, t'] is forbidden for t' in [t, t+tau_i]
            tau = tank.get("settling_time_periods", 2)
            for t_prime in range(t, min(num_periods, t + tau + 1)):
                if t_prime != t:
                    for cdu in cdus:
                        c_id = cdu["id"]
                        row_settle = {
                            r_vars[(t_id, t)]: 1.0,
                            z_vars[(t_id, c_id, t_prime)]: 1.0
                        }
                        model.add_constraint(row_settle, ConstraintSense.LE, 1.0, name=f"settle_{t_id}_{c_id}_t{t}_to_t{t_prime}")

            # (E) Inflow and Outflow Link to Modes
            max_in = tank.get("max_inflow_rate", 50.0)
            row_in_link = {r_vars[(t_id, t)]: -max_in}
            for g in crude_grades:
                row_in_link[in_vars[(t_id, g, t)]] = 1.0
            model.add_constraint(row_in_link, ConstraintSense.LE, 0.0, name=f"inflow_link_{t_id}_t{t}")

            for cdu in cdus:
                c_id = cdu["id"]
                max_out = cdu.get("max_rate", 40.0)
                row_out_link = {z_vars[(t_id, c_id, t)]: -max_out}
                for g in crude_grades:
                    row_out_link[out_vars[(t_id, g, c_id, t)]] = 1.0
                model.add_constraint(row_out_link, ConstraintSense.LE, 0.0, name=f"outflow_link_{t_id}_{c_id}_t{t}")

            # (F) Inventory Balance: I[i, j, t] = I[i, j, t-1] + in[i, j, t] - sum_c out[i, j, c, t]
            for g in crude_grades:
                row_bal = {
                    I_vars[(t_id, g, t)]: 1.0,
                    in_vars[(t_id, g, t)]: -1.0
                }
                for cdu in cdus:
                    row_bal[out_vars[(t_id, g, cdu["id"], t)]] = 1.0

                if t > 0:
                    row_bal[I_vars[(t_id, g, t - 1)]] = -1.0
                    rhs = 0.0
                else:
                    init_inv = tank.get("initial_inventory", {}).get(g, 0.0)
                    rhs = init_inv

                model.add_constraint(row_bal, ConstraintSense.EQ, rhs, name=f"inv_step_{t_id}_{g}_t{t}")

        # (G) CDU Throughput Limits
        for cdu in cdus:
            c_id = cdu["id"]
            row_cdu = {}
            for tank in tanks:
                t_id = tank["id"]
                for g in crude_grades:
                    row_cdu[out_vars[(t_id, g, c_id, t)]] = 1.0
            model.add_constraint(row_cdu, ConstraintSense.LE, cdu.get("max_rate", 40.0), name=f"cdu_max_{c_id}_t{t}")

        # (H) Vessel Receipt Deliveries
        for g in crude_grades:
            vol = receipt_map.get((t, g), 0.0)
            if vol > 0:
                row_vessel = {in_vars[(tank["id"], g, t)]: 1.0 for tank in tanks}
                model.add_constraint(row_vessel, ConstraintSense.EQ, vol, name=f"vessel_receipt_{g}_t{t}")

    # (I) Symmetry Breaking for Identical Tanks (A7)
    for k in range(len(tanks) - 1):
        t1 = tanks[k]
        t2 = tanks[k + 1]
        if t1.get("capacity") == t2.get("capacity") and t1.get("settling_time_periods") == t2.get("settling_time_periods"):
            # Enforce total outflow from Tank 1 >= Tank 2
            row_sym = {}
            for t in range(num_periods):
                for g in crude_grades:
                    for cdu in cdus:
                        row_sym[out_vars[(t1["id"], g, cdu["id"], t)]] = 1.0
                        row_sym[out_vars[(t2["id"], g, cdu["id"], t)]] = -1.0
            model.add_constraint(row_sym, ConstraintSense.GE, 0.0, name=f"sym_break_{t1['id']}_{t2['id']}")

    return model


build_mrpl_level2_model = build_mrpl_level2_scheduling_model
