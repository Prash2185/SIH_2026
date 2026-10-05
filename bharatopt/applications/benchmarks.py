"""
BharatOpt Core - Industrial Benchmark Problem Suite (FR-BENCH, §11)
Clean-room problem generators:
1. Unit Commitment / Economic Power Dispatch (MILP / Convex QP)
2. Multi-Echelon Supply Chain & Facility Location (MILP)
3. Highly Degenerate Assignment / Transportation Network Flow (Degenerate LP)
4. Markowitz Portfolio Optimization with Covariance Matrix (Convex QP)
5. Scale-Ladder Sparse LP Generator (10^3 to 10^5 nonzeros)
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Any
from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense


def build_power_dispatch_qp(num_generators: int = 10, total_demand: float = 2500.0) -> CanonicalModel:
    """
    Economic Power Dispatch with Quadratic Fuel Cost:
      min sum_i (0.5 * alpha_i * p_i^2 + beta_i * p_i + gamma_i)
      s.t. sum_i p_i == total_demand,  P_i_min <= p_i <= P_i_max
    """
    model = CanonicalModel(name="Economic_Power_Dispatch_Convex_QP", sense=Sense.MINIMIZE)

    p_vars = []
    demand_row: Dict[int, float] = {}

    rng = np.random.RandomState(42)
    for i in range(num_generators):
        alpha = 0.05 + 0.02 * rng.rand()   # Quadratic fuel curvature
        beta = 15.0 + 5.0 * rng.rand()     # Linear cost
        p_min = 50.0 + 20.0 * rng.rand()
        p_max = 300.0 + 100.0 * rng.rand()

        p_idx = model.add_var(
            name=f"p_gen_{i}",
            lb=p_min, ub=p_max,
            obj=beta,
            var_type=VarType.CONTINUOUS
        )
        p_vars.append(p_idx)
        demand_row[p_idx] = 1.0

        # Set quadratic term 0.5 * alpha * p_i^2
        model.set_quadratic_term(p_idx, p_idx, alpha)

    # Power balance equality
    model.add_constraint(demand_row, ConstraintSense.EQ, total_demand, name="power_balance")
    return model


def build_facility_location_milp(num_facilities: int = 5, num_customers: int = 15) -> CanonicalModel:
    """
    Capacitated Facility Location (MILP):
      min sum_i f_i y_i + sum_{i,j} c_{ij} x_{ij}
      s.t. sum_i x_{ij} = d_j,  sum_j x_{ij} <= C_i y_i,  y_i in {0,1}, x_{ij} >= 0
    """
    model = CanonicalModel(name="Capacitated_Facility_Location_MILP", sense=Sense.MINIMIZE)

    rng = np.random.RandomState(101)
    y_vars = []
    x_vars: Dict[Tuple[int, int], int] = {}

    for i in range(num_facilities):
        fixed_cost = float(rng.uniform(1000.0, 3000.0))
        y_idx = model.add_var(name=f"open_fac_{i}", lb=0.0, ub=1.0, obj=fixed_cost, var_type=VarType.BINARY)
        y_vars.append(y_idx)

    for i in range(num_facilities):
        for j in range(num_customers):
            trans_cost = float(rng.uniform(5.0, 25.0))
            x_idx = model.add_var(name=f"flow_{i}_{j}", lb=0.0, ub=float('inf'), obj=trans_cost, var_type=VarType.CONTINUOUS)
            x_vars[(i, j)] = x_idx

    # Demand satisfaction
    for j in range(num_customers):
        demand = float(rng.uniform(20.0, 50.0))
        row = {x_vars[(i, j)]: 1.0 for i in range(num_facilities)}
        model.add_constraint(row, ConstraintSense.EQ, demand, name=f"demand_cust_{j}")

    # Facility capacity limits
    for i in range(num_facilities):
        cap = float(rng.uniform(100.0, 200.0))
        row = {x_vars[(i, j)]: 1.0 for j in range(num_customers)}
        row[y_vars[i]] = -cap
        model.add_constraint(row, ConstraintSense.LE, 0.0, name=f"capacity_fac_{i}")

    return model


def build_scale_ladder_lp(num_rows: int = 1000, num_cols: int = 2000, density: float = 0.01) -> CanonicalModel:
    """
    Generates large-scale sparse LP benchmark model (NFR-SCALE, Gate B6).
    """
    model = CanonicalModel(name=f"Scale_Ladder_LP_{num_rows}x{num_cols}", sense=Sense.MAXIMIZE)
    rng = np.random.RandomState(2026)

    # Variables
    for j in range(num_cols):
        c_val = float(rng.uniform(1.0, 10.0))
        model.add_var(name=f"x_{j}", lb=0.0, ub=float(rng.uniform(10.0, 100.0)), obj=c_val, var_type=VarType.CONTINUOUS)

    # Sparse constraints
    for i in range(num_rows):
        nnz_row = max(2, int(num_cols * density))
        cols_chosen = rng.choice(num_cols, size=nnz_row, replace=False)
        row_dict = {}
        for c in cols_chosen:
            row_dict[int(c)] = float(rng.uniform(0.1, 5.0))
        rhs = float(rng.uniform(50.0, 500.0))
        model.add_constraint(row_dict, ConstraintSense.LE, rhs, name=f"row_{i}")

    return model
