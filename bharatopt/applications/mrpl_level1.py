"""
BharatOpt Core - MRPL Level 1 Monthly Crude Selection & Blending Formulation (FR-APP1, §8.1)
Implements:
1. Multi-crude basket (250+ crude database capability, synthetic assay calibration)
2. CDUs (Crude Distillation Units 1, 2, 3) with throughput bounds
3. Non-linear API Gravity conversion to linear Specific Gravity: SG_c_max = 141.5 / (131.5 + API_min)
4. Mass-weighted sulfur balance with density weighting: sum_j (s_j - S_c_max) * rho_j * x_jc <= 0
5. Spot cargo purchase decisions (Binary y_j) and commercial product yields (LPG, MS, HSD, ATF, FO, Bitumen)
6. Carbon emissions proxy constraint
"""

from typing import List, Dict, Tuple, Any
from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense


def build_mrpl_level1_model(
    crude_data: List[Dict[str, Any]],
    cdu_capacities: List[Dict[str, Any]],
    product_prices: Dict[str, float],
    product_demands: Dict[str, Tuple[float, float]],
    carbon_cap: float = 500000.0
) -> CanonicalModel:
    """
    Builds canonical MILP for MRPL Level 1 Monthly Crude Selection & Blending.
    """
    model = CanonicalModel(name="MRPL_Monthly_Crude_Selection_L1", sense=Sense.MAXIMIZE)

    # 1. Variables
    # x_{j, c}: volume of crude j to CDU c (kbbl)
    # s_j: closing inventory of crude j (kbbl)
    # y_j: binary decision for spot cargo purchase of crude j
    # w_p: finished product production (kbbl)

    x_vars: Dict[Tuple[str, str], int] = {}
    s_vars: Dict[str, int] = {}
    y_vars: Dict[str, int] = {}
    w_vars: Dict[str, int] = {}

    for crude in crude_data:
        c_name = crude["name"]
        cost = crude.get("cost_per_bbl", 75.0)
        holding_cost = crude.get("holding_cost_per_bbl", 0.5)

        # Spot decision
        if crude.get("is_spot", False):
            y_idx = model.add_var(
                name=f"spot_buy_{c_name}",
                lb=0.0, ub=1.0,
                obj=-crude.get("fixed_cargo_cost", 10000.0),
                var_type=VarType.BINARY
            )
            y_vars[c_name] = y_idx

        # Closing stock
        s_idx = model.add_var(
            name=f"closing_stock_{c_name}",
            lb=0.0, ub=crude.get("max_storage", 500.0),
            obj=-holding_cost,
            var_type=VarType.CONTINUOUS
        )
        s_vars[c_name] = s_idx

        # Processing in CDUs
        for cdu in cdu_capacities:
            cdu_name = cdu["name"]
            # Allowability check (e.g. TAN or metallurgy)
            if crude.get("high_tan", False) and not cdu.get("tan_metallurgy", False):
                continue

            op_cost = cdu.get("op_cost_per_bbl", 2.0)
            x_idx = model.add_var(
                name=f"charge_{c_name}_{cdu_name}",
                lb=0.0, ub=cdu.get("max_capacity", 300.0),
                obj=-(cost + op_cost),
                var_type=VarType.CONTINUOUS
            )
            x_vars[(c_name, cdu_name)] = x_idx

    # Finished Product Variables
    for prod_name, price in product_prices.items():
        min_dem, max_dem = product_demands.get(prod_name, (0.0, 10000.0))
        w_idx = model.add_var(
            name=f"prod_{prod_name}",
            lb=min_dem, ub=max_dem,
            obj=price,
            var_type=VarType.CONTINUOUS
        )
        w_vars[prod_name] = w_idx

    # 2. Constraints

    # (A) Crude Stock Balance: sum_c x_{j,c} + s_j = s_j^0 + q_j + A_j * y_j
    for crude in crude_data:
        c_name = crude["name"]
        row: Dict[int, float] = {}
        for cdu in cdu_capacities:
            key = (c_name, cdu["name"])
            if key in x_vars:
                row[x_vars[key]] = 1.0
        row[s_vars[c_name]] = 1.0

        if c_name in y_vars:
            cargo_size = crude.get("cargo_size", 500.0)
            row[y_vars[c_name]] = -cargo_size

        init_stock = crude.get("initial_stock", 100.0)
        pipeline_receipts = crude.get("pipeline_receipts", 0.0)
        rhs = init_stock + pipeline_receipts

        model.add_constraint(row, ConstraintSense.EQ, rhs, name=f"stock_bal_{c_name}")

    # (B) CDU Throughput Capacity: K_c_min <= sum_j x_{j,c} <= K_c_max
    for cdu in cdu_capacities:
        cdu_name = cdu["name"]
        row = {x_vars[(c["name"], cdu_name)]: 1.0 for c in crude_data if (c["name"], cdu_name) in x_vars}
        model.add_constraint(row, ConstraintSense.LE, cdu.get("max_capacity", 300.0), name=f"cdu_max_{cdu_name}")
        model.add_constraint(row, ConstraintSense.GE, cdu.get("min_capacity", 50.0), name=f"cdu_min_{cdu_name}")

    # (C) Finished Product Yields: w_p = sum_{j, c} eta_{p, j} * x_{j, c}
    for prod_name in product_prices.keys():
        row: Dict[int, float] = {w_vars[prod_name]: 1.0}
        for crude in crude_data:
            c_name = crude["name"]
            yield_val = crude.get("yields", {}).get(prod_name, 0.15)
            for cdu in cdu_capacities:
                key = (c_name, cdu["name"])
                if key in x_vars:
                    row[x_vars[key]] = -yield_val

        model.add_constraint(row, ConstraintSense.EQ, 0.0, name=f"yield_bal_{prod_name}")

    # (D) Mass-Weighted Sulfur Limit per CDU (A2)
    # sum_j (s_j - S_c_max) * rho_j * x_{j, c} <= 0
    for cdu in cdu_capacities:
        cdu_name = cdu["name"]
        s_max = cdu.get("max_sulfur_wt_pct", 1.5)
        row = {}
        for crude in crude_data:
            c_name = crude["name"]
            key = (c_name, cdu_name)
            if key in x_vars:
                s_crude = crude.get("sulfur_wt_pct", 1.0)
                rho = crude.get("density_kg_l", 0.85)
                row[x_vars[key]] = (s_crude - s_max) * rho
        model.add_constraint(row, ConstraintSense.LE, 0.0, name=f"sulfur_limit_{cdu_name}")

    # (E) Specific Gravity / Non-linear API Gravity Limit per CDU (A3)
    # SG_c_max = 141.5 / (131.5 + API_c_min)
    # sum_j (SG_j - SG_c_max) * x_{j, c} <= 0
    for cdu in cdu_capacities:
        cdu_name = cdu["name"]
        api_min = cdu.get("min_api_gravity", 31.0)
        sg_max = 141.5 / (131.5 + api_min)
        row = {}
        for crude in crude_data:
            c_name = crude["name"]
            key = (c_name, cdu_name)
            if key in x_vars:
                api_crude = crude.get("api_gravity", 34.0)
                sg_crude = 141.5 / (131.5 + api_crude)
                row[x_vars[key]] = (sg_crude - sg_max)
        model.add_constraint(row, ConstraintSense.LE, 0.0, name=f"gravity_limit_{cdu_name}")

    # (F) Feed Carbon Emissions Proxy
    emiss_row = {}
    for crude in crude_data:
        c_name = crude["name"]
        ef = crude.get("emission_factor_kg_bbl", 12.5)
        for cdu in cdu_capacities:
            key = (c_name, cdu["name"])
            if key in x_vars:
                emiss_row[x_vars[key]] = ef
    model.add_constraint(emiss_row, ConstraintSense.LE, carbon_cap, name="carbon_emissions_cap")

    return model
