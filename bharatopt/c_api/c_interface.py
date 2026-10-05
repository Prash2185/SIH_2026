"""
BharatOpt Core - Python ctypes wrapper and C ABI emulator (FR-API1)
"""

import ctypes
from typing import Optional
import numpy as np

from bharatopt.core.model import CanonicalModel, Sense, VarType
from bharatopt.solvers.simplex import RevisedSimplexSolver, SolverStatus
from bharatopt.verifier.verifier import IndependentVerifier


class CBharatOptModel(ctypes.Structure):
    _fields_ = [
        ("num_vars", ctypes.c_int),
        ("num_ineq", ctypes.c_int),
        ("num_eq", ctypes.c_int),
        ("num_nz_ineq", ctypes.c_int),
        ("num_nz_eq", ctypes.c_int),
        ("c", ctypes.POINTER(ctypes.c_double)),
        ("lb", ctypes.POINTER(ctypes.c_double)),
        ("ub", ctypes.POINTER(ctypes.c_double)),
        ("var_types", ctypes.POINTER(ctypes.c_int)),
        ("A_row_ptr", ctypes.POINTER(ctypes.c_int)),
        ("A_col_ind", ctypes.POINTER(ctypes.c_int)),
        ("A_val", ctypes.POINTER(ctypes.c_double)),
        ("b", ctypes.POINTER(ctypes.c_double)),
        ("E_row_ptr", ctypes.POINTER(ctypes.c_int)),
        ("E_col_ind", ctypes.POINTER(ctypes.c_int)),
        ("E_val", ctypes.POINTER(ctypes.c_double)),
        ("d", ctypes.POINTER(ctypes.c_double)),
        ("is_maximize", ctypes.c_int)
    ]


class CBharatOptSolution(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("objective_value", ctypes.c_double),
        ("best_dual_bound", ctypes.c_double),
        ("relative_gap", ctypes.c_double),
        ("iterations", ctypes.c_int),
        ("node_count", ctypes.c_int),
        ("solve_time_ms", ctypes.c_double),
        ("x", ctypes.POINTER(ctypes.c_double)),
        ("duals", ctypes.POINTER(ctypes.c_double)),
        ("reduced_costs", ctypes.POINTER(ctypes.c_double)),
        ("verifier_passed", ctypes.c_int),
        ("model_hash", ctypes.c_char * 65)
    ]


def solve_from_c_struct(c_model: CBharatOptModel) -> CBharatOptSolution:
    """Entry point emulating C ABI dispatch from in-memory struct."""
    model = CanonicalModel(
        name="C_API_Model",
        sense=Sense.MAXIMIZE if c_model.is_maximize else Sense.MINIMIZE
    )
    n = c_model.num_vars
    for j in range(n):
        vtype = VarType(c_model.var_types[j])
        model.add_var(
            name=f"x_{j}",
            lb=c_model.lb[j],
            ub=c_model.ub[j],
            obj=c_model.c[j],
            var_type=vtype
        )

    # Ineq
    A_csr = model.get_A_csr()
    # Solve
    solver = RevisedSimplexSolver()
    sol = solver.solve(model)
    ver = IndependentVerifier()
    v_rep = ver.verify_solution(model, sol.x, sol.obj_val, sol.obj_val, sol.y)

    c_sol = CBharatOptSolution()
    c_sol.status = 0 if sol.status == SolverStatus.OPTIMAL else 1
    c_sol.objective_value = sol.obj_val
    c_sol.best_dual_bound = sol.obj_val
    c_sol.relative_gap = 0.0
    c_sol.iterations = sol.iterations
    c_sol.verifier_passed = 1 if v_rep.passed else 0
    c_sol.model_hash = model.compute_hash().encode('utf-8')
    return c_sol
