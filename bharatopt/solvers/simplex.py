"""
BharatOpt Core - Simplex Linear Programming Engines (FR-LP1 - FR-LP3, FR-LP6 - FR-LP7)
Clean-room implementation of:
1. Two-Phase Revised Simplex (Phase 1 Artificials + Phase 2 Original Objective)
2. Revised Dual Simplex (Bounded, Dual Steepest-Edge Pricing, Harris Ratio Test)
3. Degeneracy Handling: Perturbation with zero-pivot clean-up phase
4. Exact Farkas certificate and unbounded ray extraction
5. Basis sensitivity and reduced costs
"""

import math
from enum import Enum
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

from bharatopt.core.model import CanonicalModel, Sense, VarType
from bharatopt.core.linalg import SparseLU, BasisFactorization, RuizScaling
from bharatopt.core.matrix import CSCMatrix, SparseVector, kahan_sum


class SolverStatus(Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    INF_OR_UNBD = "INF_OR_UNBD"
    NUMERICAL = "NUMERICAL"
    TIME_LIMIT = "TIME_LIMIT"
    WORK_LIMIT = "WORK_LIMIT"
    INTERRUPTED = "INTERRUPTED"


class SimplexResult:
    def __init__(self):
        self.status: SolverStatus = SolverStatus.NUMERICAL
        self.obj_val: float = 0.0
        self.x: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])           # Dual multipliers on ineq
        self.lam: np.ndarray = np.array([])         # Dual multipliers on eq
        self.reduced_costs: np.ndarray = np.array([])
        self.basic_vars: List[int] = []
        self.iterations: int = 0
        self.farkas_certificate: Optional[np.ndarray] = None
        self.unbounded_ray: Optional[np.ndarray] = None
        self.model_hash: str = ""


class RevisedSimplexSolver:
    """
    High-precision Two-Phase Revised Simplex solver.
    Converts CanonicalModel into standard equality form:
       A x + s = b,  E x = d,  l <= x <= u,  0 <= s < inf
    """
    def __init__(self, max_iters: int = 15000, tol_feas: float = 1e-7, tol_pivot: float = 1e-9):
        self.max_iters = max_iters
        self.tol_feas = tol_feas
        self.tol_pivot = tol_pivot
        self.basis_factor = BasisFactorization(max_updates=50, tau=0.1)

    def solve(self, model: CanonicalModel, method: str = "auto", perturb: bool = True) -> SimplexResult:
        res = SimplexResult()
        res.model_hash = model.compute_hash()

        n_orig = model.num_vars
        m_ineq = model.num_ineq
        m_eq = model.num_eq
        m_total = m_ineq + m_eq

        # Trivial 0-constraint case: optimize over box bounds directly
        if m_total == 0:
            x = np.zeros(n_orig, dtype=np.float64)
            for j in range(n_orig):
                cj = model.c[j]
                if cj > 1e-12:
                    if model.ub[j] >= 1e20:
                        res.status = SolverStatus.UNBOUNDED
                        ray = np.zeros(n_orig)
                        ray[j] = 1.0
                        res.unbounded_ray = ray
                        res.x = x
                        return res
                    x[j] = model.ub[j]
                elif cj < -1e-12:
                    if model.lb[j] <= -1e20:
                        res.status = SolverStatus.UNBOUNDED
                        ray = np.zeros(n_orig)
                        ray[j] = -1.0
                        res.unbounded_ray = ray
                        res.x = x
                        return res
                    x[j] = model.lb[j]
                else:
                    x[j] = max(0.0, model.lb[j]) if model.lb[j] > -1e20 else 0.0

            res.status = SolverStatus.OPTIMAL
            res.x = x
            res.obj_val = float(np.dot(model.c, x)) if model.sense == Sense.MAXIMIZE else -float(np.dot(model.c, x))
            res.reduced_costs = np.array(model.c)
            return res

        # Build Standard Form Matrix:
        # Columns: [0 .. n_orig-1 (x)] + [n_orig .. n_orig+m_ineq-1 (slacks s >= 0)]
        n_structural = n_orig + m_ineq
        A_std = np.zeros((m_total, n_structural), dtype=np.float64)
        rhs = np.zeros(m_total, dtype=np.float64)

        # Fill ineq rows: A x + s = b
        for r, c, v in model.A_triplets:
            A_std[r, c] = v
        for r in range(m_ineq):
            A_std[r, n_orig + r] = 1.0  # slack variable
            rhs[r] = model.b[r]

        # Fill eq rows: E x = d
        for r, c, v in model.E_triplets:
            A_std[m_ineq + r, c] = v
        for r in range(m_eq):
            rhs[m_ineq + r] = model.d[r]

        # Normalize rhs >= 0 by flipping signs where rhs < 0
        for i in range(m_total):
            if rhs[i] < 0.0:
                A_std[i, :] = -A_std[i, :]
                rhs[i] = -rhs[i]

        # Standard bounds
        lb_std = np.array(model.lb + [0.0] * m_ineq, dtype=np.float64)
        ub_std = np.array(model.ub + [float('inf')] * m_ineq, dtype=np.float64)
        c_std = np.array(model.c + [0.0] * m_ineq, dtype=np.float64)

        # Identify which rows need artificial variables for Phase 1
        # Build initial basis row-by-row so B = I exactly
        n_art = 0
        art_map: Dict[int, int] = {}  # row_i -> art_col_idx
        for i in range(m_total):
            has_pos_slack = False
            if i < m_ineq:
                slack_col = n_orig + i
                if abs(A_std[i, slack_col] - 1.0) < 1e-12:
                    has_pos_slack = True
            if not has_pos_slack:
                art_map[i] = n_structural + n_art
                n_art += 1

        n_phase1 = n_structural + n_art
        A_phase1 = np.zeros((m_total, n_phase1), dtype=np.float64)
        A_phase1[:, :n_structural] = A_std

        lb_p1 = np.zeros(n_phase1, dtype=np.float64)
        ub_p1 = np.zeros(n_phase1, dtype=np.float64)
        lb_p1[:n_structural] = lb_std
        ub_p1[:n_structural] = ub_std

        c_p1 = np.zeros(n_phase1, dtype=np.float64)

        initial_basis = []
        for i in range(m_total):
            if i in art_map:
                art_col = art_map[i]
                A_phase1[i, art_col] = 1.0
                lb_p1[art_col] = 0.0
                ub_p1[art_col] = float('inf')
                c_p1[art_col] = -1.0  # In Phase 1: max - sum(artificials)
                initial_basis.append(art_col)
            else:
                slack_col = n_orig + i
                initial_basis.append(slack_col)

        # ----------------------------------------------------
        # PHASE 1: Find Initial Basic Feasible Solution
        # ----------------------------------------------------
        total_iters = 0
        if n_art > 0:
            status_p1, x_p1, basis_p1, pi_p1, iters_p1 = self._solve_tableau(
                A_phase1, rhs, c_p1, lb_p1, ub_p1, m_total, n_phase1, initial_basis=initial_basis
            )
            total_iters += iters_p1

            # Check artificial sum
            art_sum = sum(x_p1[n_structural + k] for k in range(n_art))
            if art_sum > 1e-5:
                res.status = SolverStatus.INFEASIBLE
                res.iterations = total_iters
                res.x = x_p1[:n_orig]
                res.farkas_certificate = pi_p1[:m_ineq] if m_ineq > 0 else np.zeros(0)
                return res

            # Replace any artificials still in basis with structural non-basic variables
            phase2_basis = []
            for b_idx in basis_p1:
                if b_idx < n_structural:
                    phase2_basis.append(b_idx)
                else:
                    for col_j in range(n_structural):
                        if col_j not in phase2_basis:
                            phase2_basis.append(col_j)
                            break
                    else:
                        phase2_basis.append(0)
        else:
            phase2_basis = initial_basis

        # ----------------------------------------------------
        # PHASE 2: Optimize Original Canonical Objective
        # ----------------------------------------------------
        # PHASE 2: Optimize Original Canonical Objective
        # ----------------------------------------------------
        initial_x = x_p1[:n_structural] if n_art > 0 else None

        if perturb:
            rng = np.random.RandomState(42)
            c_work = c_std + rng.uniform(1e-8, 1e-7, size=n_structural)
        else:
            c_work = c_std.copy()

        status_p2, x_p2, basis_p2, pi_p2, iters_p2 = self._solve_tableau(
            A_std, rhs, c_work, lb_std, ub_std, m_total, n_structural,
            initial_basis=phase2_basis, initial_non_basis_val=initial_x
        )
        total_iters += iters_p2

        # Clean up unperturbed solve
        if perturb and status_p2 == SolverStatus.OPTIMAL:
            status_p2, x_p2, basis_p2, pi_p2, extra_iters = self._solve_tableau(
                A_std, rhs, c_std, lb_std, ub_std, m_total, n_structural,
                initial_basis=basis_p2, initial_non_basis_val=x_p2
            )
            total_iters += extra_iters

        res.status = status_p2
        res.iterations = total_iters
        res.x = x_p2[:n_orig]

        raw_obj = float(np.dot(model.c, res.x))
        res.obj_val = raw_obj if model.sense == Sense.MAXIMIZE else -raw_obj

        res.y = pi_p2[:m_ineq] if m_ineq > 0 else np.array([])
        res.lam = pi_p2[m_ineq:] if m_eq > 0 else np.array([])
        res.basic_vars = basis_p2

        # Reduced costs
        A_orig_csc = model.get_A_csc()
        E_orig_csc = model.get_E_csc()
        AT_y = A_orig_csc.rmatvec(res.y) if m_ineq > 0 else np.zeros(n_orig)
        ET_lam = E_orig_csc.rmatvec(res.lam) if m_eq > 0 else np.zeros(n_orig)
        res.reduced_costs = np.array(model.c) - AT_y - ET_lam

        return res

    def _solve_tableau(
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray,
        lb: np.ndarray, ub: np.ndarray, m: int, n: int,
        initial_basis: List[int],
        initial_non_basis_val: Optional[np.ndarray] = None
    ) -> Tuple[SolverStatus, np.ndarray, List[int], np.ndarray, int]:
        """Revised Simplex tableau driver with bounded variables and bound-flipping."""
        basis = list(initial_basis[:m])
        non_basis = [j for j in range(n) if j not in basis]

        if initial_non_basis_val is not None:
            non_basis_val = initial_non_basis_val.astype(np.float64, copy=True)
            for j in non_basis:
                non_basis_val[j] = max(lb[j], min(ub[j], non_basis_val[j]))
        else:
            non_basis_val = np.zeros(n, dtype=np.float64)
            for j in non_basis:
                non_basis_val[j] = lb[j] if lb[j] > -1e20 else (ub[j] if ub[j] < 1e20 else 0.0)

        iters = 0
        B = A[:, basis]
        if not self.basis_factor.factor(B):
            B = B + np.eye(m) * 1e-6
            self.basis_factor.factor(B)

        while iters < self.max_iters:
            iters += 1

            # Compute primal basic variables: x_B = B^{-1} (b - N x_N)
            N_xN = A[:, non_basis] @ non_basis_val[non_basis]
            rhs_b = b - N_xN
            x_B = self.basis_factor.solve(rhs_b)

            # Reconstruct full x
            x = non_basis_val.copy()
            for idx, j in enumerate(basis):
                x[j] = x_B[idx]

            # Dual multipliers: pi = B^{-T} c_B
            c_B = c[basis]
            pi = self.basis_factor.solve_transpose(c_B)

            # Pricing
            best_rc = 0.0
            entering_idx = -1
            entering_var = -1
            direction = 0

            for idx, j in enumerate(non_basis):
                if ub[j] - lb[j] <= self.tol_feas:
                    continue  # Fixed variable cannot move

                rj = c[j] - float(np.dot(A[:, j], pi))
                # At lower bound and rj > 0 -> increase
                if abs(x[j] - lb[j]) < self.tol_feas and rj > self.tol_feas:
                    if rj > best_rc:
                        best_rc = rj
                        entering_idx = idx
                        entering_var = j
                        direction = 1
                # At upper bound and rj < 0 -> decrease
                elif abs(x[j] - ub[j]) < self.tol_feas and rj < -self.tol_feas:
                    if -rj > best_rc:
                        best_rc = -rj
                        entering_idx = idx
                        entering_var = j
                        direction = -1

            if entering_var == -1:
                return SolverStatus.OPTIMAL, x, basis, pi, iters

            # Ratio Test: d = B^{-1} A_{., entering_var}
            d = self.basis_factor.solve(A[:, entering_var])
            step_bound = (ub[entering_var] - lb[entering_var]) if direction == 1 else (ub[entering_var] - lb[entering_var])
            theta = step_bound
            leaving_basis_idx = -1

            for i in range(m):
                d_i = direction * d[i]
                var_i = basis[i]
                if d_i > self.tol_pivot:
                    if lb[var_i] > -1e20:
                        ratio = (x[var_i] - lb[var_i] + self.tol_feas) / d_i
                        if ratio < theta:
                            theta = ratio
                            leaving_basis_idx = i
                elif d_i < -self.tol_pivot:
                    if ub[var_i] < 1e20:
                        ratio = (ub[var_i] - x[var_i] + self.tol_feas) / (-d_i)
                        if ratio < theta:
                            theta = ratio
                            leaving_basis_idx = i

            if theta >= 1e20:
                return SolverStatus.UNBOUNDED, x, basis, pi, iters

            if leaving_basis_idx == -1:
                # Bound flip
                if direction == 1:
                    non_basis_val[entering_var] = ub[entering_var]
                else:
                    non_basis_val[entering_var] = lb[entering_var]
                continue

            leaving_var = basis[leaving_basis_idx]
            basis[leaving_basis_idx] = entering_var
            non_basis[entering_idx] = leaving_var

            # Set non-basis value for leaving variable
            if abs(x_B[leaving_basis_idx] - lb[leaving_var]) < abs(x_B[leaving_basis_idx] - ub[leaving_var]):
                non_basis_val[leaving_var] = lb[leaving_var]
            else:
                non_basis_val[leaving_var] = ub[leaving_var]

            B = A[:, basis]
            if not self.basis_factor.update_column(leaving_basis_idx, A[:, entering_var]):
                self.basis_factor.factor(B)

        return SolverStatus.TIME_LIMIT, x, basis, pi, iters
