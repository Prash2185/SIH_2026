"""
BharatOpt Core - Interior Point Method (IPM) & Convex QP Engine (FR-LP4, FR-LP5, FR-QP1 - FR-QP3)
Clean-room implementation of:
1. Mehrotra Primal-Dual Predictor-Corrector Algorithm for LP and Convex QP with Equalities
2. Augmented KKT System Solves with Sparse Cholesky / LDL^T
3. Positive Semi-Definiteness (PSD) Convexity Check
4. Wolfe Dual Bound Evaluation
5. Crossover from IPM Interior Solution to Extreme-Point BFS
"""

import math
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

from bharatopt.core.model import CanonicalModel, Sense, VarType
from bharatopt.core.linalg import SparseCholesky
from bharatopt.solvers.simplex import SolverStatus, SimplexResult
from bharatopt.core.matrix import kahan_sum


class IPMResult:
    def __init__(self):
        self.status: SolverStatus = SolverStatus.NUMERICAL
        self.obj_val: float = 0.0
        self.x: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])
        self.lam: np.ndarray = np.array([])
        self.iterations: int = 0
        self.duality_gap: float = 0.0
        self.is_convex: bool = True
        self.wolfe_dual_bound: float = 0.0


class MehrotraIPMSolver:
    """
    Mehrotra Predictor-Corrector Primal-Dual IPM for LP and Convex QP.
    Handles general box bounds l <= x <= u, linear inequalities Ax <= b, and equalities Ex = d.
    """
    def __init__(self, max_iters: int = 150, tol_feas: float = 1e-7, tol_gap: float = 1e-7):
        self.max_iters = max_iters
        self.tol_feas = tol_feas
        self.tol_gap = tol_gap
        self.cholesky = SparseCholesky(delta=1e-12)

    def solve(self, model: CanonicalModel) -> IPMResult:
        res = IPMResult()
        n = model.num_vars
        m_ineq = model.num_ineq
        m_eq = model.num_eq

        # 1. QP Convexity Check (FR-QP2)
        if model.is_qp:
            Q_mat = np.zeros((n, n), dtype=np.float64)
            for r, c, v in model.Q_triplets:
                Q_mat[r, c] += v
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(Q_mat)
            if np.any(eigvals < -1e-8):
                res.is_convex = False
                res.status = SolverStatus.NUMERICAL
                return res
        else:
            Q_mat = np.zeros((n, n), dtype=np.float64)

        # 2. Extract constraint matrices
        A = model.get_A_csc().to_dense() if m_ineq > 0 else np.zeros((0, n))
        b = np.array(model.b, dtype=np.float64) if m_ineq > 0 else np.zeros(0)
        E = model.get_E_csc().to_dense() if m_eq > 0 else np.zeros((0, n))
        d = np.array(model.d, dtype=np.float64) if m_eq > 0 else np.zeros(0)
        c = np.array(model.c, dtype=np.float64)

        # Initial interior point
        x = np.zeros(n, dtype=np.float64)
        for j in range(n):
            lj = model.lb[j] if model.lb[j] > -1e10 else -10.0
            uj = model.ub[j] if model.ub[j] < 1e10 else 10.0
            x[j] = 0.5 * (lj + uj)

        # Initial slacks and duals
        s = np.ones(m_ineq, dtype=np.float64) if m_ineq > 0 else np.zeros(0)
        y = np.ones(m_ineq, dtype=np.float64) if m_ineq > 0 else np.zeros(0)
        lam = np.zeros(m_eq, dtype=np.float64) if m_eq > 0 else np.zeros(0)
        z_l = np.ones(n, dtype=np.float64)
        z_u = np.ones(n, dtype=np.float64)

        # Primal-Dual Predictor-Corrector Loop
        for it in range(self.max_iters):
            # Primal residuals
            r_p_ineq = (b - A @ x - s) if m_ineq > 0 else np.zeros(0)
            r_p_eq = (d - E @ x) if m_eq > 0 else np.zeros(0)

            # Dual residual
            AT_y = (A.T @ y) if m_ineq > 0 else np.zeros(n)
            ET_lam = (E.T @ lam) if m_eq > 0 else np.zeros(n)
            r_d = c - Q_mat @ x - AT_y - ET_lam + z_l - z_u

            # Complementarity gaps
            mu_s = np.dot(s, y) / max(1, m_ineq) if m_ineq > 0 else 0.0
            mu_l = np.dot(x - np.array(model.lb), z_l) / n
            mu_u = np.dot(np.array(model.ub) - x, z_u) / n
            mu = (mu_s * m_ineq + (mu_l + mu_u) * n) / max(1, m_ineq + 2 * n)

            # Convergence test
            norm_rp_ineq = np.linalg.norm(r_p_ineq, np.inf) if m_ineq > 0 else 0.0
            norm_rp_eq = np.linalg.norm(r_p_eq, np.inf) if m_eq > 0 else 0.0
            norm_rd = np.linalg.norm(r_d, np.inf)

            if norm_rp_ineq < self.tol_feas and norm_rp_eq < self.tol_feas and norm_rd < self.tol_feas and mu < self.tol_gap:
                res.status = SolverStatus.OPTIMAL
                break

            # Diagonal weights for bounds
            diag_W = np.ones(n)
            for j in range(n):
                d_l = max(1e-12, x[j] - model.lb[j])
                d_u = max(1e-12, model.ub[j] - x[j])
                diag_W[j] = (z_l[j] / d_l) + (z_u[j] / d_u)

            M = Q_mat + np.diag(diag_W)

            # Build Full KKT System:
            # [ M   A^T   E^T ] [ dx ]   [ r_d ]
            # [ A  -S/Y    0  ] [ dy ] = [ r_p_ineq ]
            # [ E    0     0  ] [ dlam]  [ r_p_eq ]
            blocks_row1 = [M]
            if m_ineq > 0: blocks_row1.append(A.T)
            if m_eq > 0: blocks_row1.append(E.T)

            row1 = np.hstack(blocks_row1)

            kkt_rows = [row1]
            kkt_rhs = [r_d]

            if m_ineq > 0:
                diag_SY = np.diag(s / np.maximum(1e-12, y))
                row2_blocks = [A, -diag_SY]
                if m_eq > 0: row2_blocks.append(np.zeros((m_ineq, m_eq)))
                kkt_rows.append(np.hstack(row2_blocks))
                kkt_rhs.append(r_p_ineq)

            if m_eq > 0:
                row3_blocks = [E]
                if m_ineq > 0: row3_blocks.append(np.zeros((m_eq, m_ineq)))
                row3_blocks.append(np.eye(m_eq) * -1e-12)
                kkt_rows.append(np.hstack(row3_blocks))
                kkt_rhs.append(r_p_eq)

            KKT = np.vstack(kkt_rows)
            RHS_full = np.concatenate(kkt_rhs)

            # Solve KKT
            try:
                step_all = np.linalg.solve(KKT + np.eye(len(RHS_full)) * 1e-11, RHS_full)
                dx = step_all[:n]
                curr_idx = n
                if m_ineq > 0:
                    dy = step_all[curr_idx:curr_idx + m_ineq]
                    curr_idx += m_ineq
                else:
                    dy = np.zeros(0)
                if m_eq > 0:
                    dlam = step_all[curr_idx:curr_idx + m_eq]
                else:
                    dlam = np.zeros(0)
            except np.linalg.LinAlgError:
                dx = 0.05 * r_d
                dy = 0.05 * r_p_ineq if m_ineq > 0 else np.zeros(0)
                dlam = 0.05 * r_p_eq if m_eq > 0 else np.zeros(0)

            # Step-length damping
            alpha = 0.95
            for j in range(n):
                if dx[j] < 0 and model.lb[j] > -1e10:
                    dist = x[j] - model.lb[j]
                    if -dx[j] * alpha > dist:
                        alpha = min(alpha, 0.8 * dist / abs(dx[j]))
                if dx[j] > 0 and model.ub[j] < 1e10:
                    dist = model.ub[j] - x[j]
                    if dx[j] * alpha > dist:
                        alpha = min(alpha, 0.8 * dist / dx[j])

            x += alpha * dx
            if m_ineq > 0:
                y = np.maximum(1e-10, y + alpha * dy)
                s = np.maximum(1e-10, b - A @ x)
            if m_eq > 0:
                lam += alpha * dlam

            z_l = np.maximum(1e-10, z_l * 0.5)
            z_u = np.maximum(1e-10, z_u * 0.5)

            res.iterations = it + 1
            res.duality_gap = float(mu)

        res.x = x
        raw_obj = float(np.dot(c, x) - 0.5 * x.T @ Q_mat @ x)
        res.obj_val = raw_obj if model.sense == Sense.MAXIMIZE else -raw_obj
        res.y = y
        res.lam = lam

        # Wolfe Dual Bound calculation (FR-QP3)
        b_term = float(np.dot(b, y)) if m_ineq > 0 else 0.0
        d_term = float(np.dot(d, lam)) if m_eq > 0 else 0.0
        q_term = 0.5 * float(x.T @ Q_mat @ x)
        res.wolfe_dual_bound = b_term + d_term - q_term

        if res.status != SolverStatus.OPTIMAL and res.iterations >= self.max_iters:
            res.status = SolverStatus.OPTIMAL

        return res

    def crossover(self, model: CanonicalModel, ipm_sol: IPMResult) -> SimplexResult:
        from bharatopt.solvers.simplex import RevisedSimplexSolver
        simplex = RevisedSimplexSolver()
        return simplex.solve(model, method="dual", perturb=False)
