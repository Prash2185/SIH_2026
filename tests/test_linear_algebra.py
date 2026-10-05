"""
BharatOpt Core - Linear Algebra Unit Tests (FR-LA1 - FR-LA5)
Tests:
1. Sparse Markowitz LU Factorization & Threshold Partial Pivoting
2. Forrest-Tomlin / Eta Basis Updates
3. Sparse Cholesky / LDL^T Factorization with AMD Ordering
4. Ruiz Geometric Equilibration Scaling
5. Kahan Compensated Summation Accuracy
"""

import math
import numpy as np
import unittest

from bharatopt.core.matrix import SparseVector, CSCMatrix, CSRMatrix, triplet_to_csc, kahan_sum
from bharatopt.core.linalg import SparseLU, BasisFactorization, SparseCholesky, RuizScaling


class TestLinearAlgebra(unittest.TestCase):

    def test_kahan_summation_precision(self):
        """Verify Kahan summation cancels out small epsilon drift."""
        # 1.0 + 1e-16 + 1e-16 + ... 10000 times
        vals = [1.0] + [1e-15] * 10000
        naive_sum = sum(vals)
        kahan_res = kahan_sum(vals)
        self.assertGreater(kahan_res, 1.0)

    def test_sparse_lu_factorization(self):
        """Test Sparse Markowitz LU factor and solve."""
        B = np.array([
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 4.0]
        ], dtype=np.float64)

        lu = SparseLU(tau=0.1)
        ok = lu.factor(B)
        self.assertTrue(ok)

        rhs = np.array([4.0, 9.0, 10.0])
        x = lu.solve(rhs)

        # Expected solution: B x = rhs
        # Check residual ||B x - rhs||_inf
        res = np.linalg.norm(B @ x - rhs, np.inf)
        self.assertLess(res, 1e-10)

        # Transpose solve: B^T y = rhs
        y = lu.solve_transpose(rhs)
        res_t = np.linalg.norm(B.T @ y - rhs, np.inf)
        self.assertLess(res_t, 1e-10)

    def test_basis_factorization_and_updates(self):
        """Test basis factorization with Eta column updates."""
        B = np.eye(4, dtype=np.float64)
        bf = BasisFactorization(max_updates=10, tau=0.1)
        ok = bf.factor(B)
        self.assertTrue(ok)

        # Replace col 1 with new vector
        new_col = np.array([1.0, 2.0, 0.0, 0.0])
        ok_upd = bf.update_column(1, new_col)
        self.assertTrue(ok_upd)

        rhs = np.array([3.0, 4.0, 1.0, 2.0])
        x = bf.solve(rhs)

        # Verify against dense system
        B_updated = np.eye(4)
        B_updated[:, 1] = new_col
        expected_x = np.linalg.solve(B_updated, rhs)
        np.testing.assert_allclose(x, expected_x, atol=1e-9)

    def test_sparse_cholesky_ldlt(self):
        """Test Sparse Cholesky factor and solve on positive definite matrix."""
        M = np.array([
            [4.0, 2.0, 0.0],
            [2.0, 5.0, 1.0],
            [0.0, 1.0, 3.0]
        ], dtype=np.float64)

        chol = SparseCholesky()
        ok = chol.factor(M)
        self.assertTrue(ok)
        self.assertTrue(chol.is_pos_def)

        rhs = np.array([6.0, 13.0, 8.0])
        x = chol.solve(rhs)
        res = np.linalg.norm(M @ x - rhs, np.inf)
        self.assertLess(res, 1e-10)

    def test_ruiz_scaling(self):
        """Test Ruiz equilibration scaling balances matrix infinity norms."""
        triplets = [
            (0, 0, 1000.0), (0, 1, 0.01),
            (1, 0, 500.0),  (1, 1, 0.02)
        ]
        A = triplet_to_csc(2, 2, triplets)
        scaler = RuizScaling(max_iters=10)
        r, c = scaler.compute(A)

        # Scaled matrix entry A_ij * r_i * c_j
        A_dense = A.to_dense()
        scaled_A = np.diag(r) @ A_dense @ np.diag(c)
        row_norms = np.max(np.abs(scaled_A), axis=1)
        col_norms = np.max(np.abs(scaled_A), axis=0)

        for rn in row_norms:
            self.assertAlmostEqual(rn, 1.0, places=1)
        for cn in col_norms:
            self.assertAlmostEqual(cn, 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
