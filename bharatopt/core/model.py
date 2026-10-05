"""
BharatOpt Core - Canonical Model & Mathematical Representation (FR-IO1 - FR-IO4, §7.1 - §7.3)
Canonical form:
  max  c^T x - 0.5 * x^T Q x
  s.t. A x <= b   (dual y >= 0)
       E x == d   (dual lambda free)
       l <= x <= u (x_j integer for j in I)
"""

import hashlib
import json
import math
from enum import Enum
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

from bharatopt.core.matrix import CSCMatrix, CSRMatrix, SparseVector, triplet_to_csc, triplet_to_csr, kahan_sum


class VarType(Enum):
    CONTINUOUS = 0
    INTEGER = 1
    BINARY = 2


class Sense(Enum):
    MAXIMIZE = 1
    MINIMIZE = -1


class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "=="


class CanonicalModel:
    """
    Unified canonical mathematical representation for LP, QP, and MILP problems.
    Supports continuous, integer, binary variables, box bounds, quadratic objectives,
    deterministic hashing, and valid dual bound evaluation.
    """
    def __init__(self, name: str = "BharatOptModel", sense: Sense = Sense.MAXIMIZE):
        self.name = name
        self.sense = sense  # User's original objective sense
        self.var_names: List[str] = []
        self.var_name_to_idx: Dict[str, int] = {}
        self.var_types: List[VarType] = []
        self.lb: List[float] = []
        self.ub: List[float] = []
        self.c: List[float] = []  # In canonical MAX sense: c_canon = c if MAX else -c

        # Quadratic objective Q matrix in 0.5 * x^T Q x (positive semi-definite for convex min, negative semi-definite for max)
        self.Q_triplets: List[Tuple[int, int, float]] = []

        # Inequality constraints A x <= b
        self.row_names_ineq: List[str] = []
        self.A_triplets: List[Tuple[int, int, float]] = []
        self.b: List[float] = []

        # Equality constraints E x == d
        self.row_names_eq: List[str] = []
        self.E_triplets: List[Tuple[int, int, float]] = []
        self.d: List[float] = []

        # SOS and Indicator constraints (FR-IO2)
        self.sos1_sets: List[List[int]] = []
        self.sos2_sets: List[List[int]] = []

        # Cached compiled matrices
        self._A_csc: Optional[CSCMatrix] = None
        self._A_csr: Optional[CSRMatrix] = None
        self._E_csc: Optional[CSCMatrix] = None
        self._E_csr: Optional[CSRMatrix] = None
        self._Q_csc: Optional[CSCMatrix] = None

    @property
    def num_vars(self) -> int:
        return len(self.var_names)

    @property
    def num_ineq(self) -> int:
        return len(self.b)

    @property
    def num_eq(self) -> int:
        return len(self.d)

    @property
    def num_constraints(self) -> int:
        return self.num_ineq + self.num_eq

    @property
    def integer_indices(self) -> List[int]:
        return [j for j, vt in enumerate(self.var_types) if vt in (VarType.INTEGER, VarType.BINARY)]

    @property
    def is_qp(self) -> bool:
        return len(self.Q_triplets) > 0

    @property
    def is_mip(self) -> bool:
        return len(self.integer_indices) > 0

    def add_var(self, name: Optional[str] = None, lb: float = 0.0, ub: float = float('inf'),
                obj: float = 0.0, var_type: VarType = VarType.CONTINUOUS) -> int:
        idx = len(self.var_names)
        if name is None:
            name = f"x_{idx}"
        if name in self.var_name_to_idx:
            raise ValueError(f"Variable with name '{name}' already exists.")

        if var_type == VarType.BINARY:
            lb = max(0.0, lb)
            ub = min(1.0, ub)

        self.var_names.append(name)
        self.var_name_to_idx[name] = idx
        self.var_types.append(var_type)
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        # Internal canonical objective is always MAXIMIZE
        canon_obj = float(obj) if self.sense == Sense.MAXIMIZE else -float(obj)
        self.c.append(canon_obj)
        self._invalidate_cache()
        return idx

    def set_quadratic_term(self, i: int, j: int, val: float):
        """Set term in 0.5 * x^T Q x."""
        canon_val = float(val)
        self.Q_triplets.append((i, j, canon_val))
        if i != j:
            self.Q_triplets.append((j, i, canon_val))
        self._invalidate_cache()

    def add_constraint(self, row_entries: Dict[int, float], sense: ConstraintSense, rhs: float,
                       name: Optional[str] = None) -> int:
        rhs_val = float(rhs)
        if sense == ConstraintSense.LE:
            row_idx = len(self.b)
            if name is None:
                name = f"c_le_{row_idx}"
            self.row_names_ineq.append(name)
            self.b.append(rhs_val)
            for col_idx, coeff in row_entries.items():
                if abs(coeff) > 1e-15:
                    self.A_triplets.append((row_idx, col_idx, float(coeff)))
            self._invalidate_cache()
            return row_idx
        elif sense == ConstraintSense.GE:
            # Multiply by -1 to convert to <=
            row_idx = len(self.b)
            if name is None:
                name = f"c_ge_{row_idx}"
            self.row_names_ineq.append(name)
            self.b.append(-rhs_val)
            for col_idx, coeff in row_entries.items():
                if abs(coeff) > 1e-15:
                    self.A_triplets.append((row_idx, col_idx, -float(coeff)))
            self._invalidate_cache()
            return row_idx
        elif sense == ConstraintSense.EQ:
            row_idx = len(self.d)
            if name is None:
                name = f"c_eq_{row_idx}"
            self.row_names_eq.append(name)
            self.d.append(rhs_val)
            for col_idx, coeff in row_entries.items():
                if abs(coeff) > 1e-15:
                    self.E_triplets.append((row_idx, col_idx, float(coeff)))
            self._invalidate_cache()
            return row_idx
        else:
            raise ValueError(f"Unknown constraint sense: {sense}")

    def _invalidate_cache(self):
        self._A_csc = None
        self._A_csr = None
        self._E_csc = None
        self._E_csr = None
        self._Q_csc = None

    def get_A_csc(self) -> CSCMatrix:
        if self._A_csc is None:
            self._A_csc = triplet_to_csc(self.num_ineq, self.num_vars, self.A_triplets)
        return self._A_csc

    def get_A_csr(self) -> CSRMatrix:
        if self._A_csr is None:
            self._A_csr = triplet_to_csr(self.num_ineq, self.num_vars, self.A_triplets)
        return self._A_csr

    def get_E_csc(self) -> CSCMatrix:
        if self._E_csc is None:
            self._E_csc = triplet_to_csc(self.num_eq, self.num_vars, self.E_triplets)
        return self._E_csc

    def get_E_csr(self) -> CSRMatrix:
        if self._E_csr is None:
            self._E_csr = triplet_to_csr(self.num_eq, self.num_vars, self.E_triplets)
        return self._E_csr

    def get_Q_csc(self) -> CSCMatrix:
        if self._Q_csc is None:
            self._Q_csc = triplet_to_csc(self.num_vars, self.num_vars, self.Q_triplets)
        return self._Q_csc

    def compute_hash(self) -> str:
        """
        FR-IO3: Deterministic SHA-256 model hash.
        Guarantees that identical model definitions generate the exact same fingerprint.
        """
        hasher = hashlib.sha256()
        header = f"vars:{self.num_vars};ineq:{self.num_ineq};eq:{self.num_eq};sense:{self.sense.name}"
        hasher.update(header.encode('utf-8'))

        # Variables, bounds, obj
        for i in range(self.num_vars):
            s = f"v:{self.var_names[i]}:{self.var_types[i].value}:{self.lb[i]:.10e}:{self.ub[i]:.10e}:{self.c[i]:.10e};"
            hasher.update(s.encode('utf-8'))

        # Sorted triplets A
        for r, c, v in sorted(self.A_triplets):
            hasher.update(f"A:{r}:{c}:{v:.10e};".encode('utf-8'))
        for b_val in self.b:
            hasher.update(f"b:{b_val:.10e};".encode('utf-8'))

        # Sorted triplets E
        for r, c, v in sorted(self.E_triplets):
            hasher.update(f"E:{r}:{c}:{v:.10e};".encode('utf-8'))
        for d_val in self.d:
            hasher.update(f"d:{d_val:.10e};".encode('utf-8'))

        # Q triplets
        for r, c, v in sorted(self.Q_triplets):
            hasher.update(f"Q:{r}:{c}:{v:.10e};".encode('utf-8'))

        return hasher.hexdigest()

    def validate_inputs(self) -> List[str]:
        """FR-IO4: Validate ranges, bounds, and coefficient dynamic ranges."""
        warnings = []
        for i in range(self.num_vars):
            if self.lb[i] > self.ub[i]:
                raise ValueError(f"Variable '{self.var_names[i]}' has lb ({self.lb[i]}) > ub ({self.ub[i]}).")

        # Dynamic range check
        all_coeffs = [abs(v) for _, _, v in self.A_triplets + self.E_triplets + self.Q_triplets if abs(v) > 1e-15]
        if all_coeffs:
            min_c = min(all_coeffs)
            max_c = max(all_coeffs)
            if max_c / max(min_c, 1e-12) > 1e6:
                warnings.append(f"Matrix dynamic range ratio {max_c/min_c:.2e} > 1e6. Ruiz scaling recommended.")
        return warnings

    def evaluate_dual_bound(self, y: np.ndarray, lam: Optional[np.ndarray] = None) -> float:
        """
        §7.3 Valid Dual Bound (Max Form):
          g(y, lambda) = b^T y + d^T lambda + sum_j ( r_j^+ u_j - r_j^- l_j )
          with reduced costs r = c - A^T y - E^T lambda
          r_j^+ = max(r_j, 0), r_j^- = max(-r_j, 0)
        """
        A = self.get_A_csc()
        AT_y = A.rmatvec(y) if self.num_ineq > 0 else np.zeros(self.num_vars)
        ET_lam = np.zeros(self.num_vars)
        if lam is not None and self.num_eq > 0:
            E = self.get_E_csc()
            ET_lam = E.rmatvec(lam)

        r = np.array(self.c) - AT_y - ET_lam

        # Compensated summation of terms
        terms = []
        if self.num_ineq > 0:
            terms.append(float(np.dot(self.b, y)))
        if lam is not None and self.num_eq > 0:
            terms.append(float(np.dot(self.d, lam)))

        for j in range(self.num_vars):
            rj = r[j]
            rj_plus = max(0.0, rj)
            rj_minus = max(0.0, -rj)

            if rj_plus > 1e-12:
                if self.ub[j] >= 1e20:
                    return float('inf')
                terms.append(rj_plus * self.ub[j])

            if rj_minus > 1e-12:
                if self.lb[j] <= -1e20:
                    return float('inf')
                terms.append(-rj_minus * self.lb[j])

        val = kahan_sum(terms)
        # Convert back to user sense if MINIMIZE (canon is max, so bound on original min is -val)
        return val if self.sense == Sense.MAXIMIZE else -val
