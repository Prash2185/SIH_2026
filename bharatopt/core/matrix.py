"""
BharatOpt Core - Sparse Matrix and Vector Foundations
Clean-room implementation of Compressed Sparse Column (CSC), Compressed Sparse Row (CSR),
Triplet/COO representations, vector arithmetic, and Kahan compensated summation.
"""

import math
from typing import List, Tuple, Dict, Optional, Union
import numpy as np


def kahan_sum(values: List[float]) -> float:
    """
    Kahan compensated summation algorithm to minimize numerical error in 64-bit float sums.
    """
    total = 0.0
    c = 0.0  # Compensation for lost low-order bits
    for v in values:
        y = v - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


class SparseVector:
    """Sparse vector with fast dictionary and indexed storage."""
    __slots__ = ('dim', 'indices', 'values', '_dict')

    def __init__(self, dim: int, entries: Optional[Dict[int, float]] = None):
        self.dim = dim
        self._dict: Dict[int, float] = {}
        if entries:
            for k, v in entries.items():
                if abs(v) > 1e-15:
                    self._dict[int(k)] = float(v)
        self.indices: List[int] = sorted(self._dict.keys())
        self.values: List[float] = [self._dict[i] for i in self.indices]

    def set(self, idx: int, val: float):
        if 0 <= idx < self.dim:
            if abs(val) > 1e-15:
                self._dict[idx] = float(val)
            elif idx in self._dict:
                del self._dict[idx]
            self.indices = sorted(self._dict.keys())
            self.values = [self._dict[i] for i in self.indices]
        else:
            raise IndexError(f"Index {idx} out of range for vector dimension {self.dim}")

    def get(self, idx: int) -> float:
        return self._dict.get(idx, 0.0)

    def dot(self, other: Union['SparseVector', List[float], np.ndarray]) -> float:
        """Inner product with compensated summation."""
        terms = []
        if isinstance(other, SparseVector):
            # Iterate over the sparser vector
            if len(self.indices) < len(other.indices):
                for idx, val in zip(self.indices, self.values):
                    oval = other.get(idx)
                    if oval != 0.0:
                        terms.append(val * oval)
            else:
                for idx, val in zip(other.indices, other.values):
                    sval = self.get(idx)
                    if sval != 0.0:
                        terms.append(val * sval)
        else:
            for idx, val in zip(self.indices, self.values):
                if idx < len(other):
                    terms.append(val * other[idx])
        return kahan_sum(terms)

    def norm2(self) -> float:
        return math.sqrt(sum(v * v for v in self.values))

    def norm_inf(self) -> float:
        return max((abs(v) for v in self.values), default=0.0)

    def to_dense(self) -> np.ndarray:
        arr = np.zeros(self.dim, dtype=np.float64)
        for i, v in self._dict.items():
            arr[i] = v
        return arr


class CSCMatrix:
    """Compressed Sparse Column Matrix."""
    def __init__(self, shape: Tuple[int, int], col_ptr: List[int], row_ind: List[int], values: List[float]):
        self.shape = shape
        self.num_rows, self.num_cols = shape
        self.col_ptr = np.array(col_ptr, dtype=np.int32)
        self.row_ind = np.array(row_ind, dtype=np.int32)
        self.values = np.array(values, dtype=np.float64)

    @property
    def nnz(self) -> int:
        return len(self.values)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """y = A * x"""
        y = np.zeros(self.num_rows, dtype=np.float64)
        for j in range(self.num_cols):
            xj = x[j]
            if abs(xj) > 1e-15:
                start = self.col_ptr[j]
                end = self.col_ptr[j + 1]
                for k in range(start, end):
                    y[self.row_ind[k]] += self.values[k] * xj
        return y

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """x = A^T * y"""
        x = np.zeros(self.num_cols, dtype=np.float64)
        for j in range(self.num_cols):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            terms = [self.values[k] * y[self.row_ind[k]] for k in range(start, end)]
            x[j] = kahan_sum(terms)
        return x

    def get_col(self, j: int) -> SparseVector:
        start = self.col_ptr[j]
        end = self.col_ptr[j + 1]
        entries = {int(self.row_ind[k]): float(self.values[k]) for k in range(start, end)}
        return SparseVector(self.num_rows, entries)

    def to_csr(self) -> 'CSRMatrix':
        return triplet_to_csr(self.num_rows, self.num_cols, self.to_triplets())

    def to_triplets(self) -> List[Tuple[int, int, float]]:
        triplets = []
        for j in range(self.num_cols):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            for k in range(start, end):
                triplets.append((int(self.row_ind[k]), j, float(self.values[k])))
        return triplets

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=np.float64)
        for j in range(self.num_cols):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            for k in range(start, end):
                dense[self.row_ind[k], j] = self.values[k]
        return dense


class CSRMatrix:
    """Compressed Sparse Row Matrix."""
    def __init__(self, shape: Tuple[int, int], row_ptr: List[int], col_ind: List[int], values: List[float]):
        self.shape = shape
        self.num_rows, self.num_cols = shape
        self.row_ptr = np.array(row_ptr, dtype=np.int32)
        self.col_ind = np.array(col_ind, dtype=np.int32)
        self.values = np.array(values, dtype=np.float64)

    @property
    def nnz(self) -> int:
        return len(self.values)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """y = A * x"""
        y = np.zeros(self.num_rows, dtype=np.float64)
        for i in range(self.num_rows):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            terms = [self.values[k] * x[self.col_ind[k]] for k in range(start, end)]
            y[i] = kahan_sum(terms)
        return y

    def get_row(self, i: int) -> SparseVector:
        start = self.row_ptr[i]
        end = self.row_ptr[i + 1]
        entries = {int(self.col_ind[k]): float(self.values[k]) for k in range(start, end)}
        return SparseVector(self.num_cols, entries)

    def to_csc(self) -> CSCMatrix:
        return triplet_to_csc(self.num_rows, self.num_cols, self.to_triplets())

    def to_triplets(self) -> List[Tuple[int, int, float]]:
        triplets = []
        for i in range(self.num_rows):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for k in range(start, end):
                triplets.append((i, int(self.col_ind[k]), float(self.values[k])))
        return triplets

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=np.float64)
        for i in range(self.num_rows):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for k in range(start, end):
                dense[i, self.col_ind[k]] = self.values[k]
        return dense


def triplet_to_csc(num_rows: int, num_cols: int, triplets: List[Tuple[int, int, float]]) -> CSCMatrix:
    """Build CSC matrix from (row, col, val) triplets, combining duplicate entries."""
    cols: List[List[Tuple[int, float]]] = [[] for _ in range(num_cols)]
    for r, c, v in triplets:
        if 0 <= r < num_rows and 0 <= c < num_cols and abs(v) > 1e-15:
            cols[c].append((r, v))

    col_ptr = [0]
    row_ind = []
    values = []

    for c in range(num_cols):
        # Accumulate duplicates in same column
        entry_map: Dict[int, float] = {}
        for r, v in cols[c]:
            entry_map[r] = entry_map.get(r, 0.0) + v

        for r in sorted(entry_map.keys()):
            val = entry_map[r]
            if abs(val) > 1e-15:
                row_ind.append(r)
                values.append(val)
        col_ptr.append(len(values))

    return CSCMatrix((num_rows, num_cols), col_ptr, row_ind, values)


def triplet_to_csr(num_rows: int, num_cols: int, triplets: List[Tuple[int, int, float]]) -> CSRMatrix:
    """Build CSR matrix from (row, col, val) triplets, combining duplicate entries."""
    rows: List[List[Tuple[int, float]]] = [[] for _ in range(num_rows)]
    for r, c, v in triplets:
        if 0 <= r < num_rows and 0 <= c < num_cols and abs(v) > 1e-15:
            rows[r].append((c, v))

    row_ptr = [0]
    col_ind = []
    values = []

    for r in range(num_rows):
        entry_map: Dict[int, float] = {}
        for c, v in rows[r]:
            entry_map[c] = entry_map.get(c, 0.0) + v

        for c in sorted(entry_map.keys()):
            val = entry_map[c]
            if abs(val) > 1e-15:
                col_ind.append(c)
                values.append(val)
        row_ptr.append(len(values))

    return CSRMatrix((num_rows, num_cols), row_ptr, col_ind, values)
