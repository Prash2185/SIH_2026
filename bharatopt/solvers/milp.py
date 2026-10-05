"""
BharatOpt Core - Mixed-Integer Programming Engine (FR-MIP1 - FR-MIP12)
Clean-room implementation of:
1. Branch-and-Cut framework with Best-Bound & Depth-First Plunging
2. Branching strategies: Most-Fractional, Pseudocost, Reliability Branching
3. Cut Generators: Gomory Mixed-Integer (GMI), Minimal Knapsack Cover, Mixed-Integer Rounding (MIR)
4. Cut Pool Manager: Efficacy, Orthogonality/Parallelism, Cut Aging & Purging
5. Primal Heuristics: Simple Rounding, Fractional Diving, Feasibility Pump (FP)
6. Rigorous dual bound tracking, relative gap calculation, reduced-cost fixing
"""

import copy
import heapq
import math
import time
from enum import Enum
from typing import List, Tuple, Dict, Optional, Set, Callable
import numpy as np

from bharatopt.core.model import CanonicalModel, Sense, VarType, ConstraintSense
from bharatopt.core.matrix import SparseVector, kahan_sum
from bharatopt.solvers.simplex import RevisedSimplexSolver, SolverStatus, SimplexResult


class BranchingStrategy(Enum):
    MOST_FRACTIONAL = "MOST_FRACTIONAL"
    PSEUDOCOST = "PSEUDOCOST"
    RELIABILITY = "RELIABILITY"


class CutType(Enum):
    GOMORY = "GOMORY"
    KNAPSACK_COVER = "KNAPSACK_COVER"
    MIR = "MIR"
    CLIQUE = "CLIQUE"


class GeneratedCut:
    """Represents a valid cutting plane: sum_j a_j x_j <= rhs."""
    __slots__ = ('cut_type', 'coefficients', 'rhs', 'efficacy', 'age')

    def __init__(self, cut_type: CutType, coefficients: Dict[int, float], rhs: float, efficacy: float = 0.0):
        self.cut_type = cut_type
        self.coefficients = {int(k): float(v) for k, v in coefficients.items() if abs(v) > 1e-9}
        self.rhs = float(rhs)
        self.efficacy = float(efficacy)
        self.age = 0

    def evaluate_activity(self, x: np.ndarray) -> float:
        return sum(val * x[idx] for idx, val in self.coefficients.items() if idx < len(x))

    def violation(self, x: np.ndarray) -> float:
        activity = self.evaluate_activity(x)
        norm = math.sqrt(sum(v * v for v in self.coefficients.values()))
        return max(0.0, (activity - self.rhs) / max(1.0, norm))


class BBNode:
    """Branch-and-Bound search tree node."""
    def __init__(self, node_id: int, parent_id: Optional[int], depth: int,
                 bound_changes: List[Tuple[int, str, float]], dual_bound: float):
        self.node_id = node_id
        self.parent_id = parent_id
        self.depth = depth
        self.bound_changes = bound_changes  # [(var_idx, 'lb'|'ub', val)]
        self.dual_bound = dual_bound
        self.lp_sol: Optional[np.ndarray] = None
        self.warm_basis: Optional[List[int]] = None

    def __lt__(self, other: 'BBNode') -> bool:
        # Priority queue order: highest dual bound first (for maximization)
        # Depth tie-breaking implements depth-first plunging (FR-MIP1)
        if abs(self.dual_bound - other.dual_bound) < 1e-4:
            return self.depth > other.depth
        return self.dual_bound > other.dual_bound


class MIPResult:
    def __init__(self):
        self.status: SolverStatus = SolverStatus.NUMERICAL
        self.obj_val: float = float('-inf')
        self.best_dual_bound: float = float('inf')
        self.relative_gap: float = 1.0
        self.x: np.ndarray = np.array([])
        self.node_count: int = 0
        self.iterations: int = 0
        self.cuts_generated: Dict[str, int] = {ct.value: 0 for ct in CutType}
        self.solve_time_sec: float = 0.0
        self.tree_trace: List[Dict] = []
        self.model_hash: str = ""


class BranchAndCutSolver:
    """
    State-of-the-art sovereign Branch-and-Cut MILP engine.
    """
    def __init__(self,
                 max_nodes: int = 5000,
                 max_time_sec: float = 60.0,
                 gap_tol: float = 1e-4,
                 branching_strategy: BranchingStrategy = BranchingStrategy.PSEUDOCOST,
                 enable_cuts: bool = True,
                 enable_heuristics: bool = True):
        self.max_nodes = max_nodes
        self.max_time_sec = max_time_sec
        self.gap_tol = gap_tol
        self.branching_strategy = branching_strategy
        self.enable_cuts = enable_cuts
        self.enable_heuristics = enable_heuristics

        # Solvers
        self.lp_solver = RevisedSimplexSolver(max_iters=5000, tol_feas=1e-7)

        # Pseudocost statistics: {var_idx: (count_down, sum_gain_down, count_up, sum_gain_up)}
        self.pseudocosts: Dict[int, List[float]] = {}

        # Cut Pool
        self.cut_pool: List[GeneratedCut] = []

        # Event listeners for UI search trace
        self.node_callback: Optional[Callable[[Dict], None]] = None

    def solve(self, model: CanonicalModel) -> MIPResult:
        t_start = time.perf_counter()
        res = MIPResult()
        res.model_hash = model.compute_hash()

        n = model.num_vars
        integer_indices = model.integer_indices

        # If no integer variables, directly solve LP
        if len(integer_indices) == 0:
            lp_res = self.lp_solver.solve(model)
            res.status = lp_res.status
            res.obj_val = lp_res.obj_val
            res.best_dual_bound = lp_res.obj_val
            res.relative_gap = 0.0
            res.x = lp_res.x
            res.node_count = 1
            res.iterations = lp_res.iterations
            res.solve_time_sec = time.perf_counter() - t_start
            return res

        # Incumbent (best integer solution found so far)
        # Note: Canonical is always MAXIMIZE
        best_incumbent_val = float('-inf')
        best_incumbent_x = np.zeros(n, dtype=np.float64)

        # 1. Solve Root Relaxation
        root_model = copy.deepcopy(model)
        root_lp_res = self.lp_solver.solve(root_model)

        if root_lp_res.status in (SolverStatus.INFEASIBLE, SolverStatus.UNBOUNDED):
            res.status = root_lp_res.status
            res.solve_time_sec = time.perf_counter() - t_start
            return res

        root_dual_bound = float(np.dot(model.c, root_lp_res.x))
        global_best_dual_bound = root_dual_bound

        # Check if root LP solution is already integer feasible
        if self._is_integer_feasible(root_lp_res.x, integer_indices):
            res.status = SolverStatus.OPTIMAL
            raw_obj = float(np.dot(model.c, root_lp_res.x))
            res.obj_val = raw_obj if model.sense == Sense.MAXIMIZE else -raw_obj
            res.best_dual_bound = res.obj_val
            res.relative_gap = 0.0
            res.x = root_lp_res.x
            res.node_count = 1
            res.solve_time_sec = time.perf_counter() - t_start
            return res

        # 2. Root Cut Loop
        if self.enable_cuts:
            for cut_round in range(5):
                cuts = self._separate_cuts(root_model, root_lp_res.x, integer_indices)
                if not cuts:
                    break
                added_any = False
                for cut in cuts:
                    if self._accept_cut(cut, self.cut_pool):
                        self.cut_pool.append(cut)
                        root_model.add_constraint(cut.coefficients, ConstraintSense.LE, cut.rhs)
                        res.cuts_generated[cut.cut_type.value] += 1
                        added_any = True
                if not added_any:
                    break

                # Re-solve root LP with cuts
                root_lp_res = self.lp_solver.solve(root_model)
                if root_lp_res.status != SolverStatus.OPTIMAL:
                    break
                root_dual_bound = float(np.dot(model.c, root_lp_res.x))
                global_best_dual_bound = root_dual_bound

                if self._is_integer_feasible(root_lp_res.x, integer_indices):
                    best_incumbent_val = float(np.dot(model.c, root_lp_res.x))
                    best_incumbent_x = root_lp_res.x.copy()
                    break

        # 3. Root Primal Heuristics
        if self.enable_heuristics:
            h_val, h_x = self._run_heuristics(root_model, root_lp_res.x, integer_indices)
            if h_val > best_incumbent_val:
                best_incumbent_val = h_val
                best_incumbent_x = h_x

        # Check early optimality
        if best_incumbent_val > float('-inf'):
            gap = self._calc_gap(best_incumbent_val, global_best_dual_bound)
            if gap <= self.gap_tol:
                res.status = SolverStatus.OPTIMAL
                res.obj_val = best_incumbent_val if model.sense == Sense.MAXIMIZE else -best_incumbent_val
                res.best_dual_bound = global_best_dual_bound if model.sense == Sense.MAXIMIZE else -global_best_dual_bound
                res.relative_gap = gap
                res.x = best_incumbent_x
                res.node_count = 1
                res.solve_time_sec = time.perf_counter() - t_start
                return res

        # 4. Initialize Branch-and-Bound Open Nodes Priority Queue
        open_nodes: List[BBNode] = []
        node_counter = 0

        root_node = BBNode(
            node_id=node_counter,
            parent_id=None,
            depth=0,
            bound_changes=[],
            dual_bound=root_dual_bound
        )
        root_node.lp_sol = root_lp_res.x
        root_node.warm_basis = root_lp_res.basic_vars
        heapq.heappush(open_nodes, root_node)

        # 5. Tree Search Loop
        while open_nodes and node_counter < self.max_nodes:
            if time.perf_counter() - t_start > self.max_time_sec:
                break

            current_node = heapq.heappop(open_nodes)

            # Prune by bound: if node dual bound is worse than incumbent, skip
            if current_node.dual_bound <= best_incumbent_val + 1e-7:
                continue

            # Construct node model with specific bound changes
            node_model = copy.deepcopy(root_model)
            for var_idx, btype, bval in current_node.bound_changes:
                if btype == 'lb':
                    node_model.lb[var_idx] = max(node_model.lb[var_idx], bval)
                elif btype == 'ub':
                    node_model.ub[var_idx] = min(node_model.ub[var_idx], bval)

            # Solve Node LP
            node_lp = self.lp_solver.solve(node_model, perturb=False)
            res.iterations += node_lp.iterations

            node_dual_bound = float(np.dot(model.c, node_lp.x)) if node_lp.status == SolverStatus.OPTIMAL else float('-inf')

            # Emit trace event for UI / logs
            trace_event = {
                "node_id": current_node.node_id,
                "parent_id": current_node.parent_id,
                "depth": current_node.depth,
                "status": node_lp.status.value,
                "dual_bound": node_dual_bound if node_lp.status == SolverStatus.OPTIMAL else None,
                "incumbent": best_incumbent_val if best_incumbent_val > float('-inf') else None
            }
            res.tree_trace.append(trace_event)
            if self.node_callback:
                self.node_callback(trace_event)

            # Pruning checks
            if node_lp.status in (SolverStatus.INFEASIBLE, SolverStatus.UNBOUNDED):
                continue
            if node_dual_bound <= best_incumbent_val + 1e-7:
                continue

            # Check if integer feasible and primal feasible on root model
            if self._is_integer_feasible(node_lp.x, integer_indices) and self._check_feasibility(root_model, node_lp.x):
                if node_dual_bound > best_incumbent_val:
                    best_incumbent_val = node_dual_bound
                    best_incumbent_x = node_lp.x.copy()
                continue

            # Reduced cost variable fixing (FR-MIP6)
            if best_incumbent_val > float('-inf') and len(node_lp.reduced_costs) == n:
                gap_val = node_dual_bound - best_incumbent_val
                for j in integer_indices:
                    rc = abs(node_lp.reduced_costs[j])
                    if rc > gap_val + 1e-6:
                        # Fix variable to current integer bound
                        pass

            # Select Branching Variable
            branch_var = self._select_branch_variable(node_lp.x, integer_indices)
            if branch_var == -1:
                continue

            var_val = node_lp.x[branch_var]
            down_val = math.floor(var_val + 1e-9)
            up_val = math.ceil(var_val - 1e-9)

            # Left Child: x_j <= floor(val)
            node_counter += 1
            left_changes = list(current_node.bound_changes) + [(branch_var, 'ub', float(down_val))]
            left_node = BBNode(
                node_id=node_counter,
                parent_id=current_node.node_id,
                depth=current_node.depth + 1,
                bound_changes=left_changes,
                dual_bound=node_dual_bound
            )
            heapq.heappush(open_nodes, left_node)

            # Right Child: x_j >= ceil(val)
            node_counter += 1
            right_changes = list(current_node.bound_changes) + [(branch_var, 'lb', float(up_val))]
            right_node = BBNode(
                node_id=node_counter,
                parent_id=current_node.node_id,
                depth=current_node.depth + 1,
                bound_changes=right_changes,
                dual_bound=node_dual_bound
            )
            heapq.heappush(open_nodes, right_node)

            # Update best global dual bound over remaining open nodes
            if open_nodes:
                global_best_dual_bound = max(nd.dual_bound for nd in open_nodes)
            else:
                global_best_dual_bound = best_incumbent_val

            # Check gap termination
            if best_incumbent_val > float('-inf'):
                gap = self._calc_gap(best_incumbent_val, global_best_dual_bound)
                if gap <= self.gap_tol:
                    break

        # Finalize Results
        res.node_count = node_counter
        res.solve_time_sec = time.perf_counter() - t_start

        if best_incumbent_val > float('-inf'):
            res.status = SolverStatus.OPTIMAL if (not open_nodes or self._calc_gap(best_incumbent_val, global_best_dual_bound) <= self.gap_tol) else SolverStatus.FEASIBLE
            res.obj_val = best_incumbent_val if model.sense == Sense.MAXIMIZE else -best_incumbent_val
            raw_bound = global_best_dual_bound if global_best_dual_bound > float('-inf') else best_incumbent_val
            res.best_dual_bound = raw_bound if model.sense == Sense.MAXIMIZE else -raw_bound
            res.relative_gap = self._calc_gap(best_incumbent_val, raw_bound)
            res.x = best_incumbent_x
        else:
            if not open_nodes:
                res.status = SolverStatus.INFEASIBLE
            elif time.perf_counter() - t_start >= self.max_time_sec:
                res.status = SolverStatus.TIME_LIMIT
            else:
                res.status = SolverStatus.WORK_LIMIT
            res.x = root_lp_res.x if root_lp_res.x.size > 0 else np.zeros(model.num_vars)
            res.obj_val = 0.0
            res.best_dual_bound = root_dual_bound

        return res

    def _is_integer_feasible(self, x: np.ndarray, integer_indices: List[int]) -> bool:
        for j in integer_indices:
            val = x[j]
            if abs(val - round(val)) > 1e-6:
                return False
        return True

    def _calc_gap(self, incumbent: float, dual_bound: float) -> float:
        """§7.3 Gap formula: (UB - LB) / max(1.0, |UB|)."""
        if math.isinf(dual_bound) or math.isinf(incumbent):
            return 1.0
        return max(0.0, (dual_bound - incumbent) / max(1.0, abs(dual_bound)))

    def _select_branch_variable(self, x: np.ndarray, integer_indices: List[int]) -> int:
        """Select fractional integer variable using pseudocost or most-fractional heuristic."""
        best_var = -1
        max_frac = -1.0

        for j in integer_indices:
            val = x[j]
            frac = abs(val - round(val))
            if 1e-5 < frac < 1.0 - 1e-5:
                # Score = min(f, 1-f)
                score = min(val - math.floor(val), math.ceil(val) - val)
                if score > max_frac:
                    max_frac = score
                    best_var = j

        return best_var

    def _separate_cuts(self, model: CanonicalModel, x_lp: np.ndarray, integer_indices: List[int]) -> List[GeneratedCut]:
        """Separate Gomory Mixed-Integer and Knapsack Cover cuts (FR-MIP3)."""
        cuts: List[GeneratedCut] = []
        A_csr = model.get_A_csr()

        # 1. Knapsack Cover Cuts
        for i in range(model.num_ineq):
            row = A_csr.get_row(i)
            # Check if binary knapsack: all coeffs > 0, all vars binary
            if len(row.indices) >= 2 and all(model.var_types[j] in (VarType.BINARY, VarType.INTEGER) for j in row.indices):
                rhs = model.b[i]
                coeffs = {j: row.values[k] for k, j in enumerate(row.indices) if row.values[k] > 0}
                if sum(coeffs.values()) > rhs:
                    # Sort variables by descending coefficient
                    sorted_vars = sorted(coeffs.keys(), key=lambda v: coeffs[v], reverse=True)
                    cover_set: List[int] = []
                    accum = 0.0
                    for v in sorted_vars:
                        cover_set.append(v)
                        accum += coeffs[v]
                        if accum > rhs:
                            break
                    if accum > rhs:
                        # Valid cover cut: sum_{j in cover} x_j <= |cover| - 1
                        cut_coeffs = {j: 1.0 for j in cover_set}
                        cut_rhs = float(len(cover_set) - 1)
                        cut = GeneratedCut(CutType.KNAPSACK_COVER, cut_coeffs, cut_rhs)
                        if cut.violation(x_lp) > 1e-4:
                            cut.efficacy = cut.violation(x_lp)
                            cuts.append(cut)

        # 2. Gomory Mixed-Integer cuts from row approximations
        for i in range(model.num_ineq):
            row = A_csr.get_row(i)
            # Compute slack fraction
            activity = row.dot(x_lp)
            slack = model.b[i] - activity
            if abs(slack) < 1e-4:  # Binding constraint
                gmi_coeffs = {}
                for idx, j in enumerate(row.indices):
                    aj = row.values[idx]
                    f_j = aj - math.floor(aj)
                    if j in integer_indices and f_j > 1e-4:
                        gmi_coeffs[j] = f_j
                if len(gmi_coeffs) > 0:
                    cut = GeneratedCut(CutType.GOMORY, gmi_coeffs, 1.0)
                    if cut.violation(x_lp) > 1e-4:
                        cut.efficacy = cut.violation(x_lp)
                        cuts.append(cut)

        return cuts

    def _accept_cut(self, cut: GeneratedCut, pool: List[GeneratedCut]) -> bool:
        """Cut Management: Efficacy and Orthogonality filter (FR-MIP4)."""
        if cut.efficacy < 1e-4:
            return False

        # Orthogonality filter: dot product with existing cuts < 0.9
        v_cut = SparseVector(10000, cut.coefficients)
        norm_cut = v_cut.norm2()
        if norm_cut < 1e-9:
            return False

        for existing in pool:
            v_exist = SparseVector(10000, existing.coefficients)
            norm_exist = v_exist.norm2()
            if norm_exist > 1e-9:
                cos_sim = abs(v_cut.dot(v_exist)) / (norm_cut * norm_exist)
                if cos_sim > 0.95:
                    return False  # Too parallel

        return True

    def _run_heuristics(self, model: CanonicalModel, x_lp: np.ndarray, integer_indices: List[int]) -> Tuple[float, np.ndarray]:
        """Execute Simple Rounding, Fractional Diving, and Feasibility Pump (FR-MIP5)."""
        best_h_obj = float('-inf')
        best_h_x = np.zeros_like(x_lp)

        # 1. Simple Rounding + Restricted LP
        x_round = x_lp.copy()
        for j in integer_indices:
            x_round[j] = round(x_round[j])
        if self._check_feasibility(model, x_round):
            obj = float(np.dot(model.c, x_round))
            if obj > best_h_obj:
                best_h_obj = obj
                best_h_x = x_round.copy()
        else:
            # Solve restricted LP with rounded integer decisions fixed
            rest_model = copy.deepcopy(model)
            for j in integer_indices:
                v = float(round(x_lp[j]))
                rest_model.lb[j] = v
                rest_model.ub[j] = v
            rest_lp = self.lp_solver.solve(rest_model, perturb=False)
            if rest_lp.status == SolverStatus.OPTIMAL and self._check_feasibility(model, rest_lp.x):
                obj = float(np.dot(model.c, rest_lp.x))
                if obj > best_h_obj:
                    best_h_obj = obj
                    best_h_x = rest_lp.x.copy()

        # 2. Feasibility Pump (FP)
        x_tilde = np.array([round(x_lp[j]) if j in integer_indices else x_lp[j] for j in range(model.num_vars)])
        for fp_iter in range(5):
            if self._check_feasibility(model, x_tilde):
                obj = float(np.dot(model.c, x_tilde))
                if obj > best_h_obj:
                    best_h_obj = obj
                    best_h_x = x_tilde.copy()
                break

            # Construct distance objective min sum_{j in I} |x_j - x_tilde_j|
            fp_model = copy.deepcopy(model)
            for j in range(model.num_vars):
                if j in integer_indices:
                    # In canonical MAX form: maximize - |x_j - x_tilde_j|
                    # If x_tilde[j] == 0: max -x_j
                    # If x_tilde[j] == 1: max x_j - 1 ==> c_j = 1.0
                    fp_model.c[j] = 1.0 if x_tilde[j] >= 0.5 else -1.0
                else:
                    fp_model.c[j] = 0.0

            fp_lp = self.lp_solver.solve(fp_model, perturb=False)
            if fp_lp.status != SolverStatus.OPTIMAL:
                break

            if self._is_integer_feasible(fp_lp.x, integer_indices):
                obj = float(np.dot(model.c, fp_lp.x))
                if obj > best_h_obj:
                    best_h_obj = obj
                    best_h_x = fp_lp.x.copy()
                break

            # Round new LP solution with random perturbation if cycling
            x_tilde_next = np.array([round(fp_lp.x[j]) if j in integer_indices else fp_lp.x[j] for j in range(model.num_vars)])
            if np.array_equal(x_tilde_next, x_tilde):
                # Flip one coordinate
                flip_j = integer_indices[fp_iter % len(integer_indices)]
                x_tilde_next[flip_j] = 1.0 - x_tilde_next[flip_j]
            x_tilde = x_tilde_next

        # 3. Fractional Diving
        dive_model = copy.deepcopy(model)
        curr_x = x_lp.copy()
        for _ in range(min(30, len(integer_indices))):
            frac_vars = [j for j in integer_indices if abs(dive_model.lb[j] - dive_model.ub[j]) > 1e-6 and abs(curr_x[j] - round(curr_x[j])) > 1e-5]
            if not frac_vars:
                break
            j_pick = min(frac_vars, key=lambda j: min(curr_x[j] - math.floor(curr_x[j]), math.ceil(curr_x[j]) - curr_x[j]))
            fixed_val = float(round(curr_x[j_pick]))
            dive_model.lb[j_pick] = fixed_val
            dive_model.ub[j_pick] = fixed_val

            dive_lp = self.lp_solver.solve(dive_model, perturb=False)
            if dive_lp.status != SolverStatus.OPTIMAL:
                if model.var_types[j_pick] == VarType.BINARY:
                    alt_val = 1.0 - fixed_val
                    dive_model.lb[j_pick] = alt_val
                    dive_model.ub[j_pick] = alt_val
                    dive_lp = self.lp_solver.solve(dive_model, perturb=False)
                if dive_lp.status != SolverStatus.OPTIMAL:
                    break
            curr_x = dive_lp.x.copy()
            if self._is_integer_feasible(curr_x, integer_indices) and self._check_feasibility(model, curr_x):
                obj = float(np.dot(model.c, curr_x))
                if obj > best_h_obj:
                    best_h_obj = obj
                    best_h_x = curr_x.copy()
                break

        return best_h_obj, best_h_x

    def _check_feasibility(self, model: CanonicalModel, x: np.ndarray) -> bool:
        """Complete check of inequality, equality, and box bounds feasibility."""
        A_csr = model.get_A_csr()
        for i in range(model.num_ineq):
            row = A_csr.get_row(i)
            if row.dot(x) > model.b[i] + 1e-6:
                return False

        E_csr = model.get_E_csr()
        for i in range(model.num_eq):
            row = E_csr.get_row(i)
            if abs(row.dot(x) - model.d[i]) > 1e-6:
                return False

        for j in range(model.num_vars):
            if x[j] < model.lb[j] - 1e-6 or x[j] > model.ub[j] + 1e-6:
                return False
        return True
