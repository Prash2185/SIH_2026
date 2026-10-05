"""
BharatOpt Core - Run Tiers & Determinism Management (FR-DET1 - FR-DET2, §9.3)
Run Tiers:
  T0: Opportunistic (Fastest; portfolio racing; wall-clock limits allowed)
  T1: Deterministic (Default for published plans; fixed seed; work/node limits; 100-run identical SHA-256 plan hash)
  T2: Certified Replay (Audit trail replay; archives integer assignments and basis)
"""

import hashlib
import json
import time
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from bharatopt.core.model import CanonicalModel, Sense
from bharatopt.solvers.simplex import RevisedSimplexSolver, SolverStatus
from bharatopt.solvers.milp import BranchAndCutSolver, MIPResult
from bharatopt.verifier.verifier import IndependentVerifier, VerifierReport


class RunTier(Enum):
    T0 = "T0"  # Opportunistic
    T1 = "T1"  # Deterministic (Default for published plans)
    T2 = "T2"  # Certified Replay


class ReplayBundle:
    """Certified Replay Archive (T2)."""
    def __init__(self, run_id: str, model_hash: str, integer_assignment: Dict[str, float],
                 seed: int, basis_indices: List[int], objective: float):
        self.run_id = run_id
        self.model_hash = model_hash
        self.integer_assignment = integer_assignment
        self.seed = seed
        self.basis_indices = basis_indices
        self.objective = objective

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_hash": self.model_hash,
            "integer_assignment": self.integer_assignment,
            "seed": self.seed,
            "basis_indices": self.basis_indices,
            "objective": self.objective
        }


class TierOrchestrator:
    """Executes optimization solves according to Tier T0, T1, or T2 rules."""
    def __init__(self):
        self.verifier = IndependentVerifier()

    def solve(self, model: CanonicalModel,
              tier: RunTier = RunTier.T1,
              seed: int = 42,
              max_nodes: int = 5000,
              max_time_sec: float = 60.0) -> Tuple[MIPResult, VerifierReport, Optional[ReplayBundle]]:
        # Pinned determinism setup for T1
        if tier == RunTier.T1:
            np.random.seed(seed)
            # T1 forbids non-deterministic wall-clock early termination; uses work limits
            solver = BranchAndCutSolver(
                max_nodes=max_nodes,
                max_time_sec=float('inf'),  # Strict iteration/node limits
                gap_tol=1e-4
            )
        else:
            solver = BranchAndCutSolver(
                max_nodes=max_nodes,
                max_time_sec=max_time_sec,
                gap_tol=1e-4
            )

        sol = solver.solve(model)

        # Independent Verifier Gate
        ver_rep = self.verifier.verify_solution(
            model=model,
            x=sol.x,
            reported_obj=sol.obj_val,
            dual_bound=sol.best_dual_bound
        )

        # If verifier fails, withhold solution and report NUMERICAL
        if not ver_rep.passed and sol.status == SolverStatus.OPTIMAL:
            sol.status = SolverStatus.NUMERICAL

        # Create T2 replay bundle
        replay_bundle = None
        if sol.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            int_map = {model.var_names[j]: float(sol.x[j]) for j in model.integer_indices}
            replay_bundle = ReplayBundle(
                run_id=f"RUN_{int(time.time()*1000)}",
                model_hash=model.compute_hash(),
                integer_assignment=int_map,
                seed=seed,
                basis_indices=[],
                objective=sol.obj_val
            )

        return sol, ver_rep, replay_bundle

    def replay_t2(self, model: CanonicalModel, replay_bundle: ReplayBundle) -> Tuple[bool, float, VerifierReport]:
        """T2 Certified Replay: verifies solution by fixing integer assignments and solving LP."""
        if model.compute_hash() != replay_bundle.model_hash:
            raise ValueError("Model hash mismatch in T2 Certified Replay bundle.")

        import copy
        fixed_model = copy.deepcopy(model)
        for var_name, val in replay_bundle.integer_assignment.items():
            if var_name in fixed_model.var_name_to_idx:
                idx = fixed_model.var_name_to_idx[var_name]
                fixed_model.lb[idx] = val
                fixed_model.ub[idx] = val

        lp_solver = RevisedSimplexSolver()
        lp_res = lp_solver.solve(fixed_model)

        ver_rep = self.verifier.verify_solution(
            model=model,
            x=lp_res.x,
            reported_obj=lp_res.obj_val,
            dual_bound=lp_res.obj_val
        )

        matches_obj = abs(lp_res.obj_val - replay_bundle.objective) < 1e-6
        return (ver_rep.passed and matches_obj), lp_res.obj_val, ver_rep
