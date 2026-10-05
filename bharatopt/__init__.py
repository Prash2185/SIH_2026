"""
BharatOpt Core - Top-Level Package API
Sovereign LP / QP / MILP Optimization Engine with GPU Acceleration (SIH26119)
"""

__version__ = "5.0.0"
__author__ = "BharatOpt Core Team (MRPL / SIH26119)"

from typing import Tuple, Any, Optional

from bharatopt.core.matrix import CSCMatrix, CSRMatrix, SparseVector, kahan_sum
from bharatopt.core.linalg import SparseLU, BasisFactorization, SparseCholesky, RuizScaling
from bharatopt.core.model import CanonicalModel, VarType, Sense, ConstraintSense
from bharatopt.core.presolve import Presolver
from bharatopt.solvers.simplex import RevisedSimplexSolver, SolverStatus, SimplexResult
from bharatopt.solvers.ipm import MehrotraIPMSolver, IPMResult
from bharatopt.solvers.gpu_pdhg import FirstOrderPDHGSolver, PDHGResult
from bharatopt.solvers.milp import BranchAndCutSolver, MIPResult, BranchingStrategy, CutType
from bharatopt.verifier.verifier import IndependentVerifier, VerifierReport
from bharatopt.verifier.forward_sim import ForwardSimulationVerifier, ForwardSimulationReport
from bharatopt.explanation.bundle import ExplainabilityEngine, ExplanationBundle, SensitivityReport, WhyNotXAnalysis
from bharatopt.run_tiers.tiers import RunTier, TierOrchestrator, ReplayBundle
from bharatopt.io.mps_reader import MPSReader, LPFormatReader


def solve(model: CanonicalModel,
          tier: RunTier = RunTier.T1,
          method: str = "auto",
          max_time_sec: float = 60.0) -> Tuple[Any, VerifierReport]:
    """
    Unified top-level solve entrypoint.
    Routes model automatically to:
      - Revised Simplex / IPM for Continuous LP
      - Mehrotra IPM for Convex QP
      - Branch-and-Cut for Mixed-Integer (MILP)
    Enforces independent verifier gate on all solutions.
    """
    orchestrator = TierOrchestrator()
    if model.is_mip:
        sol, ver_rep, _ = orchestrator.solve(model, tier=tier, max_time_sec=max_time_sec)
        return sol, ver_rep
    elif model.is_qp:
        ipm = MehrotraIPMSolver()
        sol = ipm.solve(model)
        ver = IndependentVerifier()
        ver_rep = ver.verify_solution(model, sol.x, sol.obj_val, sol.wolfe_dual_bound, sol.y)
        return sol, ver_rep
    else:
        # Continuous LP
        if method == "ipm":
            ipm = MehrotraIPMSolver()
            sol = ipm.solve(model)
        elif method == "gpu_pdhg":
            pdhg = FirstOrderPDHGSolver()
            sol = pdhg.solve(model)
        else:
            simplex = RevisedSimplexSolver()
            sol = simplex.solve(model)

        ver = IndependentVerifier()
        dual_bnd = model.evaluate_dual_bound(sol.y) if len(sol.y) > 0 else sol.obj_val
        ver_rep = ver.verify_solution(model, sol.x, sol.obj_val, dual_bnd, sol.y)
        return sol, ver_rep
