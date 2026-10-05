# BharatOpt Core v5.0 — Sovereign LP / QP / MILP Optimization Engine with GPU Acceleration

[![Gate G0 Clean-Room](https://img.shields.io/badge/Gate_G0-Clean--Room_Verified-10b981.svg)](https://github.com/bharatopt/bharatopt)
[![Appendix A](https://img.shields.io/badge/Appendix_A-10%2F10_Passed-3b82f6.svg)](https://github.com/bharatopt/bharatopt)
[![SIH26119 Compliance](https://img.shields.io/badge/Problem_Statement-SIH26119_MRPL-f97316.svg)](https://github.com/bharatopt/bharatopt)
[![Determinism](https://img.shields.io/badge/Tier_T1-100%25_Deterministic-8b5cf6.svg)](https://github.com/bharatopt/bharatopt)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **Sovereign Mathematical Programming Core** engineered from scratch from mathematical foundations for high-stakes downstream refinery operations (Mangalore Refinery and Petrochemicals Limited - MRPL) and critical industrial infrastructure.

---

## 1. Executive Summary

BharatOpt Core v5.0 is India's first sovereign, from-scratch optimization solver engine designed for **Linear Programming (LP)**, **Convex Quadratic Programming (QP)**, and **Mixed-Integer Linear Programming (MILP)**. 

### Clean-Room Policy & Sovereign Engineering
- **Zero Third-Party Solver Dependencies:** Not built on or linked with SCIP, HiGHS, SoPlex, PaPILO, CBC, GLPK, OSQP, or Ipopt.
- **Published Literature Foundations:** All numerical linear algebra, simplex pivots, interior-point algorithms, branch-and-cut routines, and cutting plane generators are engineered 100% in-house.
- **Four-Layer Trust Architecture:** Solutions are never returned without passing an independent mathematical verifier gate.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FOUR-LAYER TRUST STACK                          │
├─────────┬────────────────────────────┬─────────────────────────────────┤
│ Layer 1 │ Verifier Gate              │ "Is this answer valid?"         │
│         │ (Primal/Dual/KKT/Sim)      │ Strict mathematical proofs      │
├─────────┼────────────────────────────┼─────────────────────────────────┤
│ Layer 2 │ Sensitivity & Explanation  │ "Why this? Why not X?"          │
│         │ (Duals, Ranging, IIS, Diff)│ Transparent economic indicators │
├─────────┼────────────────────────────┼─────────────────────────────────┤
│ Layer 3 │ Search Trace               │ "What did the solver do?"       │
│         │ (B&B Tree, Residual Logs)  │ Process audit (not causality)   │
├─────────┼────────────────────────────┼─────────────────────────────────┤
│ Layer 4 │ Comparison Mode            │ "How does it compare to oracles"│
│         │ (Side-by-side vs HiGHS/SCIP│ Benchmark & validation harness  │
└─────────┴────────────────────────────┴─────────────────────────────────┘
```

---

## 2. Key Mathematical Capabilities

### 2.1 Numerical Linear Algebra Kernel (FR-LA1 – FR-LA5)
- **Sparse Markowitz LU Factorization:** Threshold partial pivoting ($\tau \in [0.01, 0.1]$) with directed acyclic graph (DAG) triangular solves.
- **Forrest-Tomlin & Product-Form (Eta) Updates:** Dynamic basis updates with numerical growth monitoring and stability-triggered refactorization.
- **Sparse Cholesky / $LDL^T$ Factorizer:** Symmetric factorization with Approximate Minimum Degree (AMD) ordering and diagonal regularization $\delta \sim 10^{-12}$.
- **Ruiz Geometric Equilibration Scaling:** Iteratively balances matrix row and column infinity norms.
- **Kahan Compensated Summation:** Eliminates low-order floating point cancellation error.

### 2.2 Continuous Optimization (LP & Convex QP)
- **Revised Dual Simplex:** Dual steepest-edge (DSE) pricing, bounded variable ratio tests, and Harris two-pass ratio test.
- **Two-Phase Revised Primal Simplex:** Robust Phase 1 artificial variable reduction and Phase 2 optimization.
- **Degeneracy & Anti-Cycling:** Cost/bound perturbation with zero-pivot clean-up phase.
- **Mehrotra Predictor-Corrector Primal-Dual IPM:** High-order affine scaling with Mehrotra adaptive centering for LP and Convex QP.
- **Simplex Crossover:** Pushes interior solutions to exact Basic Feasible Solution (BFS) vertices.
- **GPU / First-Order Acceleration (PDHG):** Accelerated first-order LP engine with CPU dual multiplier certification.

### 2.3 Mixed-Integer Programming Engine (MILP)
- **Hybrid Branch-and-Bound:** Best-Bound priority queue combined with Depth-First Plunging.
- **Branching Rules:** Most-fractional, pseudocost branching, and reliability branching with strong branching budget.
- **Cutting Plane Separators:**
  - Gomory Mixed-Integer (GMI) cuts with numerically safe rounding.
  - Minimal Knapsack Cover cuts.
  - Mixed Integer Rounding (MIR) and c-MIR cuts.
  - Clique and Tank Settling Conflict cuts.
- **Cut Pool Manager:** Efficacy filtering ($\ge 10^{-4}$), orthogonality/parallelism filtering ($\cos \theta \le 0.95$), and cut aging.
- **Primal Heuristics:** Simple Rounding, Fractional Diving, and Feasibility Pump (FP).

### 2.4 Independent Verifier Gate (FR-VER1 – FR-VER2)
Every solution must pass 6 independent validation checks before publication:
1. **Raw Model Primal Feasibility:** $\|(Ax - b)^+\|_\infty \le 10^{-6} \max(1, \|row\|_\infty)$.
2. **Integrality Verification:** $\max_{j \in I} |x_j - \text{round}(x_j)| \le 10^{-6}$.
3. **Kahan Compensated Summation Objective Recomputation** from raw unscaled input data.
4. **Rigorous Dual Bound Proof:** $g(y, \lambda) \ge z^* - 10^{-6}$ (for max) or $g(y, \lambda) \le z^* + 10^{-6}$ (for min).
5. **KKT Complementarity Residual Check:** $\max_i |y_i s_i| \le 10^{-6}$.
6. **Forward Simulation Verifier:** Step-by-step physical mass balance integration for refinery tank schedules.

---

## 3. Industrial Flagship Application: MRPL Refinery Planning

BharatOpt Core includes realistic, MRPL-calibrated models for refinery planning and scheduling:

### 3.1 Level 1: Monthly Crude Selection & Blending (`mrpl_level1.py`)
- Multi-crude basket optimization with spot purchase binaries ($y_j \in \{0, 1\}$).
- **Physical Mass-Weighted Sulfur Balance:** $\sum_j (s_j - S_c^{\max}) \rho_j x_{jc} \le 0$.
- **Non-Linear API Gravity Conversion:** Exact linear Specific Gravity conversion $\text{SG}_c^{\max} = \frac{141.5}{131.5 + \text{API}_c^{\min}}$.
- Finished product yield allocations (LPG, MS, HSD, ATF, FO, Bitumen) and carbon emission proxy caps.

### 3.2 Level 2: 14-Day Tank & CDU Multi-Period Scheduling (`mrpl_level2.py`)
- Dynamic storage tank inventory balances over discrete time periods $t=1..T$.
- **Single-Grade Containment Assignment:** $\sum_j a_{ijt} \le 1$.
- **Receiving & Feeding Mutual Exclusion:** $r_{it} + \sum_c z_{ict} \le 1$.
- **Mandatory Settling Time Quiet Period:** $z_{ict'} + r_{it} \le 1 \quad \forall t' \in [t, t+\tau_i]$.
- **Symmetry Breaking Constraints** for identical storage tank pairs.

---

## 4. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/bharatopt/bharatopt.git
cd SIH1

# Install package in development mode
pip install -e .
```

### Python SDK Example

```python
import bharatopt as bo
from bharatopt.core.model import Sense, ConstraintSense

# 1. Build Canonical Model
model = bo.CanonicalModel(name="RefineryBlending", sense=Sense.MAXIMIZE)
x1 = model.add_var("ArabLight", lb=0.0, ub=150.0, obj=78.0)
x2 = model.add_var("ArabHeavy", lb=0.0, ub=120.0, obj=66.0)

# Add constraints
model.add_constraint({x1: 1.0, x2: 1.0}, ConstraintSense.LE, 200.0, name="CDU_Cap")
model.add_constraint({x1: 1.78*0.858 - 2.0*0.858, x2: 2.85*0.887 - 2.0*0.887}, ConstraintSense.LE, 0.0, name="SulfurCap")

# 2. Solve with Tier T1 Determinism
sol, verifier_report = bo.solve(model, tier=bo.RunTier.T1)

print(f"Status: {sol.status.value}")
print(f"Optimal Margin: ${sol.obj_val:,.2f}")
print(f"Verifier Passed: {verifier_report.passed}")
```

---

## 5. Command-Line Interface (CLI)

```bash
# 1. Run the 10 Hand-Checked Equation Tests from Appendix A
python -m bharatopt.cli.main appendix-a

# 2. Run Gate G0 Clean-Room Sovereignty Audit (Zero 3rd-party solver symbols)
python -m bharatopt.cli.main audit

# 3. Solve an MPS or LP model file in Tier T1 with ExplanationBundle output
python -m bharatopt.cli.main solve mrpl_level1.mps --tier T1 --out mrpl_solve_result.json

# 4. Execute the 5-Gate industrial benchmark suite (Netlib, QPLIB, MIPLIB, MRPL)
python -m bharatopt.cli.main benchmark --suite all

# 5. Launch the Interactive Web Dashboard & Real-Time Trust Stack Visualizer
python -m bharatopt.cli.main ui --port 8000
# -> Open browser to http://127.0.0.1:8000 to interactively solve scenarios,
#    run contrastive "Why Not X?" queries, view Gantt schedules, and inspect verifier proofs.
```

---

## 6. Appendix A: Hand-Checked Mathematical Test Suite (10/10 Passed)

| Test ID | Mathematical Concept | Expected Result | Status |
| :--- | :--- | :--- | :--- |
| **A1** | **Dual Bound Exactness (Max)** | $\max x_1+x_2$ s.t. $x_1+2x_2 \le 4$. $g(0.5)=3.5$ tight, $g(0)=6 \ge 3.5$. | **100% PASS** |
| **A2** | **Sulfur with Density** | Mass-weighted blend $S = 0.5625 \text{ wt}\% > 0.5 \implies +0.80 > 0$. | **100% PASS** |
| **A3** | **API Gravity Non-Linearity** | 50/50 blend API $30.975 < 31.0 \implies +0.0132 > 0$ violated. | **100% PASS** |
| **A4** | **Breakeven Opportunity Cost** | Reduced cost $d_3 = -1.0 \implies \kappa_3^{be} = 2.0$. | **100% PASS** |
| **A5** | **Sulfur Sensitivity Derivative** | Perturbation $\Delta z = y \sum \rho x \cdot 0.01$. | **100% PASS** |
| **A6** | **Tank Settling Constraints** | $\tau=2$, receipt at $t=3 \implies z_3=z_4=z_5=0$; feeding at $t=6$ allowed. | **100% PASS** |
| **A7** | **Symmetry Breaking** | Equal optimal objective; tank-1 receipts $\ge$ tank-2. | **100% PASS** |
| **A8** | **Degenerate LP & Anti-Cycling** | 3 active constraints at $(1,1)$; terminates with $z=2.0$ without cycling. | **100% PASS** |
| **A9** | **Convex QP KKT Exactness** | $\min \frac{1}{2}(x_1^2+x_2^2) - x_1 - 2x_2$. $x^*=(0,1), z^*=-1.5, \lambda=1.0$. | **100% PASS** |
| **A10**| **Knapsack Cover Cut Separation**| Cut $x_1 + x_2 \le 1$ removes fractional LP point $(1, 0.5)$. | **100% PASS** |

---

## 7. Project Structure

```
SIH1/
├── PRD_v5.0_BharatOpt_Core.md       # Complete 30-Page PRD Specification
├── TECHNICAL_SPECIFICATION.md       # In-Depth Engineering & Algorithm Architecture
├── pyproject.toml                   # Standard packaging file
├── bharatopt/                       # Sovereign Solver Package
│   ├── core/                        # Sparse Matrix, Markowitz LU, Cholesky, Presolve
│   ├── solvers/                     # Simplex, Mehrotra IPM, Convex QP, PDHG, B&C MILP
│   ├── verifier/                    # Zero-Trust Verifier Gate & Forward Sim Verifier
│   ├── explanation/                 # Duals, Reduced Costs, Breakevens, Why-Not X
│   ├── run_tiers/                   # T0 Opportunistic, T1 Deterministic, T2 Replay
│   ├── io/                          # MPS, LP, and QPS readers/writers
│   ├── applications/                # MRPL Level 1 & Level 2 refinery formulations
│   ├── ui/                          # FastAPI server & interactive Web Dashboard
│   └── c_api/                       # C ABI header (bharatopt.h) and ctypes interface
├── tests/                           # Complete test suite (Appendix A1-A10, LA, Simplex, etc.)
├── benchmarks/                      # Standard benchmark suites (Netlib, MIPLIB, QPLIB, MRPL)
└── examples/                        # End-to-end refinery L1, L2, and QP walkthrough scripts
```
