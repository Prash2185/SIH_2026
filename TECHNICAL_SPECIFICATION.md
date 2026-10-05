# Technical Architecture & Engineering Specification — BharatOpt Core

**Version:** 5.0.0-PROD  
**Document ID:** BHARATOPT-ENG-SPEC-2026-09  
**Target:** Sovereign LP / QP / MILP Solver with GPU Acceleration & Refinery Domain Models  
**Reference:** Problem Statement SIH26119 (MRPL / MoPNG)

---

## 1. System Architecture Overview

BharatOpt Core is architectured as a clean, decoupled, layered mathematical optimization platform. It enforces a strict separation between model ingestion, linear algebra, relaxation engines, branch-and-cut tree management, postsolve reconstruction, and the independent verifier gate.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   BHARATOPT INTERFACE LAYER                                     │
│     CLI (`bharatopt`)  │  Python SDK (`import bharatopt`)  │  C ABI Shared Lib  │  FastAPI REST │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       ORCHESTRATOR & ROUTER                                     │
│   • Run Tier Manager (T0: Opportunistic | T1: Deterministic | T2: Certified Replay)             │
│   • Problem Classifier (LP / Convex QP / MILP / MINLP Bilinear)                                │
│   • Model SHA-256 Hasher & Canonical Transformer (Min/Max, Slacks, Box Bounds)                 │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     PRESOLVE ENGINE (FR-PRE)                                    │
│   • Singleton Row/Col Elimination              • Bound Tightening & Probing                     │
│   • Fixed Variable Substitution                • Doubleton Equation Aggregation                 │
│   • Dominated Rows & Duplicate Columns         • Presolve Mapping Archive                       │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
┌──────────────────────────────────────────┐    ┌─────────────────────────────────────────────────┐
│           LINEAR ALGEBRA KERNEL          │    │             CONTINUOUS RELAXATION ENGINES       │
│  • Sparse CSC/CSR Compressed Matrices    │    │  • Revised Dual Simplex (Steepest-Edge, Harris) │
│  • Markowitz LU with Threshold Pivoting  │    │  • Revised Primal Simplex (Phase 1/2 Devex)     │
│  • Forrest-Tomlin / Eta Basis Updates    │◄───┤  • Mehrotra Primal-Dual IPM (Predictor-Corr.)   │
│  • Sparse LDLᵀ / Cholesky Factorizer     │    │  • IPM-to-Simplex Crossover (Purification)     │
│  • Ruiz Equilibration & Scaling          │    │  • Convex QP Solver (Augmented KKT Matrix)      │
│  • Compensated Kahan Summation           │    │  • GPU First-Order PDHG Kernel (Accelerated)    │
└──────────────────────────────────────────┘    └────────────────────────┬────────────────────────┘
                                                                         │
                                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MIXED-INTEGER BRANCH-AND-CUT ENGINE                             │
│   • Node Selection: Hybrid Best-Bound + Depth-First Plunging with Best-Estimate Priority        │
│   • Branching Strategies: Most-Fractional, Pseudocost Branching, Reliability Branching          │
│   • Cut Generators: Gomory Mixed-Integer (GMI), MIR / c-MIR, Knapsack Cover, Clique/Conflict    │
│   • Cut Manager: Efficacy Filter (≥1e-4), Orthogonality Filter (dot < 0.9), Ageing & Purging    │
│   • Primal Heuristics: Simple Rounding, Fractional/Guided Diving, Feasibility Pump (FP)         │
│   • Conflict Analysis: Reduced-Cost Variable Fixing, Infeasible Subtree Learning                │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    POSTSOLVE RECONSTRUCTION                                     │
│   • Re-inflation of Presolved Variables and Eliminated Constraints                             │
│   • Dual Solution and Reduced Cost Restoration                                                  │
│   • Basic Feasible Status Recovery                                                              │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 INDEPENDENT VERIFIER GATE (L1 TRUST)                            │
│   • Raw Model Primal Feasibility: ‖(Ax - b)⁺‖_∞ ≤ 10⁻⁶ · max(1, ‖row‖)                         │
│   • Integrality Check: max_{j ∈ I} |x_j - round(x_j)| ≤ 10⁻⁶                                    │
│   • Compensated Summation Objective Recomputation from Unscaled Input Data                      │
│   • Dual Certificate Validity: g(y, λ) ≥ z* (for max) or g(y, λ) ≤ z* (for min)                │
│   • Forward Simulation Verifier for Scheduling (Tank Inventory Balance & Settling Times)       │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              EXPLANATION & ARTIFACT LAYER (L2-L4 TRUST)                         │
│   • JSON ExplanationBundle (Duals, Reduced Costs, Ranging, Why-Not X IIS Analysis)              │
│   • Deterministic Plan SHA-256 Hash Generation & Audit Trail                                    │
│   • Web Dashboard, Search Tree Visualizer & Interactive Gantt Refinery Visualizer               │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Foundations & Kernel Specifications

### 2.1 Canonical Representation
Every linear, quadratic, and mixed-integer problem is canonicalized into:
$$\begin{aligned}
\max_{x \in \mathbb{R}^n} \quad & z(x) = c^T x - \frac{1}{2} x^T Q x \\
\text{s.t.} \quad & A x \le b \quad (m \text{ inequality rows, dual } y \ge 0) \\
& E x = d \quad (p \text{ equality rows, dual } \lambda \text{ free}) \\
& \ell_j \le x_j \le u_j \quad \forall j \in \{1, \dots, n\} \\
& x_j \in \mathbb{Z} \quad \forall j \in \mathcal{I} \subseteq \{1, \dots, n\}
\end{aligned}$$

Minimization problems $\min \tilde{c}^T x + \frac{1}{2} x^T \tilde{Q} x$ are transformed by setting $c = -\tilde{c}$ and $Q = \tilde{Q}$.

### 2.2 Linear Algebra Kernel Design

#### 2.2.1 Sparse Matrix Storage (CSC & CSR)
The engine maintains sparse structures in Compressed Sparse Column (CSC) and Compressed Sparse Row (CSR) formats:
- `CSC`: `col_ptr` (size $n+1$), `row_ind` (size $nnz$), `values` (size $nnz$). Enables efficient column operations for simplex pivot updates and IPM matrix-vector products.
- `CSR`: `row_ptr` (size $m+1$), `col_ind` (size $nnz$), `values` (size $nnz$). Enables ultra-fast row activity evaluations and cut generation.

#### 2.2.2 Sparse LU Factorization with Markowitz Pivoting
Given basis matrix $B \in \mathbb{R}^{m \times m}$, compute $P B Q = L U$:
1. **Markowitz Search:** At step $k$, let $r_i$ be the non-zero count of row $i$ in active submatrix, and $c_j$ be the non-zero count of column $j$. Search candidates minimizing $(r_i - 1)(c_j - 1)$.
2. **Threshold Partial Pivoting:** Ensure numerical stability by requiring:
   $$|a_{ij}| \ge \tau \cdot \max_{k} |a_{kj}|, \quad \tau = 0.1$$
3. **Triangular Solves:** Forward solve $L y = b$ and backward solve $U x = y$ using topological ordering of directed acyclic dependency graphs (DAG).

#### 2.2.3 Basis Updating: Forrest-Tomlin & Eta Updates
When column $j$ enters the basis and replaces column $p$:
- **Eta Matrix Update:** $B_{\text{new}}^{-1} = E_k B_{\text{old}}^{-1}$ where $E_k = I - \frac{(\alpha - e_p) e_p^T}{\alpha_p}$.
- **Forrest-Tomlin Scheme:** Eliminates the spike in row $p$ of $U$ by cyclic row permutations and row-elimination operations, maintaining upper triangular form of $U$ with minimal fill-in.
- **Refactorization Trigger:** Automatically trigger fresh sparse LU factorization when:
  $$\text{update\_count} \ge 50 \quad \text{or} \quad \frac{\|B x - b\|_2}{\|b\|_2} > 10^{-7}$$

#### 2.2.4 Ruiz Equilibration Matrix Scaling
Iteratively computes diagonal row scaling $D_R = \text{diag}(r)$ and column scaling $D_C = \text{diag}(s)$ such that for scaled matrix $\bar{A} = D_R A D_C$:
$$\|\bar{A}_{i, :}\|_\infty \approx 1 \quad \forall i, \quad \|\bar{A}_{:, j}\|_\infty \approx 1 \quad \forall j$$
Damping factor $\theta = 0.5$, iterated for 5 to 10 sweeps.

---

## 3. Optimization Engines

### 3.1 Revised Dual Simplex Engine
- **Pricing:** Dual Steepest-Edge (DSE) weights $\gamma_i = \|(B^{-1})_{i, :}\|_2^2$ updated recurrence:
  $$\gamma_i^{\text{new}} \approx \gamma_i + \frac{v_i^2}{\alpha_p^2} \gamma_p$$
- **Ratio Test:** Harris two-pass ratio test with tolerance $\delta_{\text{feas}} = 10^{-7}$ to avoid zero pivots on degenerate vertices.
- **Degeneracy & Anti-Cycling:** Cost perturbation:
  $$\tilde{c}_j = c_j + \epsilon_j, \quad \epsilon_j \sim \text{Uniform}(10^{-8}, 10^{-7})$$
  Followed by a zero-pivot clean-up phase on original objective.

### 3.2 Mehrotra Primal-Dual Interior-Point Method (LP & QP)
For convex quadratic programming $\max c^T x - \frac{1}{2} x^T Q x$ s.t. $Ax + s = b, x \ge 0, s \ge 0$:
1. **KKT Residuals:**
   $$\begin{aligned}
   r_p &= b - Ax - s \quad (\text{Primal Infeasibility}) \\
   r_d &= c - Qx - A^T y - z \quad (\text{Dual Infeasibility}) \\
   \mu &= \frac{x^T z + s^T w}{2n} \quad (\text{Complementarity Measure})
   \end{aligned}$$
2. **Predictor Step (Affine Scaling):** Solve KKT system with $\sigma = 0$:
   $$\begin{bmatrix} -(Q + X^{-1}Z) & A^T \\ A & S^{-1}W \end{bmatrix} \begin{bmatrix} \Delta x^{\text{aff}} \\ \Delta y^{\text{aff}} \end{bmatrix} = \begin{bmatrix} r_d - X^{-1}(XZe) \\ r_p \end{bmatrix}$$
3. **Centering Parameter:** $\sigma = \left( \frac{\mu_{\text{aff}}}{\mu} \right)^3$.
4. **Corrector & Centering Step:** Solve augmented KKT system with right-hand side augmented with second-order term $\Delta X^{\text{aff}} \Delta Z^{\text{aff}} e - \sigma \mu e$.
5. **Step Length:** Compute maximum ratio step $\alpha_p, \alpha_d \in (0, 1)$ keeping $(x, s, z, w) > 0$.

### 3.3 Simplex Crossover (Purification)
1. **Status Assignment:** Variables with $x_j \le \ell_j + \epsilon$ fixed at Lower; $x_j \ge u_j - \epsilon$ fixed at Upper; intermediate variables designated as Basic candidates.
2. **Push Steps:** Execute dual simplex pivots to eliminate artificial basic slack variables, restoring an exact extreme point vertex basis $B$.

---

## 4. Mixed-Integer Branch-and-Cut Engine

```
                                  [ Root Node ]
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                  [ Solve Root LP ]             [ Presolve MIP ]
                         │
                         ▼
             ┌───────────────────────┐
             │ Cut Separation Loop   │◄────────────────────────┐
             │ • Gomory Mixed-Int    │                         │
             │ • Knapsack Cover      │ (Gap closed > 0.001     │
             │ • MIR / c-MIR         │  and round < MaxRounds) │
             │ • Clique Conflict     │                         │
             └───────────┬───────────┘                         │
                         │                                     │
                         ▼                                     │
                  [ Re-solve LP ] ─────────────────────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │ Primal Heuristics     │
             │ • Simple Rounding     │ ──► [ New Incumbent? ] ──► [ Verifier Gate ]
             │ • Fractional Diving   │
             │ • Feasibility Pump    │
             └───────────┬───────────┘
                         │
                         ▼
             [ Reliability Branching ]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
[ Child Node: x_j ≤ ⌊x*⌋ ]       [ Child Node: x_j ≥ ⌈x*⌉ ]
        │                                 │
        ▼                                 ▼
   (Best-Bound Priority Queue with Hybrid Depth-First Plunging)
```

### 4.1 Cutting Plane Mathematics

#### 4.1.1 Gomory Mixed-Integer (GMI) Cuts
From an optimal tableau row corresponding to basic integer variable $x_i = \beta_i + \sum_{j \in N} \bar{a}_{ij} x_j$, where $\beta_i \notin \mathbb{Z}$ with fractional part $f_0 = \beta_i - \lfloor \beta_i \rfloor > 0$:
$$\sum_{j \in N} \psi( \bar{a}_{ij} ) x_j \ge 1$$
where the GMI cut function $\psi(\alpha)$ for $f = \alpha - \lfloor \alpha \rfloor$ is:
$$\psi(\alpha) = \begin{cases} 
\frac{f}{f_0}, & \text{if } f \le f_0 \\
\frac{1 - f}{1 - f_0}, & \text{if } f > f_0 
\end{cases} \quad (\text{for integer } x_j), \qquad \psi(\alpha) = \begin{cases}
\frac{\alpha}{f_0}, & \text{if } \alpha \ge 0 \\
\frac{-\alpha}{1 - f_0}, & \text{if } \alpha < 0
\end{cases} \quad (\text{for continuous } x_j)$$

#### 4.1.2 Minimal Knapsack Cover Cuts
For a binary knapsack inequality $\sum_{j \in C} a_j x_j \le b$, a subset $C \subseteq N$ is a cover if $\sum_{j \in C} a_j > b$. It is minimal if $\sum_{j \in C \setminus \{k\}} a_j \le b$ for all $k \in C$.
The valid cover cut is:
$$\sum_{j \in C} x_j \le |C| - 1$$

### 4.2 Primal Heuristics

#### 4.2.1 Feasibility Pump (FP)
Alternates between finding an integer point $\tilde{x} = \text{round}(x^*)$ and solving a projection LP:
$$\min_{x \in P} \Delta(x, \tilde{x}) = \sum_{j \in \mathcal{I}} |x_j - \tilde{x}_j|$$
If $\Delta(x^*, \tilde{x}) = 0$, an integer feasible point is discovered. If cycling occurs, a random perturbation flip of fractionally biased coordinates is executed.

---

## 5. GPU First-Order Acceleration (PDHG / PDLP Style)

For massive continuous LPs and root relaxation screening:
1. **Primal-Dual Hybrid Gradient Update:**
   $$\begin{aligned}
   x^{k+1} &= \text{proj}_{[\ell, u]} \left( x^k - \tau (c - A^T y^k) \right) \\
   \bar{x}^{k+1} &= 2 x^{k+1} - x^k \\
   y^{k+1} &= \text{proj}_{\mathbb{R}_+^m} \left( y^k + \sigma (A \bar{x}^{k+1} - b) \right)
   \end{aligned}$$
2. **Adaptive Step Size & Restarts:** Steps $\tau, \sigma$ dynamically scaled to maintain primal/dual convergence symmetry. Halting on normalized duality residual $< 10^{-6}$.
3. **Rigorous Dual Certification on CPU:** Multipliers $\bar{y}$ generated by PDHG are plugged into the CPU dual bound function:
   $$g(\bar{y}) = b^T \bar{y} + \sum_{j=1}^n \left( r_j^+ u_j - r_j^- \ell_j \right), \quad r = c - A^T \bar{y}$$
   This ensures no first-order approximation ever corrupts the global dual bound without rigorous CPU mathematical certification.

---

## 6. Independent Verifier Gate Specification

The verifier executes completely independently from solver internals:
```python
def verify_solution(model: CanonicalModel, sol: Solution) -> VerifierReport:
    # 1. Primal feasibility
    row_violations = []
    for i, row in enumerate(model.A):
        activity = sum(row.val[k] * sol.x[row.col[k]] for k in range(len(row.val)))
        scale = max(1.0, max(abs(v) for v in row.val))
        violation = max(0.0, activity - model.b[i]) / scale
        if violation > 1e-6:
            row_violations.append((i, violation))
            
    # 2. Integrality check
    int_violations = []
    for j in model.integer_indices:
        err = abs(sol.x[j] - round(sol.x[j]))
        if err > 1e-6:
            int_violations.append((j, err))
            
    # 3. Kahan compensated summation objective recomputation
    recomputed_obj = kahan_sum([model.c[j] * sol.x[j] for j in range(model.num_vars)])
    
    # 4. Valid dual bound check
    is_bound_valid = sol.dual_bound >= recomputed_obj - 1e-6  # For maximization
    
    return VerifierReport(
        passed = (len(row_violations) == 0 and len(int_violations) == 0 and is_bound_valid),
        max_row_violation = max((v for _, v in row_violations), default=0.0),
        max_int_violation = max((v for _, v in int_violations), default=0.0),
        recomputed_obj = recomputed_obj,
        dual_bound_valid = is_bound_valid
    )
```

---

## 7. ExplanationBundle Schema (JSON Contract)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "BharatOpt ExplanationBundle",
  "type": "object",
  "required": ["run_id", "model_hash", "status", "objective", "dual_bound", "gap", "verifier", "sensitivity"],
  "properties": {
    "run_id": { "type": "string" },
    "model_hash": { "type": "string" },
    "solver_tier": { "type": "string", "enum": ["T0", "T1", "T2"] },
    "status": { "type": "string", "enum": ["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED", "NUMERICAL", "TIME_LIMIT"] },
    "objective": { "type": "number" },
    "dual_bound": { "type": "number" },
    "gap": { "type": "number" },
    "iterations": { "type": "integer" },
    "node_count": { "type": "integer" },
    "solve_time_sec": { "type": "number" },
    "verifier": {
      "type": "object",
      "properties": {
        "passed": { "type": "boolean" },
        "primal_feasibility_residual": { "type": "number" },
        "integrality_residual": { "type": "number" },
        "kkt_complementarity_residual": { "type": "number" },
        "recomputed_objective": { "type": "number" }
      }
    },
    "sensitivity": {
      "type": "object",
      "properties": {
        "shadow_prices": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "row_name": { "type": "string" },
              "dual_value": { "type": "number" },
              "rhs_low": { "type": "number" },
              "rhs_up": { "type": "number" }
            }
          }
        },
        "reduced_costs": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "var_name": { "type": "string" },
              "value": { "type": "number" },
              "reduced_cost": { "type": "number" },
              "breakeven_cost": { "type": "number" }
            }
          }
        },
        "why_not_x": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "hypothesis": { "type": "string" },
              "feasible": { "type": "boolean" },
              "objective_delta": { "type": "number" },
              "limiting_constraints": { "type": "array", "items": { "type": "string" } }
            }
          }
        }
      }
    }
  }
}
```

---

## 8. C ABI Header Specification (`bharatopt.h`)

```c
#ifndef BHARATOPT_H
#define BHARATOPT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BHARATOPT_STATUS_OPTIMAL      = 0,
    BHARATOPT_STATUS_FEASIBLE     = 1,
    BHARATOPT_STATUS_INFEASIBLE   = 2,
    BHARATOPT_STATUS_UNBOUNDED    = 3,
    BHARATOPT_STATUS_NUMERICAL    = 4,
    BHARATOPT_STATUS_TIME_LIMIT   = 5,
    BHARATOPT_STATUS_WORK_LIMIT   = 6
} BharatOptStatus;

typedef enum {
    BHARATOPT_TIER_T0 = 0,
    BHARATOPT_TIER_T1 = 1,
    BHARATOPT_TIER_T2 = 2
} BharatOptRunTier;

typedef struct {
    int num_vars;
    int num_constraints;
    int num_nz;
    const double* c;
    const double* lb;
    const double* ub;
    const int* var_types; // 0: Continuous, 1: Integer, 2: Binary
    const int* A_row_ptr;
    const int* A_col_ind;
    const double* A_val;
    const char* sense;    // '<', '>', '='
    const double* rhs;
} BharatOptModel;

typedef struct {
    BharatOptStatus status;
    double objective_value;
    double best_dual_bound;
    double relative_gap;
    int iterations;
    int node_count;
    double solve_time_ms;
    double* x;
    double* duals;
    double* reduced_costs;
    int verifier_passed;
    char model_hash[65];
} BharatOptSolution;

// Core Entrypoints
void* bharatopt_create_solver(void);
void bharatopt_free_solver(void* solver);
int bharatopt_load_model(void* solver, const BharatOptModel* model);
BharatOptSolution bharatopt_solve(void* solver, BharatOptRunTier tier);
void bharatopt_free_solution(BharatOptSolution* sol);
const char* bharatopt_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // BHARATOPT_H
```
