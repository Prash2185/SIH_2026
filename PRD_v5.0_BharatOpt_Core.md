# PRD v5.0 — BharatOpt Core: A From-Scratch Sovereign LP / QP / MILP Solver with GPU Acceleration

**Problem Statement ID:** SIH26119  
**Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL) / Ministry of Petroleum and Natural Gas  
**Status:** Baseline Specification & Production Architecture Document  
**Date:** September 30, 2026  
**Clean-Room Stance:** Zero third-party solver-library dependencies linked or shipped.

---

## 0. Executive Summary & Evolution from v4.0

### 0.1 Paradigm Shift
Earlier architectural baselines (v4.0) approached industrial refinery scheduling as an application layer integrated on top of existing third-party open-source solvers (e.g., SCIP, SoPlex, HiGHS, PaPILO, cuOpt). **Problem Statement SIH26119 explicitly mandates sovereign national capability**: an optimization solver core engineered from foundational mathematics, with source-level transparency, determinism guarantees, on-premise air-gapped security, and specialized acceleration for high-stakes downstream energy infrastructure.

BharatOpt Core v5.0 inverts the priorities:
1. **Core Solver Engine First:** Sovereign implementations of Linear Algebra, Presolve, Revised Dual Simplex, Primal Simplex, Interior-Point Predictor-Corrector (IPM), Branch-and-Bound/Cut (MILP), and Convex Quadratic Programming (QP).
2. **Strict Clean-Room Policy:** No code copied, linked, or wrapped from third-party solver libraries (SCIP, HiGHS, CBC, GLPK, OSQP, Ipopt). Third-party solvers serve strictly as external reference oracles in validation test harnesses.
3. **Four-Layer Trust Architecture:** Solutions must be rigorously verified before publication. An unverified answer is never returned.
4. **Hardware & Algorithmic Scalability:** Hybrid CPU deterministic branch-and-cut combined with GPU first-order (PDHG/PDLP-style) acceleration for massive scale and primal heuristics.
5. **Industrial Flagship Benchmark:** MRPL refinery crude selection (Level 1) and CDU/Tank scheduling (Level 2) with rigorous physical balances, non-linear API/SG density conversions, and settling rules.

---

## 1. Vision, Positioning & Trust Architecture

### 1.1 Vision
To deliver India’s first indigenous, enterprise-grade mathematical programming solver capable of solving large-scale, sparse, degenerate, and ill-conditioned LP, QP, and MILP industrial instances with mathematical provability, absolute determinism, and sovereign execution control.

### 1.2 Honest Parity & Performance Statement
Commercial engines (CPLEX, Gurobi, Xpress) represent 35+ years of continuous engineering. BharatOpt Core v5.0 commits to honest, verified engineering:
- **Correctness and Robustness First:** No candidate solution is published unless certified by the independent verifier gate.
- **Competitive Open-Source Parity:** Target performance within $3\times$ to $5\times$ shifted geometric mean of premier open-source solvers on standard benchmark suites (Netlib, MIPLIB 2017 easy/benchmark subsets, QPLIB).
- **Structure & Hardware Exploit:** Superior throughput on structured batched what-if scenarios, massive continuous LPs via GPU first-order PDHG, and parallel MIP heuristics.

### 1.3 Four-Layer Trust Architecture
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

## 2. Compliance Matrix (SIH26119)

| PS Requirement | Architectural Implementation | Verification & Acceptance Standard |
| :--- | :--- | :--- |
| **Sovereign Solver Core** | From-scratch C++/Python numerical kernels, linear algebra, and branch-and-cut | Zero external solver libraries linked. Full SBOM compliance (Gate G0). |
| **LP / QP / MILP Scope** | Dual Simplex, Primal Simplex, Mehrotra IPM, B&C with Gomory/MIR/Cover cuts, Convex QP | Solves Netlib LP, QPLIB convex, MIPLIB 2017 subsets with verified residuals $< 10^{-6}$. |
| **Extensible to MIQP/NLP** | Modular abstraction for relaxations, expression graph layer, and McCormick bilinear bounds | Extensible plugin API; Level 2-B bilinear tank pooling prototype. |
| **Sparse Linear Algebra** | Custom Sparse LU with Markowitz pivoting, Forrest-Tomlin update, Sparse $LDL^T$/Cholesky | Hand-checked numerical tests (App. A), condition number tracking. |
| **Degeneracy & Ill-Conditioning** | Perturbation with clean-up, Harris ratio test, Ruiz equilibration scaling, iterative refinement | Degeneracy demonstration suite; 0 cycles on assignment/cycling LPs. |
| **Multi-Core & GPU Acceleration** | Deterministic parallel B&B (T1), GPU PDHG first-order LP kernel, GPU primal diving | Speedup measured and verified; CPU recomputes valid dual bound $g(y,\lambda)$. |
| **Refinery Applications** | Level 1 Monthly Crude Selection & Level 2 Tank/CDU Scheduling models | Reproduces physical mass/volume balances, settling times, and gravity non-linearities. |
| **API / CLI / Interfaces** | Native CLI (`bharatopt`), Python SDK, C ABI headers, REST API, Web UI dashboard | Standard MPS/LP/QPS file I/O roundtrip without data loss. |

---

## 3. Clean-Room Policy & Provenance Protocol

1. **Forbidden Artifacts:** No source code, static/shared libraries, or wrappers from SCIP, SoPlex, PaPILO, HiGHS, CBC, GLPK, OSQP, Ipopt, or cuOpt may be bundled or linked in `bharatopt`.
2. **Independent Reference Oracles:** External solvers are permitted solely in the off-line test harness as black-box subprocesses to verify objective agreement and runtime benchmarks.
3. **Algorithm Provenance:** Every mathematical routine is coded directly from published literature (Dantzig, Lemke, Forrest-Tomlin, Mehrotra, Gomory, Marchand, Wolsey, Chambolle-Pock, Applegate et al.).
4. **Gate G0 Automated Audit:** Automated CI scan verifying zero third-party solver symbols in release binaries.

---

## 4. User Personas & User Stories

- **Refinery Optimization Planner (MRPL):** Needs to run monthly crude basket optimization and 14-day tank scheduling with complete trust in volume balances, sulfur constraints, and vessel arrival windows.
- **Process & Operations Engineer:** Needs transparent shadow prices (duals), reduced costs for unselected crude slates, and contrastive "Why not crude X?" explanations.
- **OR Researcher / Core Developer:** Needs inspectable sparse LU factorizations, cut separation callbacks, and extensible node selection strategies.
- **IT / Air-Gap Security Auditor:** Requires standalone, offline execution with deterministic SHA-256 run logs and zero external network calls.

---

## 5. Functional Requirements

### 5.1 Model Representation and I/O (FR-IO)
- **FR-IO1 (Formats):** Read and write MPS (fixed and free format), CPLEX LP format, and QPS format.
- **FR-IO2 (Data Structure):** Unified canonical model representation:
  $$\max \; c^T x - \frac{1}{2} x^T Q x \quad \text{s.t.} \quad A x \le b, \quad E x = d, \quad \ell \le x \le u, \quad x_j \in \mathbb{Z} \; \forall j \in I$$
- **FR-IO3 (Deterministic Hashing):** Compute SHA-256 fingerprint of the canonical model matrix, vectors, and variable classifications.
- **FR-IO4 (Input Validation):** Explicit dimension matching, bound validity ($\ell_j \le u_j$), and warning diagnostics for matrix dynamic ranges exceeding $10^6$.

### 5.2 Linear Algebra Kernel (FR-LA)
- **FR-LA1 (Sparse LU):** Sparse LU factorization with Markowitz count ordering and threshold partial pivoting ($\tau \in [0.01, 0.1]$).
- **FR-LA2 (Basis Updating):** Forrest-Tomlin and Product-Form (Eta) updating with numerical stability checks triggering periodic refactorization.
- **FR-LA3 (Sparse Cholesky / $LDL^T$):** Symmetric sparse factorizer with Approximate Minimum Degree (AMD) ordering and diagonal regularization $\delta \sim 10^{-12}$ for IPM KKT systems.
- **FR-LA4 (Matrix Scaling):** Ruiz geometric equilibration scaling to balance row and column infinity norms.
- **FR-LA5 (Iterative Refinement):** Residual monitoring and iterative correction for ill-conditioned basis systems.

### 5.3 Presolve & Postsolve (FR-PRE)
- **FR-PRE1 (Basic Reductions):** Singleton rows/columns, empty rows/columns, fixed variable elimination, redundant bound tightening.
- **FR-PRE2 (Advanced Linear Reductions):** Dominated column detection, doubleton equation substitutions, implied free variables.
- **FR-PRE3 (MIP Reductions):** Coefficient tightening on binary inequalities, probing, and clique detection.
- **FR-PRE4 (Exact Postsolve):** Reconstruct original primal solution $x$, dual solution $y$, reduced costs $r$, and basis statuses with verifier gate validation.

### 5.4 Linear Programming Engines (FR-LP)
- **FR-LP1 (Revised Dual Simplex):** Bounded revised dual simplex with dual steepest-edge pricing and Harris two-pass ratio test.
- **FR-LP2 (Revised Primal Simplex):** Primal simplex with devex/steepest-edge pricing and Phase 1 artificial variable handling.
- **FR-LP3 (Degeneracy Handling):** Dynamic cost and bound perturbation with un-perturbing clean-up phase to guarantee anti-cycling.
- **FR-LP4 (Interior-Point Method):** Mehrotra predictor-corrector primal-dual IPM with adaptive centering parameter $\sigma$ and step length damping $\alpha \in (0, 1)$.
- **FR-LP5 (Simplex Crossover):** Basis purification and push-crossover from interior solutions to exact basic vertex solutions.
- **FR-LP6 (Infeasibility & Unboundedness Certificates):** Farkas certificate extraction for primal infeasibility; extreme ray extraction for unboundedness.
- **FR-LP7 (Sensitivity & Ranging):** Exact basis inverse sensitivity ranges for RHS $b$ and objective coefficients $c$.

### 5.5 Convex Quadratic Programming (FR-QP)
- **FR-QP1 (Convex QP IPM):** Primal-dual IPM solving the augmented KKT system:
  $$\begin{bmatrix} -(Q + \Theta^{-1}) & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix} = \begin{bmatrix} r_d \\ r_p \end{bmatrix}$$
- **FR-QP2 (Convexity Verification):** Pre-solve positive semi-definiteness ($Q \succeq 0$) verification via Cholesky pivot factorization.
- **FR-QP3 (Wolfe Dual Bound):** Evaluate exact Wolfe dual bound:
  $$g_{QP}(y, \lambda) = b^T y + d^T \lambda - \frac{1}{2} x^T Q x + \sum_j r_j(x)$$

### 5.6 Mixed-Integer Programming Engine (FR-MIP)
- **FR-MIP1 (Branch-and-Cut Loop):** LP-relaxation based B&C with hybrid Best-Bound + Depth-First Plunging node selection.
- **FR-MIP2 (Branching Rules):** Most-fractional baseline, pseudocost branching, and reliability branching with strong branching budget.
- **FR-MIP3 (Cutting Planes):**
  - Gomory Mixed-Integer (GMI) cuts from simplex tableau with numerically safe rounding.
  - Mixed Integer Rounding (MIR) and c-MIR cuts.
  - Knapsack Cover and Lifted Cover cuts.
  - Clique and Settling-time conflict cuts.
- **FR-MIP4 (Cut Pool Management):** Orthogonality/parallelism filtering, efficacy thresholding ($\ge 10^{-4}$), and cut aging/purging.
- **FR-MIP5 (Primal Heuristics):** Simple Rounding, Guided/Fractional Diving, and Feasibility Pump (FP).
- **FR-MIP6 (Dual Bound Tracking):** Rigorous global dual bound updates derived strictly from valid node bounds.
- **FR-MIP7 (Conflict Analysis & Node Presolve):** Infeasible node clause learning and reduced-cost variable fixing ($r_j > \text{Gap} \implies x_j = \ell_j$).

### 5.7 GPU & Parallel Acceleration (FR-GPU / FR-PAR)
- **FR-GPU1 (GPU First-Order LP):** Native CUDA/vectorized PDHG (Primal-Dual Hybrid Gradient) algorithm with adaptive steps and restarts.
- **FR-GPU2 (Dual Bound Certification):** Dual multipliers $\bar{y}$ produced by GPU PDHG are evaluated on CPU to compute a rigorous valid dual bound:
  $$g(\bar{y}) = b^T \bar{y} + \sum_j \left( r_j^+ u_j - r_j^- \ell_j \right), \quad r = c - A^T \bar{y}$$
- **FR-GPU3 (Automatic Fallback):** Seamless CPU fallback if GPU memory or convergence thresholds are exceeded.
- **FR-PAR1 (Deterministic Multi-Core):** T1 deterministic parallel tree search with fixed work partitions and ordered synchronizations.

### 5.8 Verification & Trust Gate (FR-VER)
- **FR-VER1 (Independent Verifier):** Standalone gate checking:
  1. Primal constraint feasibility: $\|(Ax - b)^+\|_\infty \le 10^{-6} \max(1, \|row\|_\infty)$.
  2. Integrality tolerance: $\max_{j \in I} |x_j - \text{round}(x_j)| \le 10^{-6}$.
  3. Compensated summation (Kahan) objective recomputation from raw un-scaled data.
  4. Dual feasibility and KKT complementarity for continuous relaxations.
  5. Dual bound validity check: $g(y) \ge z^* - 10^{-6}$ (for max) or $g(y) \le z^* + 10^{-6}$ (for min).
- **FR-VER2 (Refinery Simulation Verifier):** Forward time-step inventory integration for tank balances and CDU feed rates.

### 5.9 Explainability & XAI (FR-XAI)
- **FR-XAI1 (Attribution):** Shadow prices (duals), reduced costs, and constraint slacks.
- **FR-XAI2 (Breakeven Analysis):** Opportunity cost calculation for unselected crude slates ($\kappa_j^{be} = \kappa_j + d_j$).
- **FR-XAI3 (Contrastive "Why Not X?"):** IIS (Irreducible Inconsistent Subsystem) extraction on forced selection scenarios ($x_j \ge 1 \land c^Tx \ge z^*$).
- **FR-XAI4 (Structured ExplanationBundle):** Standard JSON contract detailing solver outcomes, verifier certificates, and sensitivity metrics.

---

## 6. Execution Modes & Run Tiers

| Tier | Name | Target Use Case | Determinism & Execution Rules |
| :--- | :--- | :--- | :--- |
| **T0** | **Opportunistic** | Real-time what-ifs, rapid prototyping | Non-deterministic multi-threading, wall-clock limits permitted, portfolio racing enabled. |
| **T1** | **Deterministic** | Official refinery schedules & published plans | Strict determinism; work unit / iteration limits only; pinned random seed; identical SHA-256 plan hash across 100 runs on identical hardware. |
| **T2** | **Certified Replay** | Regulatory compliance & audit trail | Archives integer basis, cut indices, and exact branch trace; allows 1-click verified replay without tree search. |

---

## 7. Mathematical & Algorithmic Formulation

### 7.1 Canonical Problem Representation
$$\begin{aligned}
\max_{x} \quad & c^T x - \frac{1}{2} x^T Q x \\
\text{s.t.} \quad & A x \le b \quad (\text{Dual } y \ge 0) \\
& E x = d \quad (\text{Dual } \lambda \text{ free}) \\
& \ell \le x \le u \quad (x_j \in \mathbb{Z} \; \forall j \in I)
\end{aligned}$$
*(Minimization problems are transformed internally by negating $c$ and $Q$.)*

### 7.2 Rigorous Valid Dual Bound Formulation
For any primal feasible point $x$, any dual multipliers $y \ge 0$, and any unconstrained multipliers $\lambda$:
$$c^T x \le c^T x + y^T (b - Ax) + \lambda^T (d - Ex) = b^T y + d^T \lambda + r^T x$$
where reduced costs $r = c - A^T y - E^T \lambda$.  
Maximizing $r^T x$ over box constraints $\ell \le x \le u$ yields the exact dual bound function:
$$g(y, \lambda) = b^T y + d^T \lambda + \sum_{j=1}^n \left( r_j^+ u_j - r_j^- \ell_j \right)$$
where $r_j^+ = \max(r_j, 0)$ and $r_j^- = \max(-r_j, 0)$.  
If $r_j > 0$ with $u_j = +\infty$ or $r_j < 0$ with $\ell_j = -\infty$, the bound evaluates to $+\infty$ (valid but uninformative).

### 7.3 Optimality Gap Metric
$$\text{Relative Gap} = \frac{|\text{UB} - \text{LB}|}{\max\left(1.0, \, |\text{UB}|\right)}$$
where $\text{LB}$ is the verified primal incumbent and $\text{UB}$ is the best valid dual bound across all open branch-and-bound nodes.

---

## 8. Refinery Application Benchmark Suite (MRPL Focus)

### 8.1 Level 1: Monthly Crude Selection & Blending Model
- **Variables:**
  - $x_{jc} \ge 0$: Volume of crude $j$ processed in Crude Distillation Unit (CDU) $c$ (kbbl/month).
  - $s_j \ge 0$: Closing inventory of crude $j$.
  - $y_j \in \{0, 1\}$: Binary decision to purchase spot cargo of crude $j$.
  - $w_p \ge 0$: Production volume of finished refined product $p$ (e.g., LPG, MS/Gasoline, HSD/Diesel, ATF/Jet, FO, Bitumen).
- **Objective:**
  $$\max \quad \sum_p \pi_p w_p - \sum_{j,c} (\kappa_j + \gamma_c) x_{jc} - \sum_{j \in J_S} f_j y_j - \sum_j h_j s_j$$
- **Constraints:**
  1. **Crude Stock Balance:** $\sum_c x_{jc} + s_j = s_j^0 + q_j + A_j y_j \quad \forall j$
  2. **Total Storage Capacity:** $\sum_j s_j \le S^{\max}$
  3. **CDU Throughput Capacity:** $K_c^{\min} \le \sum_j x_{jc} \le K_c^{\max} \quad \forall c$
  4. **Crude-CDU Metallurgy Allowability:** $x_{jc} = 0 \quad \forall (j, c) \in \text{Forbidden}$ (e.g., High-TAN / High-Sulfur limitations)
  5. **Product Yields:** $w_p = \sum_{j, c} \eta_{pj} x_{jc} \quad \forall p$
  6. **Product Commercial Demand:** $D_p^{\min} \le w_p \le D_p^{\max} \quad \forall p$
  7. **Mass-Weighted Sulfur Balance:** $\sum_j \left( s_j - S_c^{\max} \right) \rho_j x_{jc} \le 0 \quad \forall c$
  8. **Specific Gravity (Non-Linear API Conversion):**
     $$\text{SG}_c^{\max} = \frac{141.5}{131.5 + \text{API}_c^{\min}} \implies \sum_j \left( \text{SG}_j - \text{SG}_c^{\max} \right) x_{jc} \le 0 \quad \forall c$$
  9. **Carbon / Feed Emissions Proxy:** $\sum_{j,c} \text{ef}_j x_{jc} \le E_{\text{cap}}$

### 8.2 Level 2: 14-Day Tank & CDU Scheduling Model
- **Variables:**
  - $a_{ijt} \in \{0,1\}$: Tank $i$ holds crude grade $j$ at time period $t$.
  - $I_{ijt} \ge 0$: Inventory of grade $j$ in tank $i$ at period $t$.
  - $in_{ijt} \ge 0$: Inflow to tank $i$ of grade $j$ at period $t$ (from tanker/pipeline).
  - $out_{ijct} \ge 0$: Flow from tank $i$ of grade $j$ to CDU $c$ at period $t$.
  - $r_{it} \in \{0, 1\}$: Tank $i$ is in receiving mode at period $t$.
  - $z_{ict} \in \{0, 1\}$: Tank $i$ is feeding CDU $c$ at period $t$.
- **Key Constraints:**
  1. **Single Grade Assignment:** $\sum_j a_{ijt} \le 1 \quad \forall i, t$
  2. **Holding Bounds:** $I_{ijt} \le C_i a_{ijt} \quad \forall i, j, t$
  3. **Mass Inventory Balance:** $I_{ijt} = I_{ij,t-1} + in_{ijt} - \sum_c out_{ijct} \quad \forall i, j, t$
  4. **Receiving & Feeding Mutual Exclusion:** $r_{it} + \sum_c z_{ict} \le 1 \quad \forall i, t$
  5. **Mandatory Settling Time ($\tau_i$ periods):**
     $$z_{ict'} + r_{it} \le 1 \quad \forall t' \in [t, \min(T, t+\tau_i)], \quad \forall i, c$$
  6. **Symmetry Breaking Constraints:** For identical storage tanks $i_1, i_2$, enforce lexicographic flow ordering $\sum_t out_{i_1,jt} \ge \sum_t out_{i_2,jt}$.

---

## 9. Appendix A: Hand-Checked Mathematical Test Suite

Every release of BharatOpt Core must pass 100% of the hand-checked equation tests defined in Appendix A:

1. **A1 (Dual Bound Exactness):** Maximize $x_1 + x_2$ s.t. $x_1 + 2x_2 \le 4, 0 \le x_1, x_2 \le 3$. Optimal $x^* = (3, 0.5)$, value $3.5$. Verify $g(y) \ge 3.5$ for all $y \ge 0$.
2. **A2 (Sulfur Mass Averaging with Density):** Validate mass-weighted sulfur blend with densities $\rho_A=0.12, \rho_B=0.14$, confirming linear sulfur blending via mass rather than raw volume.
3. **A3 (API Gravity Non-Linearity):** Validate conversion between API and Specific Gravity ($\text{SG} = \frac{141.5}{131.5 + \text{API}}$). Prove that 50/50 volumetric blend of API 33 and API 29 gives API $30.975 < 31.0$, properly flagging volumetric linear averaging as invalid.
4. **A4 (Reduced Cost Breakeven):** Maximize $3x_1 + 2x_2 + x_3$ s.t. $x_1 + x_2 + x_3 \le 4, 0 \le x_i \le 3$. Verify reduced costs $d = (1, 0, -1)$ and breakeven cost for non-basic $x_3$ is exactly $\kappa_3^{be} = 2.0$.
5. **A5 (Sulfur Sensitivity Derivative):** Verify finite-difference perturbation $\frac{\partial z^*}{\partial S^{\max}} = y \sum \rho x$.
6. **A6 (Tank Settling Mutual Exclusion):** For $\tau=2$, receipt at $t=3$ enforces feeding variables $z_3=z_4=z_5=0$, allowing feeding at $t=6$.
7. **A7 (Symmetry Breaking):** Verify that symmetric tank constraints preserve optimal objective value while breaking node enumeration symmetry.
8. **A8 (Degenerate LP Anti-Cycling):** Maximize $x_1 + x_2$ s.t. $x_1 \le 1, x_2 \le 1, x_1 + x_2 \le 2$ (3 constraints active at $(1,1)$). Dual simplex with perturbation terminates cleanly without cycling.
9. **A9 (Convex QP KKT Check):** Minimize $\frac{1}{2}(x_1^2 + x_2^2) - x_1 - 2x_2$ s.t. $x_1 + x_2 \le 1, x \ge 0$. KKT optimal solution $x^* = (0, 1)$, objective $-1.5$, Lagrange multiplier $\lambda = 1.0$.
10. **A10 (Knapsack Cover Cut Validity):** For $5x_1 + 6x_2 \le 8, x \in \{0, 1\}^2$, verify minimal cover cut $x_1 + x_2 \le 1$ removes fractional LP point $(1, 0.5)$ while preserving all integer feasible points.

---

## 10. Non-Functional Requirements & Security

- **Air-Gapped Operation:** Zero internet connectivity required. No telemetry or phone-home requests.
- **Portability:** Pure Python & optimized C-extensions compatible with Windows, Linux (RHEL/Ubuntu), and CUDA 12.x environments.
- **Numerical Precision:** 64-bit IEEE 754 floating point arithmetic with Kahan compensated summation for objective evaluations.
- **Auditability:** Complete JSON ExplanationBundle including model hash, run parameters, simplex iteration counts, cut statistics, and verifier certificate.
