/**
 * BharatOpt Core - C ABI Interface Header (FR-API1)
 * Stable C ABI for embedding in high-performance C/C++, Fortran, and SCADA control applications.
 */

#ifndef BHARATOPT_CORE_H
#define BHARATOPT_CORE_H

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
    BHARATOPT_TIER_T0 = 0,  // Opportunistic
    BHARATOPT_TIER_T1 = 1,  // Deterministic (Default)
    BHARATOPT_TIER_T2 = 2   // Certified Replay
} BharatOptRunTier;

typedef enum {
    BHARATOPT_VAR_CONTINUOUS = 0,
    BHARATOPT_VAR_INTEGER    = 1,
    BHARATOPT_VAR_BINARY     = 2
} BharatOptVarType;

typedef struct {
    int num_vars;
    int num_ineq;
    int num_eq;
    int num_nz_ineq;
    int num_nz_eq;
    const double* c;
    const double* lb;
    const double* ub;
    const int* var_types;
    const int* A_row_ptr;
    const int* A_col_ind;
    const double* A_val;
    const double* b;
    const int* E_row_ptr;
    const int* E_col_ind;
    const double* E_val;
    const double* d;
    int is_maximize;
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

void* bharatopt_create_solver(void);
void bharatopt_free_solver(void* solver);
int bharatopt_load_model(void* solver, const BharatOptModel* model);
BharatOptSolution bharatopt_solve(void* solver, BharatOptRunTier tier);
void bharatopt_free_solution(BharatOptSolution* sol);
const char* bharatopt_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // BHARATOPT_CORE_H
