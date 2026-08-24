#ifndef NIYAH_CONSTRAINT_SOLVER_H
#define NIYAH_CONSTRAINT_SOLVER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * Guarded so the macro can also be set on the command line or by another
 * header. Define NIYAH_BRIDGE_EXPORTS when compiling the library itself.
 */
#ifndef NIYAH_CONSTRAINT_API
    #ifdef _WIN32
        #ifdef NIYAH_BRIDGE_EXPORTS
            #define NIYAH_CONSTRAINT_API __declspec(dllexport)
        #else
            #define NIYAH_CONSTRAINT_API __declspec(dllimport)
        #endif
    #else
        #define NIYAH_CONSTRAINT_API __attribute__((visibility("default")))
    #endif
#endif

#define NIYAH_CSP_MAX_DOMAIN_SIZE 64

typedef enum {
    NIYAH_CONSTRAINT_SOLVER_OK              = 0,
    NIYAH_CONSTRAINT_SOLVER_ERROR           = 1,
    NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY   = 2,
    NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS    = 3,
    NIYAH_CONSTRAINT_SOLVER_NO_SOLUTION     = 4,
    /*
     * Returned by niyah_constraint_solver_add_constraint when a newly added
     * constraint directly contradicts an existing one (e.g. A < B followed
     * by B < A). The constraint is NOT added to the solver.
     */
    NIYAH_CONSTRAINT_SOLVER_CONTRADICTION   = 5
} NiyahConstraintSolverStatus;

typedef enum {
    NIYAH_CSP_CONSTRAINT_NOT_EQUAL = 0,
    NIYAH_CSP_CONSTRAINT_EQUAL = 1,
    NIYAH_CSP_CONSTRAINT_LESS_THAN = 2,
    NIYAH_CSP_CONSTRAINT_LESS_EQUAL = 3,
    NIYAH_CSP_CONSTRAINT_GREATER_THAN = 4,
    NIYAH_CSP_CONSTRAINT_GREATER_EQUAL = 5
} NiyahCSPConstraintType;

typedef struct NiyahCSPVariable {
    char name[64];
    int domain_values[NIYAH_CSP_MAX_DOMAIN_SIZE];
    size_t domain_size;
    int assigned_value;
    bool is_assigned;
} NiyahCSPVariable;

typedef struct NiyahCSPConstraint {
    size_t var_a_index;
    size_t var_b_index;
    NiyahCSPConstraintType constraint_type;
    bool is_satisfied;
} NiyahCSPConstraint;

typedef struct NiyahConstraintSolver {
    NiyahCSPVariable *variables;
    NiyahCSPConstraint *constraints;
    size_t max_variables;
    size_t max_constraints;
    size_t variable_count;
    size_t constraint_count;
    size_t solutions_found;
} NiyahConstraintSolver;

/* ============================================================================
 * Solver lifecycle
 * ============================================================================ */

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_create(
    NiyahConstraintSolver **out,
    size_t max_variables,
    size_t max_constraints);

NIYAH_CONSTRAINT_API void niyah_constraint_solver_destroy(
    NiyahConstraintSolver *solver);

/* ============================================================================
 * Variable management
 * ============================================================================ */

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_add_variable(
    NiyahConstraintSolver *solver,
    const char *name,
    const int *domain_values,
    size_t domain_size,
    size_t *out_var_index);

/* ============================================================================
 * Constraint management
 *
 * Returns NIYAH_CONSTRAINT_SOLVER_CONTRADICTION when the new constraint
 * directly contradicts an already-registered one.  The contradiction is
 * detected at add time so that niyah_constraint_solver_solve() never wastes
 * time on an unsatisfiable problem that could have been rejected early.
 * ============================================================================ */

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_add_constraint(
    NiyahConstraintSolver *solver,
    size_t var_a_index,
    size_t var_b_index,
    NiyahCSPConstraintType constraint_type,
    size_t *out_constraint_index);

/* ============================================================================
 * Solving
 *
 * After a successful call (return value OK or NO_SOLUTION), the solver's
 * internal variable state reflects the FIRST solution found, so that
 * constraint is_satisfied flags are meaningful without re-running solve.
 * If no solution was found, all variables are left unassigned.
 * ============================================================================ */

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_solve(
    NiyahConstraintSolver *solver,
    int *solutions,
    size_t max_solutions,
    size_t *out_solution_count);

/* ============================================================================
 * Reset solver for new search
 * ============================================================================ */

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_reset(
    NiyahConstraintSolver *solver);

#endif /* NIYAH_CONSTRAINT_SOLVER_H */
