#ifndef NIYAH_CONSTRAINT_SOLVER_H
#define NIYAH_CONSTRAINT_SOLVER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef NIYAH_CONSTRAINT_API
#if defined(_WIN32)
#if defined(NIYAH_BRIDGE_EXPORTS)
#define NIYAH_CONSTRAINT_API __declspec(dllexport)
#else
#define NIYAH_CONSTRAINT_API __declspec(dllimport)
#endif
#else
#define NIYAH_CONSTRAINT_API __attribute__((visibility("default")))
#endif
#endif

#define NIYAH_CSP_MAX_DOMAIN_SIZE 64u
#define NIYAH_CSP_VARIABLE_NAME_SIZE 64u

typedef enum NiyahConstraintSolverStatus {
    NIYAH_CONSTRAINT_SOLVER_OK = 0,
    NIYAH_CONSTRAINT_SOLVER_ERROR = 1,
    NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY = 2,
    NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS = 3,
    NIYAH_CONSTRAINT_SOLVER_NO_SOLUTION = 4
} NiyahConstraintSolverStatus;

typedef enum NiyahCSPConstraintType {
    NIYAH_CSP_CONSTRAINT_NOT_EQUAL = 0,
    NIYAH_CSP_CONSTRAINT_EQUAL = 1,
    NIYAH_CSP_CONSTRAINT_LESS_THAN = 2,
    NIYAH_CSP_CONSTRAINT_LESS_EQUAL = 3,
    NIYAH_CSP_CONSTRAINT_GREATER_THAN = 4,
    NIYAH_CSP_CONSTRAINT_GREATER_EQUAL = 5
} NiyahCSPConstraintType;

typedef struct NiyahCSPVariable {
    char name[NIYAH_CSP_VARIABLE_NAME_SIZE];
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

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_create(
    NiyahConstraintSolver **out,
    size_t max_variables,
    size_t max_constraints);

NIYAH_CONSTRAINT_API void niyah_constraint_solver_destroy(
    NiyahConstraintSolver *solver);

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_add_variable(
    NiyahConstraintSolver *solver,
    const char *name,
    const int *domain_values,
    size_t domain_size,
    size_t *out_var_index);

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_add_constraint(
    NiyahConstraintSolver *solver,
    size_t var_a_index,
    size_t var_b_index,
    NiyahCSPConstraintType constraint_type,
    size_t *out_constraint_index);

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_solve(
    NiyahConstraintSolver *solver,
    int *solutions,
    size_t max_solutions,
    size_t *out_solution_count);

NIYAH_CONSTRAINT_API NiyahConstraintSolverStatus niyah_constraint_solver_reset(
    NiyahConstraintSolver *solver);

#endif
