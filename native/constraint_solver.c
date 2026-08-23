#include "constraint_solver.h"
#include "niyah_core.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ============================================================================
 * Constraint solver lifecycle
 * ============================================================================ */

NiyahConstraintSolverStatus niyah_constraint_solver_create(
    NiyahConstraintSolver **out,
    size_t max_variables,
    size_t max_constraints)
{
    if (!out || max_variables == 0 || max_constraints == 0) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    NiyahConstraintSolver *solver = calloc(1, sizeof(NiyahConstraintSolver));
    if (!solver) {
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    solver->variables = calloc(max_variables, sizeof(NiyahCSPVariable));
    if (!solver->variables) {
        free(solver);
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    solver->constraints = calloc(max_constraints, sizeof(NiyahCSPConstraint));
    if (!solver->constraints) {
        free(solver->variables);
        free(solver);
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    solver->max_variables = max_variables;
    solver->max_constraints = max_constraints;
    solver->variable_count = 0;
    solver->constraint_count = 0;
    solver->solutions_found = 0;

    *out = solver;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}

void niyah_constraint_solver_destroy(NiyahConstraintSolver *solver) {
    if (!solver) {
        return;
    }
    free(solver->constraints);
    free(solver->variables);
    free(solver);
}

/* ============================================================================
 * Variable management
 * ============================================================================ */

NiyahConstraintSolverStatus niyah_constraint_solver_add_variable(
    NiyahConstraintSolver *solver,
    const char *name,
    const int *domain_values,
    size_t domain_size,
    size_t *out_var_index)
{
    if (!solver || !name || !domain_values || !out_var_index) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    if (solver->variable_count >= solver->max_variables) {
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    /* Check for duplicate name */
    for (size_t i = 0; i < solver->variable_count; ++i) {
        if (strcmp(solver->variables[i].name, name) == 0) {
            return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
        }
    }

    NiyahCSPVariable *var = &solver->variables[solver->variable_count];
    memset(var, 0, sizeof(NiyahCSPVariable));

    size_t name_len = strlen(name);
    if (name_len >= sizeof(var->name)) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    memcpy(var->name, name, name_len + 1);

    /* Copy domain values */
    if (domain_size > NIYAH_CSP_MAX_DOMAIN_SIZE) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    memcpy(var->domain_values, domain_values, domain_size * sizeof(int));
    var->domain_size = domain_size;
    var->assigned_value = 0;
    var->is_assigned = false;

    *out_var_index = solver->variable_count;
    solver->variable_count++;

    return NIYAH_CONSTRAINT_SOLVER_OK;
}

/* ============================================================================
 * Constraint management
 * ============================================================================ */

NiyahConstraintSolverStatus niyah_constraint_solver_add_constraint(
    NiyahConstraintSolver *solver,
    size_t var_a_index,
    size_t var_b_index,
    NiyahCSPConstraintType constraint_type,
    size_t *out_constraint_index)
{
    if (!solver || !out_constraint_index) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    if (var_a_index >= solver->variable_count ||
        var_b_index >= solver->variable_count) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    if (solver->constraint_count >= solver->max_constraints) {
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    NiyahCSPConstraint *constraint = &solver->constraints[solver->constraint_count];
    constraint->var_a_index = var_a_index;
    constraint->var_b_index = var_b_index;
    constraint->constraint_type = constraint_type;
    constraint->is_satisfied = false;

    *out_constraint_index = solver->constraint_count;
    solver->constraint_count++;

    return NIYAH_CONSTRAINT_SOLVER_OK;
}

/* ============================================================================
 * Constraint checking
 * ============================================================================ */

static bool check_constraint(
    const NiyahCSPConstraint *constraint,
    const NiyahCSPVariable *variables)
{
    int val_a = variables[constraint->var_a_index].assigned_value;
    int val_b = variables[constraint->var_b_index].assigned_value;

    switch (constraint->constraint_type) {
        case NIYAH_CSP_CONSTRAINT_NOT_EQUAL:
            return (val_a != val_b);
        case NIYAH_CSP_CONSTRAINT_EQUAL:
            return (val_a == val_b);
        case NIYAH_CSP_CONSTRAINT_LESS_THAN:
            return (val_a < val_b);
        case NIYAH_CSP_CONSTRAINT_LESS_EQUAL:
            return (val_a <= val_b);
        case NIYAH_CSP_CONSTRAINT_GREATER_THAN:
            return (val_a > val_b);
        case NIYAH_CSP_CONSTRAINT_GREATER_EQUAL:
            return (val_a >= val_b);
        default:
            return false;
    }
}

/* ============================================================================
 * Backtracking solver
 * ============================================================================ */

static bool solve_recursive(
    NiyahConstraintSolver *solver,
    size_t var_index,
    size_t *solutions,
    size_t max_solutions)
{
    if (solver->solutions_found >= max_solutions) {
        return false; /* Stop searching */
    }

    /* Base case: all variables assigned */
    if (var_index >= solver->variable_count) {
        /* Record solution */
        size_t sol_index = solver->solutions_found;
        for (size_t i = 0; i < solver->variable_count; ++i) {
            solutions[sol_index * solver->variable_count + i] =
                solver->variables[i].assigned_value;
        }
        solver->solutions_found++;
        return true;
    }

    NiyahCSPVariable *var = &solver->variables[var_index];

    /* Try each value in domain */
    for (size_t i = 0; i < var->domain_size; ++i) {
        var->assigned_value = var->domain_values[i];
        var->is_assigned = true;

        /* Check all constraints involving this variable */
        bool consistent = true;
        for (size_t c = 0; c < solver->constraint_count; ++c) {
            NiyahCSPConstraint *constraint = &solver->constraints[c];

            /* Only check if both variables are assigned */
            if (solver->variables[constraint->var_a_index].is_assigned &&
                solver->variables[constraint->var_b_index].is_assigned) {
                
                if (!check_constraint(constraint, solver->variables)) {
                    consistent = false;
                    break;
                }
            }
        }

        if (consistent) {
            /* Recurse to next variable */
            if (!solve_recursive(solver, var_index + 1, solutions, max_solutions)) {
                var->is_assigned = false;
                return false; /* Stop searching */
            }
        }

        var->is_assigned = false;
    }

    return true;
}

NiyahConstraintSolverStatus niyah_constraint_solver_solve(
    NiyahConstraintSolver *solver,
    int *solutions,
    size_t max_solutions,
    size_t *out_solution_count)
{
    if (!solver || !solutions || !out_solution_count) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    /* Check that all variables have non-empty domains */
    for (size_t i = 0; i < solver->variable_count; ++i) {
        if (solver->variables[i].domain_size == 0) {
            *out_solution_count = 0;
            return NIYAH_CONSTRAINT_SOLVER_OK; /* No solution */
        }
    }

    solver->solutions_found = 0;
    solve_recursive(solver, 0, solutions, max_solutions);

    *out_solution_count = solver->solutions_found;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}

/* ============================================================================
 * Reset solver for new search
 * ============================================================================ */

NiyahConstraintSolverStatus niyah_constraint_solver_reset(
    NiyahConstraintSolver *solver)
{
    if (!solver) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    for (size_t i = 0; i < solver->variable_count; ++i) {
        solver->variables[i].assigned_value = 0;
        solver->variables[i].is_assigned = false;
    }

    solver->solutions_found = 0;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}
