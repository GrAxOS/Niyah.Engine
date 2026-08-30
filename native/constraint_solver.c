#include "constraint_solver.h"

#include <stdlib.h>
#include <string.h>

/*
 * Was: `// Constraint solver stubs`.
 *
 * Chronological backtracking with incremental consistency checking: a value is
 * rejected as soon as it violates a constraint against an already-assigned
 * variable, so whole subtrees are pruned instead of enumerated.
 */

NiyahConstraintSolverStatus niyah_constraint_solver_create(
    NiyahConstraintSolver **out,
    size_t max_variables,
    size_t max_constraints)
{
    if (!out || max_variables == 0u) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    *out = NULL;

    NiyahConstraintSolver *solver =
        (NiyahConstraintSolver *)calloc(1, sizeof(NiyahConstraintSolver));
    if (!solver) {
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    solver->variables = (NiyahCSPVariable *)calloc(max_variables,
                                                   sizeof(NiyahCSPVariable));
    if (!solver->variables) {
        free(solver);
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    /* A CSP with zero constraints is legal, so tolerate max_constraints == 0. */
    if (max_constraints > 0u) {
        solver->constraints = (NiyahCSPConstraint *)calloc(
            max_constraints, sizeof(NiyahCSPConstraint));
        if (!solver->constraints) {
            free(solver->variables);
            free(solver);
            return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
        }
    }

    solver->max_variables = max_variables;
    solver->max_constraints = max_constraints;
    solver->variable_count = 0u;
    solver->constraint_count = 0u;
    solver->solutions_found = 0u;

    *out = solver;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}

void niyah_constraint_solver_destroy(NiyahConstraintSolver *solver)
{
    if (!solver) {
        return;
    }
    free(solver->variables);
    free(solver->constraints);
    free(solver);
}

NiyahConstraintSolverStatus niyah_constraint_solver_add_variable(
    NiyahConstraintSolver *solver,
    const char *name,
    const int *domain_values,
    size_t domain_size,
    size_t *out_var_index)
{
    if (!solver || !name || !domain_values) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    if (domain_size == 0u || domain_size > NIYAH_CSP_MAX_DOMAIN_SIZE) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    if (strlen(name) >= sizeof(solver->variables[0].name)) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    if (solver->variable_count >= solver->max_variables) {
        return NIYAH_CONSTRAINT_SOLVER_ERROR;
    }

    NiyahCSPVariable *var = &solver->variables[solver->variable_count];
    memset(var, 0, sizeof(*var));
    memcpy(var->name, name, strlen(name));
    memcpy(var->domain_values, domain_values, domain_size * sizeof(int));
    var->domain_size = domain_size;
    var->is_assigned = false;
    var->assigned_value = 0;

    if (out_var_index) {
        *out_var_index = solver->variable_count;
    }
    ++solver->variable_count;

    return NIYAH_CONSTRAINT_SOLVER_OK;
}

NiyahConstraintSolverStatus niyah_constraint_solver_add_constraint(
    NiyahConstraintSolver *solver,
    size_t var_a_index,
    size_t var_b_index,
    NiyahCSPConstraintType constraint_type,
    size_t *out_constraint_index)
{
    if (!solver || !solver->constraints) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    if (var_a_index >= solver->variable_count ||
        var_b_index >= solver->variable_count) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    if (var_a_index == var_b_index) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    if (constraint_type < NIYAH_CSP_CONSTRAINT_NOT_EQUAL ||
        constraint_type > NIYAH_CSP_CONSTRAINT_GREATER_EQUAL) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    if (solver->constraint_count >= solver->max_constraints) {
        return NIYAH_CONSTRAINT_SOLVER_ERROR;
    }

    NiyahCSPConstraint *c = &solver->constraints[solver->constraint_count];
    c->var_a_index = var_a_index;
    c->var_b_index = var_b_index;
    c->constraint_type = constraint_type;
    c->is_satisfied = false;

    if (out_constraint_index) {
        *out_constraint_index = solver->constraint_count;
    }
    ++solver->constraint_count;

    return NIYAH_CONSTRAINT_SOLVER_OK;
}

static bool holds(NiyahCSPConstraintType type, int a, int b)
{
    switch (type) {
        case NIYAH_CSP_CONSTRAINT_NOT_EQUAL:     return a != b;
        case NIYAH_CSP_CONSTRAINT_EQUAL:         return a == b;
        case NIYAH_CSP_CONSTRAINT_LESS_THAN:     return a < b;
        case NIYAH_CSP_CONSTRAINT_LESS_EQUAL:    return a <= b;
        case NIYAH_CSP_CONSTRAINT_GREATER_THAN:  return a > b;
        case NIYAH_CSP_CONSTRAINT_GREATER_EQUAL: return a >= b;
        default:                                 return false;
    }
}

/* Is the just-assigned variable consistent with everything decided so far? */
static bool consistent(const NiyahConstraintSolver *solver, size_t var_index)
{
    for (size_t i = 0; i < solver->constraint_count; ++i) {
        const NiyahCSPConstraint *c = &solver->constraints[i];
        if (c->var_a_index != var_index && c->var_b_index != var_index) {
            continue;
        }

        const NiyahCSPVariable *a = &solver->variables[c->var_a_index];
        const NiyahCSPVariable *b = &solver->variables[c->var_b_index];
        if (!a->is_assigned || !b->is_assigned) {
            continue;   /* not yet decidable */
        }
        if (!holds(c->constraint_type, a->assigned_value, b->assigned_value)) {
            return false;
        }
    }
    return true;
}

static void mark_constraints(NiyahConstraintSolver *solver)
{
    for (size_t i = 0; i < solver->constraint_count; ++i) {
        NiyahCSPConstraint *c = &solver->constraints[i];
        const NiyahCSPVariable *a = &solver->variables[c->var_a_index];
        const NiyahCSPVariable *b = &solver->variables[c->var_b_index];
        c->is_satisfied = a->is_assigned && b->is_assigned &&
            holds(c->constraint_type, a->assigned_value, b->assigned_value);
    }
}

/* Returns true when the caller should stop searching (buffer is full). */
static bool backtrack(NiyahConstraintSolver *solver,
                      size_t var_index,
                      int *solutions,
                      size_t max_solutions,
                      size_t *found)
{
    if (var_index == solver->variable_count) {
        int *row = solutions + (*found) * solver->variable_count;
        for (size_t i = 0; i < solver->variable_count; ++i) {
            row[i] = solver->variables[i].assigned_value;
        }
        ++(*found);
        mark_constraints(solver);
        return *found >= max_solutions;
    }

    NiyahCSPVariable *var = &solver->variables[var_index];

    for (size_t d = 0; d < var->domain_size; ++d) {
        var->assigned_value = var->domain_values[d];
        var->is_assigned = true;

        if (consistent(solver, var_index)) {
            if (backtrack(solver, var_index + 1u, solutions,
                          max_solutions, found)) {
                var->is_assigned = false;
                return true;
            }
        }

        var->is_assigned = false;
    }

    return false;
}

NiyahConstraintSolverStatus niyah_constraint_solver_solve(
    NiyahConstraintSolver *solver,
    int *solutions,
    size_t max_solutions,
    size_t *out_solution_count)
{
    if (!solver || !solutions || max_solutions == 0u) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }
    if (solver->variable_count == 0u) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    for (size_t i = 0; i < solver->variable_count; ++i) {
        solver->variables[i].is_assigned = false;
        solver->variables[i].assigned_value = 0;
    }

    size_t found = 0u;
    (void)backtrack(solver, 0u, solutions, max_solutions, &found);

    solver->solutions_found = found;
    if (out_solution_count) {
        *out_solution_count = found;
    }

    return (found == 0u)
        ? NIYAH_CONSTRAINT_SOLVER_NO_SOLUTION
        : NIYAH_CONSTRAINT_SOLVER_OK;
}

NiyahConstraintSolverStatus niyah_constraint_solver_reset(
    NiyahConstraintSolver *solver)
{
    if (!solver) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    for (size_t i = 0; i < solver->variable_count; ++i) {
        solver->variables[i].is_assigned = false;
        solver->variables[i].assigned_value = 0;
    }
    for (size_t i = 0; i < solver->constraint_count; ++i) {
        solver->constraints[i].is_satisfied = false;
    }
    solver->solutions_found = 0u;

    return NIYAH_CONSTRAINT_SOLVER_OK;
}
