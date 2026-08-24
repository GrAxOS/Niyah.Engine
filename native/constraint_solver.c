#include "constraint_solver.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static bool niyah_csp_holds(NiyahCSPConstraintType type, int a, int b)
{
    switch (type) {
        case NIYAH_CSP_CONSTRAINT_NOT_EQUAL:
            return a != b;
        case NIYAH_CSP_CONSTRAINT_EQUAL:
            return a == b;
        case NIYAH_CSP_CONSTRAINT_LESS_THAN:
            return a < b;
        case NIYAH_CSP_CONSTRAINT_LESS_EQUAL:
            return a <= b;
        case NIYAH_CSP_CONSTRAINT_GREATER_THAN:
            return a > b;
        case NIYAH_CSP_CONSTRAINT_GREATER_EQUAL:
            return a >= b;
        default:
            return false;
    }
}

static bool niyah_csp_is_valid_constraint_type(NiyahCSPConstraintType type)
{
    return type >= NIYAH_CSP_CONSTRAINT_NOT_EQUAL &&
           type <= NIYAH_CSP_CONSTRAINT_GREATER_EQUAL;
}

static bool niyah_csp_constraint_is_consistent(
    const NiyahConstraintSolver *solver,
    size_t var_index)
{
    if (solver == NULL || var_index >= solver->variable_count) {
        return false;
    }

    for (size_t i = 0u; i < solver->constraint_count; ++i) {
        const NiyahCSPConstraint *constraint = &solver->constraints[i];
        if (constraint->var_a_index != var_index &&
            constraint->var_b_index != var_index) {
            continue;
        }

        const NiyahCSPVariable *a =
            &solver->variables[constraint->var_a_index];
        const NiyahCSPVariable *b =
            &solver->variables[constraint->var_b_index];

        if (!a->is_assigned || !b->is_assigned) {
            continue;
        }

        if (!niyah_csp_holds(constraint->constraint_type,
                             a->assigned_value,
                             b->assigned_value)) {
            return false;
        }
    }

    return true;
}

static void niyah_csp_update_constraint_states(NiyahConstraintSolver *solver)
{
    if (solver == NULL) {
        return;
    }

    for (size_t i = 0u; i < solver->constraint_count; ++i) {
        NiyahCSPConstraint *constraint = &solver->constraints[i];
        const NiyahCSPVariable *a =
            &solver->variables[constraint->var_a_index];
        const NiyahCSPVariable *b =
            &solver->variables[constraint->var_b_index];

        constraint->is_satisfied =
            a->is_assigned &&
            b->is_assigned &&
            niyah_csp_holds(constraint->constraint_type,
                            a->assigned_value,
                            b->assigned_value);
    }
}

static bool niyah_csp_search(
    NiyahConstraintSolver *solver,
    size_t variable_index,
    int *solutions,
    size_t max_solutions,
    size_t *found)
{
    if (solver == NULL || solutions == NULL || found == NULL ||
        max_solutions == 0u || *found >= max_solutions) {
        return true;
    }

    if (variable_index == solver->variable_count) {
        if (solver->variable_count > SIZE_MAX / sizeof(int)) {
            return true;
        }

        if (*found > SIZE_MAX / solver->variable_count) {
            return true;
        }

        const size_t offset = *found * solver->variable_count;
        for (size_t i = 0u; i < solver->variable_count; ++i) {
            solutions[offset + i] = solver->variables[i].assigned_value;
        }

        ++(*found);
        niyah_csp_update_constraint_states(solver);
        return *found >= max_solutions;
    }

    NiyahCSPVariable *variable = &solver->variables[variable_index];

    for (size_t d = 0u; d < variable->domain_size; ++d) {
        variable->assigned_value = variable->domain_values[d];
        variable->is_assigned = true;

        if (niyah_csp_constraint_is_consistent(solver, variable_index) &&
            niyah_csp_search(solver,
                             variable_index + 1u,
                             solutions,
                             max_solutions,
                             found)) {
            variable->is_assigned = false;
            return true;
        }

        variable->is_assigned = false;
    }

    variable->assigned_value = 0;
    return false;
}

NiyahConstraintSolverStatus niyah_constraint_solver_create(
    NiyahConstraintSolver **out,
    size_t max_variables,
    size_t max_constraints)
{
    if (out == NULL || max_variables == 0u) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    *out = NULL;

    if (max_variables > SIZE_MAX / sizeof(NiyahCSPVariable) ||
        max_constraints > SIZE_MAX / sizeof(NiyahCSPConstraint)) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    NiyahConstraintSolver *solver =
        (NiyahConstraintSolver *)calloc(1u, sizeof(*solver));
    if (solver == NULL) {
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    solver->variables = (NiyahCSPVariable *)calloc(
        max_variables, sizeof(*solver->variables));
    if (solver->variables == NULL) {
        free(solver);
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    if (max_constraints > 0u) {
        solver->constraints = (NiyahCSPConstraint *)calloc(
            max_constraints, sizeof(*solver->constraints));
        if (solver->constraints == NULL) {
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
    if (solver == NULL) {
        return;
    }

    free(solver->constraints);
    free(solver->variables);
    free(solver);
}

NiyahConstraintSolverStatus niyah_constraint_solver_add_variable(
    NiyahConstraintSolver *solver,
    const char *name,
    const int *domain_values,
    size_t domain_size,
    size_t *out_var_index)
{
    if (solver == NULL || name == NULL || domain_values == NULL) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    if (solver->variables == NULL ||
        solver->variable_count >= solver->max_variables ||
        domain_size == 0u ||
        domain_size > NIYAH_CSP_MAX_DOMAIN_SIZE) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    const size_t name_length = strnlen(name, NIYAH_CSP_VARIABLE_NAME_SIZE);
    if (name_length == 0u || name_length >= NIYAH_CSP_VARIABLE_NAME_SIZE) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    for (size_t i = 0u; i < solver->variable_count; ++i) {
        if (strncmp(solver->variables[i].name,
                    name,
                    NIYAH_CSP_VARIABLE_NAME_SIZE) == 0) {
            return NIYAH_CONSTRAINT_SOLVER_ERROR;
        }
    }

    NiyahCSPVariable *variable = &solver->variables[solver->variable_count];
    memset(variable, 0, sizeof(*variable));
    memcpy(variable->name, name, name_length);
    variable->name[name_length] = '\0';

    for (size_t i = 0u; i < domain_size; ++i) {
        variable->domain_values[i] = domain_values[i];
    }
    variable->domain_size = domain_size;
    variable->assigned_value = 0;
    variable->is_assigned = false;

    if (out_var_index != NULL) {
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
    if (solver == NULL || solver->constraints == NULL ||
        solver->constraint_count >= solver->max_constraints ||
        var_a_index >= solver->variable_count ||
        var_b_index >= solver->variable_count ||
        var_a_index == var_b_index ||
        !niyah_csp_is_valid_constraint_type(constraint_type)) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    for (size_t i = 0u; i < solver->constraint_count; ++i) {
        const NiyahCSPConstraint *existing = &solver->constraints[i];
        if (existing->var_a_index == var_a_index &&
            existing->var_b_index == var_b_index &&
            existing->constraint_type == constraint_type) {
            return NIYAH_CONSTRAINT_SOLVER_ERROR;
        }
        if (existing->var_a_index == var_b_index &&
            existing->var_b_index == var_a_index &&
            existing->constraint_type == constraint_type) {
            const bool symmetric =
                constraint_type == NIYAH_CSP_CONSTRAINT_NOT_EQUAL ||
                constraint_type == NIYAH_CSP_CONSTRAINT_EQUAL;
            if (symmetric) {
                return NIYAH_CONSTRAINT_SOLVER_ERROR;
            }
        }
    }

    NiyahCSPConstraint *constraint =
        &solver->constraints[solver->constraint_count];
    constraint->var_a_index = var_a_index;
    constraint->var_b_index = var_b_index;
    constraint->constraint_type = constraint_type;
    constraint->is_satisfied = false;

    if (out_constraint_index != NULL) {
        *out_constraint_index = solver->constraint_count;
    }
    ++solver->constraint_count;

    return NIYAH_CONSTRAINT_SOLVER_OK;
}

NiyahConstraintSolverStatus niyah_constraint_solver_solve(
    NiyahConstraintSolver *solver,
    int *solutions,
    size_t max_solutions,
    size_t *out_solution_count)
{
    if (solver == NULL || solutions == NULL || max_solutions == 0u ||
        solver->variable_count == 0u) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    if (solver->variable_count > SIZE_MAX / max_solutions) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    if (solver->variable_count * max_solutions > SIZE_MAX / sizeof(int)) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    if (out_solution_count != NULL) {
        *out_solution_count = 0u;
    }

    for (size_t i = 0u; i < solver->variable_count; ++i) {
        solver->variables[i].assigned_value = 0;
        solver->variables[i].is_assigned = false;
    }

    for (size_t i = 0u; i < solver->constraint_count; ++i) {
        solver->constraints[i].is_satisfied = false;
    }

    size_t found = 0u;
    (void)niyah_csp_search(solver, 0u, solutions, max_solutions, &found);
    solver->solutions_found = found;

    if (out_solution_count != NULL) {
        *out_solution_count = found;
    }

    for (size_t i = 0u; i < solver->variable_count; ++i) {
        solver->variables[i].assigned_value = 0;
        solver->variables[i].is_assigned = false;
    }

    if (found == 0u) {
        niyah_csp_update_constraint_states(solver);
        return NIYAH_CONSTRAINT_SOLVER_NO_SOLUTION;
    }

    return NIYAH_CONSTRAINT_SOLVER_OK;
}

NiyahConstraintSolverStatus niyah_constraint_solver_reset(
    NiyahConstraintSolver *solver)
{
    if (solver == NULL) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    for (size_t i = 0u; i < solver->variable_count; ++i) {
        solver->variables[i].assigned_value = 0;
        solver->variables[i].is_assigned = false;
    }

    for (size_t i = 0u; i < solver->constraint_count; ++i) {
        solver->constraints[i].is_satisfied = false;
    }

    solver->solutions_found = 0u;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}
