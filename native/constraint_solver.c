#include "constraint_solver.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

NiyahConstraintSolverStatus niyah_constraint_solver_create(
        NiyahConstraintSolver** out,
        size_t max_variables,
        size_t max_constraints) {
    if (!out || max_variables == 0) return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;

    *out = (NiyahConstraintSolver*)calloc(1, sizeof(NiyahConstraintSolver));
    if (!*out) return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;

    (*out)->variables   = (NiyahCSPVariable*)calloc(max_variables, sizeof(NiyahCSPVariable));
    (*out)->constraints = (NiyahCSPConstraint*)calloc(max_constraints, sizeof(NiyahCSPConstraint));
    if (!(*out)->variables || !(*out)->constraints) {
        free((*out)->variables); free((*out)->constraints); free(*out);
        return NIYAH_CONSTRAINT_SOLVER_OUT_OF_MEMORY;
    }

    (*out)->max_variables   = max_variables;
    (*out)->max_constraints = max_constraints;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}

void niyah_constraint_solver_destroy(NiyahConstraintSolver* solver) {
    if (!solver) return;
    free(solver->variables);
    free(solver->constraints);
    free(solver);
}

NiyahConstraintSolverStatus niyah_constraint_solver_add_variable(
        NiyahConstraintSolver* solver,
        const char* name,
        const int*  domain_values,
        size_t      domain_size,
        size_t*     out_var_index) {
    if (!solver || !name || !domain_values || domain_size == 0) return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    if (solver->variable_count >= solver->max_variables) return NIYAH_CONSTRAINT_SOLVER_ERROR;
    if (domain_size > NIYAH_CSP_MAX_DOMAIN_SIZE) domain_size = NIYAH_CSP_MAX_DOMAIN_SIZE;

    NiyahCSPVariable* v = &solver->variables[solver->variable_count];
    strncpy(v->name, name, sizeof(v->name) - 1);
    memcpy(v->domain_values, domain_values, domain_size * sizeof(int));
    v->domain_size    = domain_size;
    v->is_assigned    = false;
    v->assigned_value = 0;

    if (out_var_index) *out_var_index = solver->variable_count;
    solver->variable_count++;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}

NiyahConstraintSolverStatus niyah_constraint_solver_add_constraint(
        NiyahConstraintSolver*   solver,
        size_t var_a_index,
        size_t var_b_index,
        NiyahCSPConstraintType   constraint_type,
        size_t*                  out_constraint_index) {
    if (!solver) return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    if (var_a_index >= solver->variable_count || var_b_index >= solver->variable_count)
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    if (solver->constraint_count >= solver->max_constraints) return NIYAH_CONSTRAINT_SOLVER_ERROR;

    NiyahCSPConstraint* c = &solver->constraints[solver->constraint_count];
    c->var_a_index    = var_a_index;
    c->var_b_index    = var_b_index;
    c->constraint_type = constraint_type;
    c->is_satisfied   = false;

    if (out_constraint_index) *out_constraint_index = solver->constraint_count;
    solver->constraint_count++;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}

/* ── Constraint check ──────────────────────────────────────────────────── */
static bool check_constraint(const NiyahCSPConstraint* c, int va, int vb) {
    switch (c->constraint_type) {
        case NIYAH_CSP_CONSTRAINT_NOT_EQUAL:    return va != vb;
        case NIYAH_CSP_CONSTRAINT_EQUAL:        return va == vb;
        case NIYAH_CSP_CONSTRAINT_LESS_THAN:    return va <  vb;
        case NIYAH_CSP_CONSTRAINT_LESS_EQUAL:   return va <= vb;
        case NIYAH_CSP_CONSTRAINT_GREATER_THAN: return va >  vb;
        case NIYAH_CSP_CONSTRAINT_GREATER_EQUAL:return va >= vb;
        default: return true;
    }
}

static bool is_consistent(const NiyahConstraintSolver* s, size_t var_idx, int value) {
    for (size_t c = 0; c < s->constraint_count; c++) {
        const NiyahCSPConstraint* con = &s->constraints[c];
        size_t other = SIZE_MAX;
        int    oval  = 0;
        bool   flip  = false;

        if (con->var_a_index == var_idx && s->variables[con->var_b_index].is_assigned) {
            other = con->var_b_index;
            oval  = s->variables[other].assigned_value;
        } else if (con->var_b_index == var_idx && s->variables[con->var_a_index].is_assigned) {
            other = con->var_a_index;
            oval  = s->variables[other].assigned_value;
            flip  = true;
        }

        if (other == SIZE_MAX) continue;
        bool ok = flip ? check_constraint(con, oval, value)
                       : check_constraint(con, value, oval);
        if (!ok) return false;
    }
    return true;
}

/* ── Backtracking search ───────────────────────────────────────────────── */
static bool backtrack(NiyahConstraintSolver* s,
                       size_t var_idx,
                       int*   solutions,
                       size_t max_solutions,
                       size_t* solution_count) {

    if (var_idx == s->variable_count) {
        if (*solution_count < max_solutions) {
            for (size_t i = 0; i < s->variable_count; i++)
                solutions[(*solution_count) * s->variable_count + i] =
                    s->variables[i].assigned_value;
            (*solution_count)++;
        }
        return *solution_count > 0;
    }

    NiyahCSPVariable* v = &s->variables[var_idx];
    for (size_t d = 0; d < v->domain_size; d++) {
        int val = v->domain_values[d];
        if (!is_consistent(s, var_idx, val)) continue;

        v->assigned_value = val;
        v->is_assigned    = true;

        bool found = backtrack(s, var_idx + 1, solutions, max_solutions, solution_count);
        if (found && *solution_count >= max_solutions) return true;

        v->is_assigned = false;
    }
    return *solution_count > 0;
}

NiyahConstraintSolverStatus niyah_constraint_solver_solve(
        NiyahConstraintSolver* solver,
        int*    solutions,
        size_t  max_solutions,
        size_t* out_solution_count) {
    if (!solver || !solutions || !out_solution_count) return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;

    *out_solution_count = 0;
    for (size_t i = 0; i < solver->variable_count; i++)
        solver->variables[i].is_assigned = false;

    bool found = backtrack(solver, 0, solutions, max_solutions, out_solution_count);
    solver->solutions_found = *out_solution_count;

    if (!found && *out_solution_count == 0) return NIYAH_CONSTRAINT_SOLVER_NO_SOLUTION;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}

NiyahConstraintSolverStatus niyah_constraint_solver_reset(NiyahConstraintSolver* solver) {
    if (!solver) return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    for (size_t i = 0; i < solver->variable_count; i++)
        solver->variables[i].is_assigned = false;
    solver->solutions_found = 0;
    return NIYAH_CONSTRAINT_SOLVER_OK;
}
