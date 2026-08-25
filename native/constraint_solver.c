#include "constraint_solver.h"
<<<<<<< HEAD
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
=======

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

    solver->max_variables   = max_variables;
    solver->max_constraints = max_constraints;
    solver->variable_count  = 0u;
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
>>>>>>> origin/main
    free(solver->variables);
    free(solver->constraints);
    free(solver);
}

NiyahConstraintSolverStatus niyah_constraint_solver_add_variable(
<<<<<<< HEAD
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
=======
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
    var->domain_size    = domain_size;
    var->is_assigned    = false;
    var->assigned_value = 0;

    if (out_var_index) {
        *out_var_index = solver->variable_count;
    }
    ++solver->variable_count;

    return NIYAH_CONSTRAINT_SOLVER_OK;
}

/*
 * Returns true when `new_type` applied to (A, B) directly contradicts
 * `existing_type` already registered for the same or reversed pair.
 *
 * Only catches pairwise contradictions detectable without domain knowledge.
 * Transitivity (A<B, B<C, C<A) is not checked here -- that is left to solve().
 */
static bool directly_contradicts(
    NiyahCSPConstraintType existing_type,
    NiyahCSPConstraintType new_type,
    bool reversed)   /* true when the new constraint has A,B swapped vs existing */
{
    if (!reversed) {
        /* Same ordered pair: exact duplicate is a logic error, not a
         * contradiction per se, but reject it anyway to keep the list clean. */
        return existing_type == new_type;
    }

    /* Reversed pair: check each case that forms a direct contradiction. */
    switch (existing_type) {
        case NIYAH_CSP_CONSTRAINT_LESS_THAN:
            /* existing: A < B; new (reversed): A < B  means B < A => contradiction */
            return new_type == NIYAH_CSP_CONSTRAINT_LESS_THAN    ||
                   new_type == NIYAH_CSP_CONSTRAINT_LESS_EQUAL   ||
                   new_type == NIYAH_CSP_CONSTRAINT_EQUAL;
        case NIYAH_CSP_CONSTRAINT_LESS_EQUAL:
            /* existing: A <= B; new reversed: B <= A  => only contradiction if strict */
            return new_type == NIYAH_CSP_CONSTRAINT_LESS_THAN;
        case NIYAH_CSP_CONSTRAINT_GREATER_THAN:
            return new_type == NIYAH_CSP_CONSTRAINT_GREATER_THAN  ||
                   new_type == NIYAH_CSP_CONSTRAINT_GREATER_EQUAL ||
                   new_type == NIYAH_CSP_CONSTRAINT_EQUAL;
        case NIYAH_CSP_CONSTRAINT_GREATER_EQUAL:
            return new_type == NIYAH_CSP_CONSTRAINT_GREATER_THAN;
        case NIYAH_CSP_CONSTRAINT_EQUAL:
            return new_type == NIYAH_CSP_CONSTRAINT_NOT_EQUAL    ||
                   new_type == NIYAH_CSP_CONSTRAINT_LESS_THAN    ||
                   new_type == NIYAH_CSP_CONSTRAINT_GREATER_THAN;
        case NIYAH_CSP_CONSTRAINT_NOT_EQUAL:
            return new_type == NIYAH_CSP_CONSTRAINT_EQUAL;
        default:
            return false;
    }
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

    /*
     * Bug 2 fix: scan existing constraints for direct contradictions and
     * exact duplicates before inserting.
     *
     * Previously, directional constraints (LESS_THAN etc.) had no duplicate
     * or contradiction check at add time, so adding A<B and then B<A was
     * silently accepted and only discovered (expensively) during solve().
     *
     * Returns NIYAH_CONSTRAINT_SOLVER_CONTRADICTION when a direct conflict
     * is found.  The new constraint is NOT added in that case.
     */
    for (size_t i = 0u; i < solver->constraint_count; ++i) {
        const NiyahCSPConstraint *e = &solver->constraints[i];

        const bool same_pair = (e->var_a_index == var_a_index &&
                                e->var_b_index == var_b_index);
        const bool rev_pair  = (e->var_a_index == var_b_index &&
                                e->var_b_index == var_a_index);

        if (!same_pair && !rev_pair) {
            continue;
        }

        if (directly_contradicts(e->constraint_type, constraint_type, rev_pair)) {
            return NIYAH_CONSTRAINT_SOLVER_CONTRADICTION;
        }
    }

    NiyahCSPConstraint *c = &solver->constraints[solver->constraint_count];
>>>>>>> origin/main
    c->var_a_index    = var_a_index;
    c->var_b_index    = var_b_index;
    c->constraint_type = constraint_type;
    c->is_satisfied   = false;

<<<<<<< HEAD
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
=======
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
>>>>>>> origin/main
    }
    return true;
}

<<<<<<< HEAD
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
=======
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

/*
 * Recursive backtracker.
 *
 * Returns true when the caller should stop searching (solution buffer full).
 *
 * NOTE on early-return semantics: returning true propagates upward through
 * the call stack immediately, abandoning any unexplored domain values for
 * the current variable.  This is intentional: once max_solutions have been
 * collected there is nothing left to do.  The behaviour is NOT a bug;
 * it is a deliberate early-exit optimisation equivalent to a labelled break
 * in an iterative formulation.
 */
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
        /* mark_constraints is called once after all backtracking completes
         * (see niyah_constraint_solver_solve) so we do NOT call it here;
         * doing so mid-search would leave stale flags on a partial state. */
        return *found >= max_solutions;
    }

    NiyahCSPVariable *var = &solver->variables[var_index];

    for (size_t d = 0; d < var->domain_size; ++d) {
        var->assigned_value = var->domain_values[d];
        var->is_assigned    = true;

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

    /*
     * Bug 4 fix: overflow guard for solutions buffer allocation.
     *
     * The caller is expected to provide a buffer of at least
     * (variable_count * max_solutions) ints.  Guard against the multiply
     * overflowing size_t before backtrack writes past the end.
     *
     * max_solutions == 0 is already rejected above, so the divide is safe.
     */
    if (solver->variable_count > SIZE_MAX / max_solutions) {
        return NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS;
    }

    /* Reset all variable assignments before starting a fresh search. */
    for (size_t i = 0; i < solver->variable_count; ++i) {
        solver->variables[i].is_assigned    = false;
        solver->variables[i].assigned_value = 0;
    }

    size_t found = 0u;
    (void)backtrack(solver, 0u, solutions, max_solutions, &found);

    solver->solutions_found = found;
    if (out_solution_count) {
        *out_solution_count = found;
    }

    /*
     * Bug 3 fix: post-solve state restoration.
     *
     * backtrack() unwinds is_assigned=false on every variable as it returns,
     * so after solve() the solver's variables are all unassigned.  Any caller
     * inspecting constraint.is_satisfied after solve() would find all flags
     * false (mark_constraints sees unassigned variables and short-circuits).
     *
     * Fix: if at least one solution was found, replay solutions[0] (the first
     * found solution) back into the variable state and re-run mark_constraints.
     * This makes is_satisfied reflect a real, valid assignment the caller can
     * inspect without needing to re-run solve().
     *
     * If no solution was found, variables are explicitly left unassigned and
     * all is_satisfied flags remain false, which is the correct observable
     * state for an unsatisfiable problem.
     */
    if (found > 0u) {
        const int *first = solutions;   /* solutions[0..variable_count-1] */
        for (size_t i = 0; i < solver->variable_count; ++i) {
            solver->variables[i].assigned_value = first[i];
            solver->variables[i].is_assigned    = true;
        }
        mark_constraints(solver);
    } else {
        for (size_t i = 0; i < solver->variable_count; ++i) {
            solver->variables[i].is_assigned    = false;
            solver->variables[i].assigned_value = 0;
        }
        for (size_t i = 0; i < solver->constraint_count; ++i) {
            solver->constraints[i].is_satisfied = false;
        }
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
        solver->variables[i].is_assigned    = false;
        solver->variables[i].assigned_value = 0;
    }
    for (size_t i = 0; i < solver->constraint_count; ++i) {
        solver->constraints[i].is_satisfied = false;
    }
    solver->solutions_found = 0u;

>>>>>>> origin/main
    return NIYAH_CONSTRAINT_SOLVER_OK;
}
