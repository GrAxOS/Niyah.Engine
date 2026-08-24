#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "constraint_solver.h"

static bool all_distinct(const int* row, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1u; j < n; ++j) {
            if (row[i] == row[j]) {
                return false;
            }
        }
    }
    return true;
}

static void test_all_different(void)
{
    NiyahConstraintSolver* solver = NULL;
    assert(niyah_constraint_solver_create(&solver, 4, 8)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(solver != NULL);

    const int domain[3] = {1, 2, 3};
    size_t a = 99, b = 99, c = 99;
    assert(niyah_constraint_solver_add_variable(solver, "a", domain, 3, &a)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(niyah_constraint_solver_add_variable(solver, "b", domain, 3, &b)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(niyah_constraint_solver_add_variable(solver, "c", domain, 3, &c)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(a == 0 && b == 1 && c == 2);
    assert(solver->variable_count == 3);

    /* All-different over 3 variables with 3 values: exactly 6 permutations. */
    assert(niyah_constraint_solver_add_constraint(
               solver, a, b, NIYAH_CSP_CONSTRAINT_NOT_EQUAL, NULL)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(niyah_constraint_solver_add_constraint(
               solver, b, c, NIYAH_CSP_CONSTRAINT_NOT_EQUAL, NULL)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(niyah_constraint_solver_add_constraint(
               solver, a, c, NIYAH_CSP_CONSTRAINT_NOT_EQUAL, NULL)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(solver->constraint_count == 3);

    int solutions[16 * 3];
    size_t found = 0;
    assert(niyah_constraint_solver_solve(solver, solutions, 16, &found)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(found == 6);
    assert(solver->solutions_found == 6);

    /* Every solution must be a genuine permutation of {1,2,3}. */
    for (size_t s = 0; s < found; ++s) {
        const int* row = solutions + s * 3u;
        assert(all_distinct(row, 3));
        for (int i = 0; i < 3; ++i) {
            assert(row[i] >= 1 && row[i] <= 3);
        }
    }

    /* Solutions are distinct from one another. */
    for (size_t i = 0; i < found; ++i) {
        for (size_t j = i + 1u; j < found; ++j) {
            assert(memcmp(solutions + i * 3u, solutions + j * 3u,
                          3u * sizeof(int)) != 0);
        }
    }

    /* max_solutions caps the search and respects the caller's buffer. */
    int capped[2 * 3];
    size_t capped_found = 0;
    assert(niyah_constraint_solver_solve(solver, capped, 2, &capped_found)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(capped_found == 2);

    niyah_constraint_solver_destroy(solver);
}

static void test_ordering_constraint(void)
{
    NiyahConstraintSolver* solver = NULL;
    assert(niyah_constraint_solver_create(&solver, 2, 2)
           == NIYAH_CONSTRAINT_SOLVER_OK);

    const int domain[3] = {1, 2, 3};
    size_t x = 0, y = 0;
    assert(niyah_constraint_solver_add_variable(solver, "x", domain, 3, &x)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(niyah_constraint_solver_add_variable(solver, "y", domain, 3, &y)
           == NIYAH_CONSTRAINT_SOLVER_OK);

    /* x < y over {1,2,3}: (1,2), (1,3), (2,3) -> 3 solutions. */
    assert(niyah_constraint_solver_add_constraint(
               solver, x, y, NIYAH_CSP_CONSTRAINT_LESS_THAN, NULL)
           == NIYAH_CONSTRAINT_SOLVER_OK);

    int solutions[16 * 2];
    size_t found = 0;
    assert(niyah_constraint_solver_solve(solver, solutions, 16, &found)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(found == 3);

    for (size_t s = 0; s < found; ++s) {
        assert(solutions[s * 2u] < solutions[s * 2u + 1u]);
    }

    /* Every satisfied constraint is flagged after a complete assignment. */
    assert(solver->constraints[0].is_satisfied == true);

    /* reset clears assignments and the satisfaction flags. */
    assert(niyah_constraint_solver_reset(solver)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(solver->solutions_found == 0);
    assert(solver->constraints[0].is_satisfied == false);
    assert(solver->variables[0].is_assigned == false);

    niyah_constraint_solver_destroy(solver);
}

static void test_unsatisfiable(void)
{
    NiyahConstraintSolver* solver = NULL;
    assert(niyah_constraint_solver_create(&solver, 2, 2)
           == NIYAH_CONSTRAINT_SOLVER_OK);

    const int domain[2] = {0, 1};
    size_t p = 0, q = 0;
    assert(niyah_constraint_solver_add_variable(solver, "p", domain, 2, &p)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(niyah_constraint_solver_add_variable(solver, "q", domain, 2, &q)
           == NIYAH_CONSTRAINT_SOLVER_OK);

    /* p == q AND p != q is unsatisfiable. */
    assert(niyah_constraint_solver_add_constraint(
               solver, p, q, NIYAH_CSP_CONSTRAINT_EQUAL, NULL)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(niyah_constraint_solver_add_constraint(
               solver, p, q, NIYAH_CSP_CONSTRAINT_NOT_EQUAL, NULL)
           == NIYAH_CONSTRAINT_SOLVER_OK);

    int solutions[4 * 2];
    size_t found = 99;
    /* Must report NO_SOLUTION rather than returning a bogus assignment. */
    assert(niyah_constraint_solver_solve(solver, solutions, 4, &found)
           == NIYAH_CONSTRAINT_SOLVER_NO_SOLUTION);
    assert(found == 0);

    niyah_constraint_solver_destroy(solver);
}

static void test_validation(void)
{
    NiyahConstraintSolver* solver = NULL;
    assert(niyah_constraint_solver_create(&solver, 2, 2)
           == NIYAH_CONSTRAINT_SOLVER_OK);

    const int domain[2] = {1, 2};

    /* An empty domain is meaningless. */
    assert(niyah_constraint_solver_add_variable(solver, "bad", domain, 0, NULL)
           == NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS);

    /* A domain larger than the fixed array must be refused, not overflowed. */
    int big[NIYAH_CSP_MAX_DOMAIN_SIZE + 8];
    for (size_t i = 0; i < sizeof(big) / sizeof(big[0]); ++i) {
        big[i] = (int)i;
    }
    assert(niyah_constraint_solver_add_variable(
               solver, "big", big, sizeof(big) / sizeof(big[0]), NULL)
           == NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS);

    /* An over-long name is refused rather than truncated. */
    char long_name[128];
    memset(long_name, 'v', sizeof(long_name) - 1u);
    long_name[sizeof(long_name) - 1u] = '\0';
    assert(niyah_constraint_solver_add_variable(solver, long_name, domain, 2, NULL)
           == NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS);

    assert(niyah_constraint_solver_add_variable(solver, "ok", domain, 2, NULL)
           == NIYAH_CONSTRAINT_SOLVER_OK);

    /* Constraints referencing unknown variables. */
    assert(niyah_constraint_solver_add_constraint(
               solver, 0, 99, NIYAH_CSP_CONSTRAINT_EQUAL, NULL)
           == NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS);

    /* A variable constrained against itself is a modelling error. */
    assert(niyah_constraint_solver_add_constraint(
               solver, 0, 0, NIYAH_CSP_CONSTRAINT_NOT_EQUAL, NULL)
           == NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS);

    /* Capacity limits. */
    assert(niyah_constraint_solver_add_variable(solver, "two", domain, 2, NULL)
           == NIYAH_CONSTRAINT_SOLVER_OK);
    assert(niyah_constraint_solver_add_variable(solver, "three", domain, 2, NULL)
           == NIYAH_CONSTRAINT_SOLVER_ERROR);

    /* solve needs somewhere to put the answers. */
    int solutions[4];
    size_t found = 0;
    assert(niyah_constraint_solver_solve(solver, NULL, 4, &found)
           == NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS);
    assert(niyah_constraint_solver_solve(solver, solutions, 0, &found)
           == NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS);

    /* NULL handles. */
    assert(niyah_constraint_solver_create(NULL, 2, 2)
           == NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS);
    assert(niyah_constraint_solver_reset(NULL)
           == NIYAH_CONSTRAINT_SOLVER_INVALID_ARGS);
    niyah_constraint_solver_destroy(NULL);

    niyah_constraint_solver_destroy(solver);
}

int main(void)
{
    test_all_different();
    test_ordering_constraint();
    test_unsatisfiable();
    test_validation();
    return 0;
}
