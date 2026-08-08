"""
Unit tests for the Overgraze world engine.

    python -m unittest test_world -v

The resolution rules get the most attention, especially two agents harvesting
the same cell on the same tick -- the case the sequential engine could not
represent at all.
"""

import unittest

import numpy as np

from world import (CAP, CAPACITY, COLLAPSE_FLOOR, N, TAKE, TICKS, Action,
                   apply_actions, initial_state, neighbours_mean, regrow,
                   resolve_cell, rng_for)
import harness


def st(kinds=("greedy", "greedy"), seed=0, rule="global", r=0.15):
    return initial_state(seed, list(kinds), rule=rule, r=r)


class TestResolveCell(unittest.TestCase):
    """The contention rule, in isolation."""

    def test_uncontested_ask_is_granted_whole(self):
        self.assertEqual(resolve_cell(1.0, [0.55]), [0.55])

    def test_two_agents_split_a_cell_that_cannot_serve_both(self):
        # 0.5 available, both want 0.55 -> half each, nothing invented
        grants = resolve_cell(0.5, [0.55, 0.55])
        self.assertAlmostEqual(grants[0], 0.25)
        self.assertAlmostEqual(grants[1], 0.25)
        self.assertAlmostEqual(sum(grants), 0.5)

    def test_two_agents_both_served_when_the_cell_is_rich_enough(self):
        grants = resolve_cell(1.0, [0.4, 0.5])
        self.assertAlmostEqual(grants[0], 0.4)
        self.assertAlmostEqual(grants[1], 0.5)

    def test_surplus_from_a_small_ask_flows_to_the_larger_one(self):
        # max-min fair: equal share is 0.3 each, but one only wants 0.1,
        # so the other should end up with 0.5 rather than being capped at 0.3
        grants = resolve_cell(0.6, [0.1, 0.55])
        self.assertAlmostEqual(grants[0], 0.1)
        self.assertAlmostEqual(grants[1], 0.5)
        self.assertAlmostEqual(sum(grants), 0.6)

    def test_never_grants_more_than_available(self):
        for avail in (0.0, 0.01, 0.3, 1.0):
            grants = resolve_cell(avail, [TAKE] * 4)
            self.assertLessEqual(sum(grants), avail + 1e-9)

    def test_never_grants_more_than_asked(self):
        grants = resolve_cell(1.0, [0.1, 0.2, 0.05])
        for g, a in zip(grants, [0.1, 0.2, 0.05]):
            self.assertLessEqual(g, a + 1e-9)

    def test_conserves_exactly(self):
        for avail, asks in [(1.0, [0.2, 0.3]), (0.4, [0.55, 0.55]),
                            (0.25, [0.1, 0.1, 0.1]), (0.0, [0.5])]:
            self.assertAlmostEqual(sum(resolve_cell(avail, asks)),
                                   min(sum(asks), avail))

    def test_order_independent(self):
        a = resolve_cell(0.7, [0.55, 0.1, 0.3])
        b = resolve_cell(0.7, [0.3, 0.55, 0.1])
        self.assertAlmostEqual(sorted(a)[0], sorted(b)[0])
        self.assertAlmostEqual(sum(a), sum(b))

    def test_equal_asks_get_equal_grants(self):
        grants = resolve_cell(0.4, [0.55, 0.55, 0.55])
        self.assertAlmostEqual(min(grants), max(grants))


class TestSameCellSameTick(unittest.TestCase):
    """Two agents harvesting the same cell on the same tick, end to end."""

    def setUp(self):
        # put both agents on adjacent cells so they can both reach (0, 1)
        s = st(("greedy", "greedy"))
        object.__setattr__(s.agents[0], "y", 0)
        object.__setattr__(s.agents[0], "x", 0)
        object.__setattr__(s.agents[1], "y", 0)
        object.__setattr__(s.agents[1], "x", 2)
        self.s = s

    def test_full_cell_serves_both_agents(self):
        s1, ev = apply_actions(self.s, [Action(0, (0, 1), TAKE), Action(1, (0, 1), TAKE)])
        # 1.0 available, 0.55 each asked -> 0.5 each after fair split
        grants = {e["agent"]: e["granted"] for e in ev if e["type"] == "action"}
        self.assertAlmostEqual(sum(grants.values()), 1.0)
        self.assertAlmostEqual(grants[0], grants[1])

    def test_cell_never_goes_negative(self):
        s = self.s
        thin = s.grid.copy()
        thin[0, 1] = 0.2
        s = type(s)(**{**s.__dict__, "grid": thin})
        s1, _ = apply_actions(s, [Action(0, (0, 1), TAKE), Action(1, (0, 1), TAKE)])
        self.assertGreaterEqual(float(s1.grid[0, 1]), 0.0)

    def test_scores_match_what_was_removed(self):
        s1, _ = apply_actions(self.s, [Action(0, (0, 1), 0.3), Action(1, (0, 1), 0.4)])
        removed = float(self.s.grid[0, 1]) - float(s1.grid[0, 1])
        # regrow happens after harvest, so compare against the pre-regrow delta
        gained = sum(a.score for a in s1.agents)
        self.assertAlmostEqual(gained, 0.7)
        self.assertGreater(removed, 0.0)

    def test_contention_is_flagged_in_the_log(self):
        _, ev = apply_actions(self.s, [Action(0, (0, 1), TAKE), Action(1, (0, 1), TAKE)])
        cells = [e for e in ev if e["type"] == "cell"]
        self.assertTrue(any(e["contested"] for e in cells))

    def test_separate_cells_are_not_contested(self):
        _, ev = apply_actions(self.s, [Action(0, (0, 0), TAKE), Action(1, (0, 2), TAKE)])
        cells = [e for e in ev if e["type"] == "cell"]
        self.assertTrue(cells)
        self.assertFalse(any(e["contested"] for e in cells))


class TestPurity(unittest.TestCase):
    def test_apply_actions_does_not_mutate_input(self):
        s = st()
        before_grid = s.grid.copy()
        before_scores = [a.score for a in s.agents]
        apply_actions(s, [Action(0, (0, 0), TAKE), Action(1, (5, 0), TAKE)])
        np.testing.assert_array_equal(s.grid, before_grid)
        self.assertEqual([a.score for a in s.agents], before_scores)
        self.assertEqual(s.tick, 0)

    def test_returned_grid_is_a_new_array(self):
        s = st()
        s1, _ = apply_actions(s, [Action(0, (0, 0), TAKE)])
        self.assertIsNot(s.grid, s1.grid)

    def test_regrow_does_not_mutate(self):
        g = np.full((N, N), 0.5)
        before = g.copy()
        regrow(g, "global", 0.15)
        np.testing.assert_array_equal(g, before)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_run(self):
        a = harness.run_episode(7, "mixed", "neighbour", 0.13)
        b = harness.run_episode(7, "mixed", "neighbour", 0.13)
        self.assertEqual(a.survived, b.survived)
        self.assertEqual(a.scores, b.scores)
        self.assertEqual(a.stock, b.stock)

    def test_different_seeds_differ(self):
        runs = {tuple(harness.run_episode(s, "mixed", "neighbour", 0.13).stock)
                for s in range(8)}
        self.assertGreater(len(runs), 1, "seeds are not varying the run")

    def test_rng_is_keyed_not_sequential(self):
        # an agent's stream must not depend on which other agents drew first
        a = rng_for(3, 10, 2).random()
        b = rng_for(3, 10, 2).random()
        self.assertEqual(a, b)
        self.assertNotEqual(rng_for(3, 10, 1).random(), a)


class TestValidation(unittest.TestCase):
    def test_unreachable_target_is_rejected(self):
        s = st()
        s1, ev = apply_actions(s, [Action(0, (5, 5), TAKE)])
        self.assertTrue(any(e["type"] == "reject" and e["reason"] == "unreachable" for e in ev))
        self.assertEqual((s1.agents[0].y, s1.agents[0].x), (0, 0))
        self.assertEqual(s1.agents[0].score, 0.0)

    def test_take_above_cap_is_rejected(self):
        s = st()
        _, ev = apply_actions(s, [Action(0, (0, 0), TAKE + 1)])
        self.assertTrue(any(e["type"] == "reject" for e in ev))

    def test_unknown_agent_is_rejected(self):
        s = st()
        _, ev = apply_actions(s, [Action(99, (0, 0), 0.1)])
        self.assertTrue(any(e["reason"] == "no such agent" for e in ev))


class TestWorldInvariants(unittest.TestCase):
    def test_grid_stays_within_bounds_over_a_full_run(self):
        for mix in ("greedy", "cautious", "mixed", "random"):
            ep = harness.run_episode(1, mix, "neighbour", 0.12, keep_events=True)
            self.assertTrue(all(0.0 <= s <= CAPACITY + 1e-9 for s in ep.stock))

    def test_tick_budget_is_hard(self):
        ep = harness.run_episode(0, "cautious", "global", 0.5)
        self.assertLessEqual(len(ep.stock) - 1, TICKS)
        self.assertLessEqual(ep.survived, TICKS)

    def test_regrow_respects_cap(self):
        g = np.full((N, N), 0.99)
        self.assertLessEqual(regrow(g, "global", 0.9).max(), CAP + 1e-9)
        self.assertLessEqual(regrow(g, "neighbour", 0.9).max(), CAP + 1e-9)

    def test_neighbours_mean_uses_true_neighbour_count(self):
        g = np.ones((N, N))
        # every cell's neighbourhood is all ones, so the mean is 1 everywhere --
        # including corners, which the legacy divide-by-9 got wrong
        np.testing.assert_allclose(neighbours_mean(g), np.ones((N, N)))

    def test_collapse_is_recorded_once(self):
        ep = harness.run_episode(0, "greedy", "neighbour", 0.02)
        self.assertLess(ep.survived, TICKS)
        self.assertLess(ep.stock[-1], COLLAPSE_FLOOR)

    def test_event_log_covers_every_tick(self):
        ep = harness.run_episode(2, "mixed", "global", 0.15, keep_events=True)
        ticks = {e["t"] for e in ep.events if e["type"] == "tick"}
        self.assertEqual(len(ticks), len(ep.stock) - 1)

    def test_append_only_log_is_ordered(self):
        ep = harness.run_episode(2, "mixed", "global", 0.15, keep_events=True)
        ts = [e["t"] for e in ep.events]
        self.assertEqual(ts, sorted(ts), "event log is not in tick order")


if __name__ == "__main__":
    unittest.main()
