"""
Unit tests for the Overgraze world engine.

    python -m unittest test_world -v

The resolution rules get the most attention, especially two agents harvesting
the same cell on the same tick -- the case the sequential engine could not
represent at all.
"""

import unittest

import numpy as np

from world import (CAP, CAPACITY, COLLAPSE_FLOOR, N, PLANT, PUNISH_COST,
                   PUNISH_FINE, SAY_LIMIT, TAKE, TICKS, Action, apply_actions,
                   history, initial_state, listen, look, neighbours_mean,
                   regrow, resolve_cell, rng_for, status)
import harness


def st(kinds=("greedy", "greedy"), seed=0, rule="global", r=0.15, **ab):
    return initial_state(seed, list(kinds), rule=rule, r=r, **ab)


def at(state, agent_id, y, x):
    """Return a copy of `state` with one agent moved -- test scaffolding only."""
    agents = tuple(type(a)(a.id, a.kind, y, x, a.score) if a.id == agent_id else a
                   for a in state.agents)
    return type(state)(**{**state.__dict__, "agents": agents})


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
    """Two agents harvesting the same cell on the same tick, end to end.

    Harvest now takes from the agent's own cell, so contention means two agents
    standing on the same square and both harvesting.
    """

    def setUp(self):
        s = st(("greedy", "greedy"))
        self.s = at(at(s, 0, 2, 2), 1, 2, 2)     # both standing on (2, 2)

    def _harvest_both(self, a=TAKE, b=TAKE, state=None):
        return apply_actions(state or self.s,
                             [Action(0, "harvest", amount=a), Action(1, "harvest", amount=b)])

    def test_full_cell_serves_both_agents(self):
        _, ev = self._harvest_both()
        grants = {e["agent"]: e["granted"] for e in ev if e["type"] == "action"}
        self.assertAlmostEqual(sum(grants.values()), 1.0)   # 0.55+0.55 capped at 1.0
        self.assertAlmostEqual(grants[0], grants[1])

    def test_cell_never_goes_negative(self):
        thin = self.s.grid.copy()
        thin[2, 2] = 0.2
        s = type(self.s)(**{**self.s.__dict__, "grid": thin})
        s1, _ = self._harvest_both(state=s)
        self.assertGreaterEqual(float(s1.grid[2, 2]), 0.0)

    def test_scores_match_what_was_asked(self):
        s1, _ = self._harvest_both(0.3, 0.4)
        self.assertAlmostEqual(sum(a.score for a in s1.agents), 0.7)

    def test_contention_is_flagged_in_the_log(self):
        _, ev = self._harvest_both()
        self.assertTrue(any(e["contested"] for e in ev if e["type"] == "cell"))

    def test_separate_cells_are_not_contested(self):
        s = at(at(st(("greedy", "greedy")), 0, 1, 1), 1, 4, 4)
        _, ev = apply_actions(s, [Action(0, "harvest", amount=TAKE),
                                  Action(1, "harvest", amount=TAKE)])
        cells = [e for e in ev if e["type"] == "cell"]
        self.assertTrue(cells)
        self.assertFalse(any(e["contested"] for e in cells))


class TestOneActionPerTick(unittest.TestCase):
    """'You already acted this tick' has to be an error, not a silent overwrite."""

    def test_second_physical_action_is_rejected(self):
        s = st()
        s1, ev = apply_actions(s, [Action(0, "harvest", amount=0.3),
                                   Action(0, "harvest", amount=0.5)])
        rejects = [e for e in ev if e["type"] == "reject"]
        self.assertTrue(any(e["reason"] == "you already acted this tick" for e in rejects))
        # the FIRST intent stands; the second does not overwrite it
        self.assertAlmostEqual(s1.agents[0].score, 0.3)

    def test_mixed_kinds_still_count_as_acting_twice(self):
        s = st()
        _, ev = apply_actions(s, [Action(0, "harvest", amount=0.3), Action(0, "plant")])
        self.assertTrue(any(e["reason"] == "you already acted this tick"
                            for e in ev if e["type"] == "reject"))

    def test_speech_does_not_count_as_the_physical_action(self):
        s = st()
        s1, ev = apply_actions(s, [Action(0, "harvest", amount=0.4),
                                   Action(0, "say", text="mine")])
        self.assertFalse([e for e in ev if e["type"] == "reject"])
        self.assertAlmostEqual(s1.agents[0].score, 0.4)
        self.assertEqual(len(s1.messages), 1)

    def test_second_speech_is_rejected(self):
        s = st()
        _, ev = apply_actions(s, [Action(0, "say", text="one"), Action(0, "say", text="two")])
        self.assertTrue(any(e["reason"] == "you already spoke this tick"
                            for e in ev if e["type"] == "reject"))

    def test_harvesting_an_empty_cell_says_so(self):
        s = st()
        empty = s.grid.copy(); empty[0, 0] = 0.0
        s = type(s)(**{**s.__dict__, "grid": empty})
        _, ev = apply_actions(s, [Action(0, "harvest", amount=0.5)])
        self.assertTrue(any(e["reason"] == "nothing left in this cell"
                            for e in ev if e["type"] == "reject"))


class TestMoveAndPlant(unittest.TestCase):
    def test_move_changes_position(self):
        s = st()
        s1, _ = apply_actions(s, [Action(0, "move", direction="east")])
        self.assertEqual((s1.agents[0].y, s1.agents[0].x), (0, 1))

    def test_move_off_the_grid_is_rejected(self):
        s = st()
        s1, ev = apply_actions(s, [Action(0, "move", direction="north")])
        self.assertTrue(any(e["reason"] == "that would leave the world"
                            for e in ev if e["type"] == "reject"))
        self.assertEqual((s1.agents[0].y, s1.agents[0].x), (0, 0))

    def test_moving_does_not_harvest(self):
        s = st()
        s1, _ = apply_actions(s, [Action(0, "move", direction="east")])
        self.assertEqual(s1.agents[0].score, 0.0)

    def test_plant_adds_to_the_agents_own_cell(self):
        s = st()
        thin = s.grid.copy(); thin[0, 0] = 0.2
        s = type(s)(**{**s.__dict__, "grid": thin})
        s1, ev = apply_actions(s, [Action(0, "plant")])
        planted = [e for e in ev if e["type"] == "cell" and e["cause"] == "plant"]
        self.assertEqual(len(planted), 1)
        self.assertAlmostEqual(planted[0]["after"], 0.2 + PLANT)

    def test_plant_never_exceeds_capacity(self):
        s = st()
        near = s.grid.copy(); near[0, 0] = CAP - 0.01
        s = type(s)(**{**s.__dict__, "grid": near})
        s1, _ = apply_actions(s, [Action(0, "plant")])
        self.assertLessEqual(float(s1.grid[0, 0]), CAP + 1e-9)

    def test_planting_a_full_cell_is_rejected(self):
        s = st()
        _, ev = apply_actions(s, [Action(0, "plant")])
        self.assertTrue(any(e["reason"] == "this cell is already full"
                            for e in ev if e["type"] == "reject"))

    def test_harvest_and_plant_on_one_cell_are_order_independent(self):
        s = at(at(st(("greedy", "cautious")), 0, 3, 3), 1, 3, 3)
        a, _ = apply_actions(s, [Action(0, "harvest", amount=0.5), Action(1, "plant")])
        b, _ = apply_actions(s, [Action(1, "plant"), Action(0, "harvest", amount=0.5)])
        np.testing.assert_allclose(a.grid, b.grid)


class TestMoveAndActShareATick(unittest.TestCase):
    """A move and a resource action may share a tick; the move resolves first."""

    def test_harvest_lands_on_the_destination_cell(self):
        s = at(st(), 0, 2, 2)
        thin = s.grid.copy(); thin[2, 2] = 0.1; thin[2, 3] = 1.0
        s = type(s)(**{**s.__dict__, "grid": thin})
        s1, _ = apply_actions(s, [Action(0, "move", direction="east"),
                                  Action(0, "harvest", amount=TAKE)])
        self.assertEqual((s1.agents[0].y, s1.agents[0].x), (2, 3))
        self.assertAlmostEqual(s1.agents[0].score, TAKE)      # took from (2,3), not (2,2)
        self.assertAlmostEqual(float(s.grid[2, 2]), 0.1)      # origin untouched

    def test_second_move_is_rejected(self):
        s = at(st(), 0, 2, 2)
        s1, ev = apply_actions(s, [Action(0, "move", direction="east"),
                                   Action(0, "move", direction="south")])
        self.assertTrue(any(e["reason"] == "you already moved this tick"
                            for e in ev if e["type"] == "reject"))
        self.assertEqual((s1.agents[0].y, s1.agents[0].x), (2, 3))   # first move stands

    def test_harvest_is_validated_against_the_destination(self):
        s = at(st(), 0, 2, 2)
        empty = s.grid.copy(); empty[2, 3] = 0.0
        s = type(s)(**{**s.__dict__, "grid": empty})
        _, ev = apply_actions(s, [Action(0, "move", direction="east"),
                                  Action(0, "harvest", amount=TAKE)])
        self.assertTrue(any(e["reason"] == "nothing left in this cell"
                            for e in ev if e["type"] == "reject"))

    def test_two_agents_converging_on_one_cell_contend(self):
        s = at(at(st(("greedy", "greedy")), 0, 2, 1), 1, 2, 3)
        _, ev = apply_actions(s, [Action(0, "move", direction="east"),
                                  Action(0, "harvest", amount=TAKE),
                                  Action(1, "move", direction="west"),
                                  Action(1, "harvest", amount=TAKE)])
        contested = [e for e in ev if e["type"] == "cell" and e["contested"]]
        self.assertEqual(len(contested), 1)
        self.assertEqual(contested[0]["cell"], (2, 2))

    def test_move_plus_plant_plants_at_the_destination(self):
        s = at(st(), 0, 2, 2)
        thin = s.grid.copy(); thin[2, 3] = 0.2
        s = type(s)(**{**s.__dict__, "grid": thin})
        # assert on the event, not the grid: regrowth runs after the plant
        _, ev = apply_actions(s, [Action(0, "move", direction="east"),
                                  Action(0, "plant")])
        planted = [e for e in ev if e["type"] == "cell" and e["cause"] == "plant"]
        self.assertEqual([e["cell"] for e in planted], [(2, 3)])
        self.assertAlmostEqual(planted[0]["after"], 0.2 + PLANT)


class TestSpeech(unittest.TestCase):
    def test_message_is_heard_in_range(self):
        s = at(at(st(("greedy", "greedy")), 0, 2, 2), 1, 2, 3)
        s1, ev = apply_actions(s, [Action(0, "say", text="stop at a third")])
        self.assertEqual([e["heard_by"] for e in ev if e["type"] == "speech"], [[1]])
        self.assertEqual(listen(s1, 1)[0]["text"], "stop at a third")

    def test_message_is_not_heard_out_of_range(self):
        # speech is grid-wide by default, so range has to be set to test it
        s = at(at(st(("greedy", "greedy"), speech_radius=1), 0, 0, 0), 1, 5, 5)
        s1, ev = apply_actions(s, [Action(0, "say", text="hello")])
        self.assertEqual([e["heard_by"] for e in ev if e["type"] == "speech"], [[]])
        self.assertEqual(listen(s1, 1), [])

    def test_you_do_not_hear_yourself(self):
        s = st()
        s1, _ = apply_actions(s, [Action(0, "say", text="thinking aloud")])
        self.assertEqual(listen(s1, 0), [])

    def test_chat_disabled_rejects_speech(self):
        s = st(chat=False)
        s1, ev = apply_actions(s, [Action(0, "say", text="hello")])
        self.assertTrue(any(e["reason"] == "chat is disabled in this run"
                            for e in ev if e["type"] == "reject"))
        self.assertEqual(len(s1.messages), 0)

    def test_overlong_and_empty_messages_are_rejected(self):
        s = st()
        _, ev = apply_actions(s, [Action(0, "say", text="x" * (SAY_LIMIT + 1))])
        self.assertTrue([e for e in ev if e["type"] == "reject"])
        _, ev2 = apply_actions(s, [Action(0, "say", text="   ")])
        self.assertTrue(any(e["reason"] == "empty message"
                            for e in ev2 if e["type"] == "reject"))

    def test_anonymity_masks_the_speaker_in_the_view_but_not_the_log(self):
        s = at(at(st(("greedy", "greedy"), anonymous=True), 0, 2, 2), 1, 2, 3)
        s1, ev = apply_actions(s, [Action(0, "say", text="who said that")])
        self.assertIsNone(listen(s1, 1)[0]["speaker"])
        self.assertEqual([e["agent"] for e in ev if e["type"] == "speech"], [0])
        self.assertEqual(s1.messages[0].speaker, 0)


class TestSpeechReach(unittest.TestCase):
    """Speech is grid-wide by default; range is a per-run switch."""

    def test_broadcast_reaches_the_far_corner(self):
        s = at(at(st(("greedy", "greedy")), 0, 0, 0), 1, 5, 5)
        s1, ev = apply_actions(s, [Action(0, "say", text="lets stop at a third")])
        self.assertEqual([e["heard_by"] for e in ev if e["type"] == "speech"], [[1]])
        self.assertEqual(listen(s1, 1)[0]["text"], "lets stop at a third")

    def test_radius_limits_reach_when_set(self):
        s = at(at(st(("greedy", "greedy"), speech_radius=1), 0, 0, 0), 1, 5, 5)
        s1, ev = apply_actions(s, [Action(0, "say", text="too far")])
        self.assertEqual([e["heard_by"] for e in ev if e["type"] == "speech"], [[]])
        self.assertEqual(listen(s1, 1), [])

    def test_listening_is_fixed_at_speaking_time_not_current_position(self):
        """Walking to where a conversation happened must not reveal it."""
        s = at(at(st(("greedy", "greedy"), speech_radius=1), 0, 0, 0), 1, 5, 5)
        s1, _ = apply_actions(s, [Action(0, "say", text="a secret")])
        self.assertEqual(listen(s1, 1), [])
        walked = at(s1, 1, 0, 1)              # agent 1 walks right next door
        self.assertEqual(listen(walked, 1), [], "past messages leaked by position")

    def test_audience_is_recorded_on_the_message(self):
        s = at(at(st(("greedy", "greedy")), 0, 2, 2), 1, 2, 3)
        s1, _ = apply_actions(s, [Action(0, "say", text="hello")])
        self.assertEqual(s1.messages[0].heard_by, (1,))


class TestHistory(unittest.TestCase):
    def test_own_actions_are_recorded(self):
        s = st()
        s1, _ = apply_actions(s, [Action(0, "harvest", amount=0.4)])
        s2, _ = apply_actions(s1, [Action(0, "harvest", amount=0.3)])
        h = history(s2, 0)
        self.assertEqual([m["action"] for m in h["my_actions"]], ["harvest", "harvest"])
        self.assertAlmostEqual(h["my_recent_harvest"], 0.7)

    def test_history_is_per_agent(self):
        s = st(("greedy", "greedy"))
        s1, _ = apply_actions(s, [Action(0, "harvest", amount=0.4)])
        self.assertEqual(history(s1, 1)["my_actions"], [])

    def test_commons_series_tracks_stock(self):
        s = st()
        s1, _ = apply_actions(s, [Action(0, "harvest", amount=TAKE)])
        h = history(s1, 0)
        self.assertEqual(h["commons_capacity"], CAPACITY)
        self.assertEqual(len(h["commons"]), 2)          # start plus one tick
        self.assertAlmostEqual(h["commons_now"], s1.stock, places=3)   # reported rounded

    def test_falling_commons_is_reported_as_falling(self):
        ep_state = initial_state(0, ["greedy"] * 4, rule="global", r=0.05)
        import harness
        streams = harness.agent_streams(0, 4)
        for _ in range(12):
            acts = [a2 for a in ep_state.agents
                    for a2 in harness.greedy(ep_state, a, streams[a.id])]
            ep_state, _ = apply_actions(ep_state, acts)
        h = history(ep_state, 0)
        self.assertEqual(h["commons_trend"], "falling")
        self.assertIn("ticks_to_collapse_at_this_rate", h)

    def test_share_stock_off_withholds_the_commons(self):
        s = st(share_stock=False)
        s1, _ = apply_actions(s, [Action(0, "harvest", amount=0.4)])
        h = history(s1, 0)
        self.assertNotIn("commons", h)
        self.assertIn("my_actions", h)          # own history is always available

    def test_window_limits_how_far_back_an_agent_sees(self):
        s = st()
        for _ in range(8):
            s, _ = apply_actions(s, [Action(0, "harvest", amount=0.1)])
        self.assertEqual(len(history(s, 0, window=3)["my_actions"]), 3)


class TestVision(unittest.TestCase):
    def test_look_shows_only_cells_in_range(self):
        s = at(st(), 0, 2, 2)
        v = look(s, 0)
        self.assertEqual(len(v["cells"]), 9)                 # 3x3 at radius 1
        self.assertNotIn((5, 5), v["cells"])

    def test_look_is_clipped_at_the_edge(self):
        s = st()
        self.assertEqual(len(look(s, 0)["cells"]), 4)        # corner sees 2x2

    def test_look_reports_nearby_agents_only(self):
        s = at(at(st(("greedy", "greedy")), 0, 2, 2), 1, 5, 5)
        self.assertEqual(look(s, 0)["agents"], [])
        s2 = at(s, 1, 2, 3)
        self.assertEqual(len(look(s2, 0)["agents"]), 1)

    def test_anonymity_masks_agent_identity_in_look(self):
        s = at(at(st(("greedy", "greedy"), anonymous=True), 0, 2, 2), 1, 2, 3)
        self.assertIsNone(look(s, 0)["agents"][0]["agent"])

    def test_status_reports_the_run_rules(self):
        s = st(punish=True)
        r = status(s, 0)["rules"]
        self.assertEqual(r["take_limit"], TAKE)
        self.assertTrue(r["punish"])
        self.assertEqual(r["run_ends_below"], COLLAPSE_FLOOR)
        self.assertEqual(r["speech_reaches"], "everyone")


class TestPunish(unittest.TestCase):
    def test_punish_is_off_by_default(self):
        s = at(at(st(("greedy", "greedy")), 0, 2, 2), 1, 2, 3)
        _, ev = apply_actions(s, [Action(0, "punish", subject=1)])
        self.assertTrue(any(e["reason"] == "punish is disabled in this run"
                            for e in ev if e["type"] == "reject"))

    def test_punish_costs_both_parties(self):
        s = at(at(st(("greedy", "greedy"), punish=True), 0, 2, 2), 1, 2, 3)
        s1, ev = apply_actions(s, [Action(0, "punish", subject=1)])
        self.assertAlmostEqual(s1.agents[0].score, -PUNISH_COST)
        self.assertAlmostEqual(s1.agents[1].score, -PUNISH_FINE)
        self.assertTrue([e for e in ev if e["type"] == "punish"])

    def test_cannot_punish_out_of_range(self):
        s = at(at(st(("greedy", "greedy"), punish=True), 0, 0, 0), 1, 5, 5)
        _, ev = apply_actions(s, [Action(0, "punish", subject=1)])
        self.assertTrue(any(e["reason"] == "that agent is out of range"
                            for e in ev if e["type"] == "reject"))

    def test_cannot_punish_yourself_or_a_stranger(self):
        s = st(("greedy", "greedy"), punish=True)
        _, ev = apply_actions(s, [Action(0, "punish", subject=0)])
        self.assertTrue(any(e["reason"] == "no such agent to punish"
                            for e in ev if e["type"] == "reject"))
        _, ev2 = apply_actions(s, [Action(0, "punish", subject=99)])
        self.assertTrue(any(e["reason"] == "no such agent to punish"
                            for e in ev2 if e["type"] == "reject"))


class TestPurity(unittest.TestCase):
    def test_apply_actions_does_not_mutate_input(self):
        s = st()
        before_grid = s.grid.copy()
        before_scores = [a.score for a in s.agents]
        apply_actions(s, [Action(0, "harvest", amount=TAKE),
                          Action(1, "harvest", amount=TAKE),
                          Action(0, "say", text="hello")])
        np.testing.assert_array_equal(s.grid, before_grid)
        self.assertEqual([a.score for a in s.agents], before_scores)
        self.assertEqual(s.tick, 0)
        self.assertEqual(s.messages, ())

    def test_returned_grid_is_a_new_array(self):
        s = st()
        s1, _ = apply_actions(s, [Action(0, "harvest", amount=TAKE)])
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
    def test_take_above_cap_is_rejected(self):
        s = st()
        s1, ev = apply_actions(s, [Action(0, "harvest", amount=TAKE + 1)])
        self.assertTrue(any(e["type"] == "reject" for e in ev))
        self.assertEqual(s1.agents[0].score, 0.0)

    def test_unknown_agent_is_rejected(self):
        s = st()
        _, ev = apply_actions(s, [Action(99, "harvest", amount=0.1)])
        self.assertTrue(any(e["reason"] == "no such agent" for e in ev))

    def test_unknown_action_kind_is_rejected(self):
        s = st()
        _, ev = apply_actions(s, [Action(0, "teleport")])
        self.assertTrue(any("unknown action" in e["reason"] for e in ev if e["type"] == "reject"))

    def test_unknown_direction_is_rejected(self):
        s = st()
        _, ev = apply_actions(s, [Action(0, "move", direction="widdershins")])
        self.assertTrue(any("unknown direction" in e["reason"] for e in ev if e["type"] == "reject"))


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
