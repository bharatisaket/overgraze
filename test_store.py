"""
Tests for persistence and turn resolution.

    python -m unittest test_store -v

The engine's own rules are covered by test_world. What matters here is that
nothing important lives in a session: state survives a fresh connection, a tick
resolves only when everyone has committed, and a stalled agent cannot freeze
the world.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np

import store
from world import TAKE, Action, apply_actions, initial_state


class TempDB(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "t.db"
        self.con = store.connect(self.path)

    def tearDown(self):
        self.con.close()
        self.dir.cleanup()


class TestStateCodec(TempDB):
    def test_a_fresh_state_round_trips(self):
        s = initial_state(3, ["a", "b"], rule="neighbour", r=0.07, monitoring="global")
        back = store.decode_state(store.encode_state(s))
        np.testing.assert_allclose(back.grid, s.grid)
        self.assertEqual(back.agents, s.agents)
        self.assertEqual((back.rule, back.r, back.monitoring), ("neighbour", 0.07, "global"))

    def test_a_played_state_round_trips_including_logs(self):
        s = initial_state(0, ["a", "b"], r=0.05, monitoring="global")
        s, _ = apply_actions(s, [Action(0, "harvest", amount=0.4),
                                 Action(0, "say", text="mine"),
                                 Action(1, "plant")])
        back = store.decode_state(store.encode_state(s))
        self.assertEqual(back.action_log, s.action_log)
        self.assertEqual(back.messages, s.messages)
        self.assertEqual(back.stock_log, s.stock_log)
        self.assertAlmostEqual(back.stock, s.stock)

    def test_ablation_switches_survive(self):
        s = initial_state(0, ["a"], noise=0.2, misreport=0.3, punish=True,
                          anonymous=True, share_stock=False, speech_radius=2)
        b = store.decode_state(store.encode_state(s))
        self.assertEqual((b.noise, b.misreport, b.punish, b.anonymous,
                          b.share_stock, b.speech_radius),
                         (0.2, 0.3, True, True, False, 2))


class TestRunsAndTokens(TempDB):
    def test_each_seat_gets_its_own_token(self):
        info = store.create_run(self.con, ["a", "b", "c", "d"])
        self.assertEqual(len(set(info["tokens"].values())), 4)
        for name, tok in info["tokens"].items():
            row = store.player_for(self.con, tok)
            self.assertEqual(row["name"], name)
            self.assertEqual(row["run_id"], info["run_id"])

    def test_an_unknown_token_resolves_to_nobody(self):
        store.create_run(self.con, ["a"])
        self.assertIsNone(store.player_for(self.con, "not-a-token"))

    def test_state_survives_a_fresh_connection(self):
        """Nothing lives in a session -- a new process must see the same world."""
        info = store.create_run(self.con, ["a", "b"], r=0.05)
        s = store.load_state(self.con, info["run_id"])
        s, _ = apply_actions(s, [Action(0, "harvest", amount=0.5)])
        store.save_state(self.con, info["run_id"], s)

        other = store.connect(self.path)
        try:
            seen = store.load_state(other, info["run_id"])
            self.assertEqual(seen.tick, 1)
            self.assertAlmostEqual(seen.agents[0].score, 0.5)
        finally:
            other.close()


class TestIntents(TempDB):
    def setUp(self):
        super().setUp()
        self.info = store.create_run(self.con, ["a", "b"], r=0.05)
        self.run = self.info["run_id"]

    def test_a_second_physical_action_is_refused(self):
        self.assertTrue(store.record_intent(self.con, self.run, 0, 0,
                                            Action(0, "harvest", amount=0.3)))
        self.assertFalse(store.record_intent(self.con, self.run, 0, 0,
                                             Action(0, "plant")))

    def test_speech_is_a_separate_channel(self):
        store.record_intent(self.con, self.run, 0, 0, Action(0, "harvest", amount=0.3))
        self.assertTrue(store.record_intent(self.con, self.run, 0, 0,
                                            Action(0, "say", text="hello")))

    def test_moving_and_harvesting_are_separate_channels(self):
        store.record_intent(self.con, self.run, 0, 0, Action(0, "move", direction="east"))
        self.assertTrue(store.record_intent(self.con, self.run, 0, 0,
                                            Action(0, "harvest", amount=0.3)))

    def test_only_committed_agents_count_toward_the_barrier(self):
        store.record_intent(self.con, self.run, 0, 0, Action(0, "say", text="hi"))
        self.assertEqual(store.acted_this_tick(self.con, self.run, 0), set())
        store.record_intent(self.con, self.run, 0, 0, Action(0, "harvest", amount=0.3))
        self.assertEqual(store.acted_this_tick(self.con, self.run, 0), {0})

    def test_resolving_clears_the_intents_and_advances(self):
        store.record_intent(self.con, self.run, 0, 0, Action(0, "harvest", amount=0.3))
        store.record_intent(self.con, self.run, 0, 1, Action(1, "harvest", amount=0.3))
        s, ev = store.resolve_tick(self.con, self.run, 0)
        self.assertEqual(s.tick, 1)
        self.assertEqual(store.pending(self.con, self.run, 0), [])
        self.assertTrue(store.read_events(self.con, self.run, 0))

    def test_resolving_a_stale_tick_does_nothing(self):
        store.resolve_tick(self.con, self.run, 0)
        s, ev = store.resolve_tick(self.con, self.run, 0)     # already past
        self.assertEqual(s.tick, 1)
        self.assertEqual(ev, [])


class TestBarrier(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = store.connect(Path(self.dir.name) / "t.db")
        self.info = store.create_run(self.con, ["a", "b"], r=0.05, monitoring="global")
        self.run = self.info["run_id"]

    def tearDown(self):
        self.con.close()
        self.dir.cleanup()

    async def test_the_tick_resolves_once_everyone_has_acted(self):
        both = await asyncio.gather(
            store.submit_and_wait(self.con, self.run, 0, [Action(0, "harvest", amount=0.4)]),
            store.submit_and_wait(self.con, self.run, 1, [Action(1, "harvest", amount=0.4)]),
        )
        for r in both:
            self.assertTrue(r["resolved"])
            self.assertEqual(r["tick"], 1)
        self.assertAlmostEqual(both[0]["score"], 0.4)

    async def test_a_lone_agent_waits_and_then_the_barrier_lets_go(self):
        """One stalled forager must not freeze the world."""
        r = await store.submit_and_wait(self.con, self.run, 0,
                                        [Action(0, "harvest", amount=0.4)], timeout=0.25)
        self.assertTrue(r["resolved"])
        self.assertEqual(r["tick"], 1)
        # the absent agent was recorded as noop, and scored nothing
        s = store.load_state(self.con, self.run)
        self.assertEqual(s.agents[1].score, 0.0)

    async def test_contention_is_reported_back_honestly(self):
        """Two agents on one cell: what you get is not what you asked for."""
        s = store.load_state(self.con, self.run)
        moved = type(s)(**{**s.__dict__, "agents": tuple(
            type(a)(a.id, a.kind, 2, 2, a.score) for a in s.agents)})
        store.save_state(self.con, self.run, moved)

        both = await asyncio.gather(
            store.submit_and_wait(self.con, self.run, 0, [Action(0, "harvest", amount=TAKE)]),
            store.submit_and_wait(self.con, self.run, 1, [Action(1, "harvest", amount=TAKE)]),
        )
        got = [r["score"] for r in both]
        self.assertAlmostEqual(sum(got), 1.0)          # the cell held 1.0, not 1.1
        self.assertLess(got[0], TAKE)

    async def test_rejections_come_back_to_the_agent_that_earned_them(self):
        r = await store.submit_and_wait(self.con, self.run, 0,
                                        [Action(0, "move", direction="north")],
                                        timeout=0.25)
        reasons = [e["reason"] for e in r["rejected"]]
        self.assertIn("that would leave the world", reasons)

    async def test_a_finished_run_refuses_further_actions(self):
        s = store.load_state(self.con, self.run)
        dead = type(s)(**{**s.__dict__, "collapsed_at": 5})
        store.save_state(self.con, self.run, dead)
        r = await store.submit_and_wait(self.con, self.run, 0,
                                        [Action(0, "harvest", amount=0.4)], timeout=0.25)
        self.assertFalse(r["resolved"])
        self.assertIn("finished", r["reason"])


if __name__ == "__main__":
    unittest.main()
