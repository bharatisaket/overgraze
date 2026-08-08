"""
Tests for the operational layer: rate limiting, health, and the admin routes.

    python -m unittest test_deploy -v

The admin routes delete data and mint credentials, so the interesting cases are
the ones where they must refuse.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import deploy


class TestConfig(unittest.TestCase):
    def test_db_path_follows_the_environment(self):
        with mock.patch.dict(os.environ, {"OVERGRAZE_DB": "/data/og.db"}):
            self.assertEqual(deploy.db_path(), Path("/data/og.db"))

    def test_db_path_falls_back_beside_the_code(self):
        with mock.patch.dict(os.environ, {"OVERGRAZE_DB": ""}):
            self.assertEqual(deploy.db_path().name, "overgraze.db")

    def test_a_platform_port_means_bind_every_interface(self):
        with mock.patch.dict(os.environ, {"PORT": "9999"}, clear=False):
            self.assertEqual(deploy.port(), 9999)
            self.assertEqual(deploy.host(), "0.0.0.0")

    def test_local_runs_stay_on_loopback(self):
        env = {k: v for k, v in os.environ.items() if k not in ("PORT", "OVERGRAZE_HOST")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(deploy.host(), "127.0.0.1")

    def test_admin_is_closed_when_no_secret_is_set(self):
        with mock.patch.dict(os.environ, {"OVERGRAZE_ADMIN_TOKEN": ""}):
            self.assertIsNone(deploy.admin_token())
        with mock.patch.dict(os.environ, {"OVERGRAZE_ADMIN_TOKEN": "   "}):
            self.assertIsNone(deploy.admin_token())


class TestRateLimiter(unittest.TestCase):
    def test_calls_under_the_limit_are_allowed(self):
        rl = deploy.RateLimiter(per_minute=5)
        for i in range(5):
            ok, _ = rl.check("tok", now=100.0 + i)
            self.assertTrue(ok)

    def test_the_call_over_the_limit_is_refused(self):
        rl = deploy.RateLimiter(per_minute=3)
        for i in range(3):
            rl.check("tok", now=100.0 + i)
        ok, wait = rl.check("tok", now=103.0)
        self.assertFalse(ok)
        self.assertGreater(wait, 0)

    def test_the_window_slides(self):
        rl = deploy.RateLimiter(per_minute=2)
        rl.check("tok", now=100.0)
        rl.check("tok", now=101.0)
        self.assertFalse(rl.check("tok", now=102.0)[0])
        # once the first call ages out of the window there is room again
        self.assertTrue(rl.check("tok", now=161.0)[0])

    def test_tokens_are_limited_separately(self):
        rl = deploy.RateLimiter(per_minute=1)
        self.assertTrue(rl.check("a", now=100.0)[0])
        self.assertTrue(rl.check("b", now=100.0)[0])
        self.assertFalse(rl.check("a", now=100.0)[0])

    def test_a_thousand_harvests_in_a_loop_get_cut_off(self):
        """The scenario the plan names outright."""
        rl = deploy.RateLimiter(per_minute=120)
        allowed = sum(rl.check("greedy", now=100.0)[0] for _ in range(1000))
        self.assertEqual(allowed, 120)


class TestAdminRoutes(unittest.IsolatedAsyncioTestCase):
    """Exercised through the ASGI app, so routing and auth are both covered."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = Path(self.dir.name) / "t.db"
        self.env = mock.patch.dict(os.environ, {
            "OVERGRAZE_DB": str(self.db),
            "OVERGRAZE_ADMIN_TOKEN": "s3cret",
        })
        self.env.start()
        import server
        server._con = None                       # pick up the temp database
        self.server = server

    def tearDown(self):
        # close the handle before the temp dir goes: Windows will not unlink an
        # open file, and the leak shows up as a confusing cleanup error
        if self.server._con is not None:
            self.server._con.close()
            self.server._con = None
        self.env.stop()
        self.dir.cleanup()

    async def _call(self, path, method="POST", headers=None, json_body=None):
        import httpx2
        transport = httpx2.ASGITransport(app=self.server.mcp.streamable_http_app())
        async with httpx2.AsyncClient(transport=transport,
                                      base_url="http://t") as c:
            return await c.request(method, path, headers=headers or {}, json=json_body)

    async def test_health_reports_the_database(self):
        r = await self._call("/healthz", method="GET")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    async def test_admin_refuses_without_the_secret(self):
        r = await self._call("/admin/new", json_body={"names": ["a"]})
        self.assertEqual(r.status_code, 403)

    async def test_admin_refuses_a_wrong_secret(self):
        r = await self._call("/admin/new", headers={"x-admin-token": "nope"},
                             json_body={"names": ["a"]})
        self.assertEqual(r.status_code, 403)

    async def test_admin_can_start_a_run(self):
        r = await self._call("/admin/new", headers={"x-admin-token": "s3cret"},
                             json_body={"names": ["a", "b"], "monitoring": "global"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["tokens"]), 2)
        self.assertEqual(body["seats"], 2)

    async def test_reset_needs_an_explicit_confirmation(self):
        r = await self._call("/admin/reset", headers={"x-admin-token": "s3cret"},
                             json_body={})
        self.assertEqual(r.status_code, 400)

    async def test_reset_clears_every_run(self):
        await self._call("/admin/new", headers={"x-admin-token": "s3cret"},
                         json_body={"names": ["a"]})
        r = await self._call("/admin/reset", headers={"x-admin-token": "s3cret"},
                             json_body={"confirm": "reset"})
        self.assertEqual(r.status_code, 200)
        health = await self._call("/healthz", method="GET")
        self.assertEqual(health.json()["runs"], 0)


if __name__ == "__main__":
    unittest.main()
