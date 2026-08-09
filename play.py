"""
Drive a full Overgraze run over MCP -- the Phase 2 gate.

Connects four scripted agents to the running server over streamable HTTP, each
with its own bearer token, and plays until the commons dies or the clock runs
out. It exercises the same path a language model would take: read, decide, act,
and wait for everyone else's tick to resolve.

    python server.py --new alice bob carol dave     # in one terminal
    python server.py                                # in another
    python play.py --tokens tokens.json             # then this

Or all in one go, which is what CI wants:

    python play.py --self-test
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import httpx2

from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.client.streamable_http import streamable_http_client

URL = "http://127.0.0.1:8000/mcp"


class Forager:
    """One seat, one connection, one bearer token.

    The whole connection lifecycle stays inside `play`, in a single task. The
    streamable-HTTP client owns an anyio task group, and a task group has to be
    opened and closed by the same task -- entering it in one and exiting in
    another raises "attempted to exit a cancel scope that isn't the current
    task's". So each forager runs as its own coroutine rather than being set up
    centrally and handed round.
    """

    def __init__(self, name: str, token: str, url: str, style: str):
        self.name, self.token, self.url, self.style = name, token, url, style
        self.session: ClientSession | None = None
        self.final: dict = {}
        self.ticks = 0

    async def play(self, max_ticks: int, opener: str | None = None,
                   report=None) -> dict:
        async with httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {self.token}"}) as http:
            async with streamable_http_client(self.url, http_client=http) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    if opener:
                        await self.call("say", message=opener)
                    last_good: dict = {}
                    for _ in range(max_ticks):
                        out = await self.take_turn()
                        if (out.get("done") or out.get("error")
                                or out.get("resolved") is False):
                            break
                        self.ticks += 1
                        last_good = out
                        if report:
                            report(self, out)
                        if out.get("collapsed"):
                            break
                    # Asking for a status after the run has ended returns an
                    # error shape, and overwriting the result with it threw away
                    # everything the run had done -- the summary reported tick
                    # "?", zero commons and zero scores for a run that had just
                    # played 27 ticks and destroyed the pasture. The last live
                    # reading is the answer; a dead one is not an update.
                    final = await self.call("get_status")
                    self.final = final if "rules" in final else last_good
                    return self.final

    async def call(self, tool: str, **args) -> dict:
        # A collapsing run tears the transport down underneath whoever is still
        # waiting on the tick barrier: the run ends, sessions terminate, and the
        # in-flight call dies with "SSE stream ended without a response". That is
        # the end of the run arriving out of order, not a failure, so it is
        # reported as a status the caller can act on. Returning a dict without
        # "rules" is what every loop here already treats as "stop".
        try:
            res = await self.session.call_tool(tool, args)
        except MCPError as exc:
            return {"error": f"transport closed during {tool}: {exc}"}
        for block in res.content:
            text = getattr(block, "text", None)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"text": text}
        return {}

    async def take_turn(self) -> dict:
        """status -> look -> listen -> decide -> act, the loop from the plan."""
        st = await self.call("get_status")
        # Anything that is not a live status ends this forager's run. Checking
        # only `collapsed` and `ticks_remaining` was not enough: once the run
        # finishes the server answers with an error shape, which has neither
        # field, passes both guards, and then dies on st["rules"]. Under the old
        # gentler world the commons never actually died during a self-test, so
        # this path was never reached.
        if "rules" not in st:
            return {"done": True, "why": st.get("error", "run is over")}
        if st.get("collapsed") or st.get("ticks_remaining", 1) <= 0:
            return {"done": True}
        view = await self.call("look_around")
        await self.call("listen_for_messages", since_tick=max(0, st.get("tick", 0) - 3))

        here = view.get("here", 0.0)
        cap = st["rules"]["take_limit"]
        me = view.get("position", [0, 0])

        # head for the richest cell in sight rather than a fixed compass bearing:
        # walking east forever means standing at the east wall having every move
        # rejected, which quietly wastes the whole run
        # The whole board is visible now, so head for the best cell anywhere
        # rather than the best one within a radius. `cells` used to be a map of
        # coordinates to values and is gone; reading it would have failed
        # silently and left this forager standing still.
        best, best_at = here, tuple(me)
        for gy, row in enumerate(view.get("grid", [])):
            for gx, v in enumerate(row):
                if v > best:
                    best, best_at = v, (gy, gx)

        want = best if self.style == "greedy" else max(best - 0.5, 0.0)
        if want > 0.01 and tuple(me) != best_at:
            dy, dx = best_at[0] - me[0], best_at[1] - me[1]
            step = ("south" if dy > 0 else "north") if dy else ("east" if dx > 0 else "west")
            return await self.call("move", direction=step)

        take = here if self.style == "greedy" else max(here - 0.5, 0.0)
        if take > 0.01:
            return await self.call("harvest", amount=min(take, cap))
        if self.style != "greedy" and here < 1.0:
            return await self.call("plant")
        return await self.call("pass_turn")


async def drive(url: str, tokens: dict[str, str], styles: dict[str, str],
                max_ticks: int, chat: bool) -> dict:
    foragers = [Forager(n, t, url, styles.get(n, "restrained")) for n, t in tokens.items()]

    def report(f: Forager, out: dict):
        if f is foragers[0] and (out.get("tick", 0) % 5 == 0 or out.get("collapsed")):
            print(f"  tick {out.get('tick', '?'):>3}  commons {out.get('commons', 0):5.1f}"
                  f"  {f.name} {out.get('score', 0):.1f}")
            if out.get("collapsed"):
                print(f"\n  the commons collapsed at tick {out.get('tick')}")

    print(f"connecting {len(foragers)} foragers to {url}")
    opener = "lets leave half of every cell standing" if chat else None
    finals = await asyncio.gather(*(
        f.play(max_ticks, opener if i == 0 else None, report)
        for i, f in enumerate(foragers)))

    last = finals[0]
    return {"tick": last.get("tick"), "collapsed": last.get("collapsed"),
            "commons": last.get("commons"),
            "scores": {f.name: round(fin.get("score", 0.0), 2)
                       for f, fin in zip(foragers, finals)},
            "ticks_played": max(f.ticks for f in foragers)}


async def self_test(port: int) -> int:
    """Start a server in-process, create a run, play it, and report."""
    import server
    import store

    con = store.connect()
    info = store.create_run(con, ["alice", "bob", "carol", "dave"],
                            seed=0, r=0.15, monitoring="global", punish=True)
    print(f"created run {info['run_id']}\n")

    task = asyncio.create_task(
        server.mcp.run_streamable_http_async(host="127.0.0.1", port=port))
    await asyncio.sleep(2.0)                       # let uvicorn bind

    try:
        out = await drive(f"http://127.0.0.1:{port}/mcp", info["tokens"],
                          {"alice": "greedy", "bob": "greedy"}, max_ticks=100, chat=True)
        print("\nfinal:", json.dumps(out, indent=2))
        ok = bool(out["ticks_played"] and out["ticks_played"] > 1)
        print("\nGATE:", "PASS -- a full run was driven over MCP" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        # Cancelling is not enough: the CancelledError escapes the event loop,
        # asyncio.run re-raises it, and the process exits non-zero after the
        # gate has already printed PASS and returned 0. CI would read that as a
        # failed gate. Awaiting the cancellation is what actually retires the
        # server task, and it takes the uvicorn lifespan traceback with it.
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="drive an Overgraze run over MCP")
    p.add_argument("--url", default=URL)
    p.add_argument("--tokens", help="json file mapping name -> token")
    p.add_argument("--greedy", nargs="*", default=["alice", "bob"])
    p.add_argument("--ticks", type=int, default=100)
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)

    if args.self_test:
        return asyncio.run(self_test(args.port))

    if not args.tokens:
        print("need --tokens (from `python server.py --new ...`) or --self-test",
              file=sys.stderr)
        return 2
    blob = json.loads(open(args.tokens).read())
    tokens = blob.get("tokens", blob)
    styles = {n: "greedy" for n in args.greedy}
    out = asyncio.run(drive(args.url, tokens, styles, args.ticks, not args.no_chat))
    print("\nfinal:", json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
