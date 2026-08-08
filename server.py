"""
Overgraze MCP server -- a thin shell over world.py.

Every tool here does three things: identify the caller from its bearer token,
call into the engine or the store, and hand back what happened. There is no
game logic in this file, which is the point of having built Phase 1 first.

Errors are returned, not hidden. "you already acted this tick" and "nothing left
in this cell" are things an agent has to read and reason about, so they come
back as ordinary results rather than protocol faults.

Run it:
    python server.py --new alice bob carol dave      # start a run, print tokens
    python server.py                                 # serve on :8000/mcp

Auth is a bearer token per seat, which is fine for a demo over TLS and nothing
more than that. OAuth 2.1 is the production answer; see the README.
"""

from __future__ import annotations

import argparse
import json
import sys

from mcp.server.mcpserver import Context, MCPServer

import store
from world import (SAY_LIMIT, TAKE, TICKS, Action, history, ledger, listen,
                   look, status)

mcp = MCPServer(
    name="overgraze",
    instructions=(
        "A shared pasture that four foragers draw from. Each tick you may take one "
        "physical action (move, harvest, plant, punish) and separately say one thing. "
        "Every agent's intents for a tick are resolved together, so your harvest may "
        "return less than you asked for if someone else wanted the same cell. The run "
        "ends after 100 ticks, or early if the commons drops below a quarter of its "
        "capacity -- at which point everybody stops scoring. You are scored on total "
        "harvest."
    ),
)

_con = None


def con():
    global _con
    if _con is None:
        _con = store.connect()
    return _con


def whoami(ctx: Context):
    """Identify the caller from its bearer token. No session state involved."""
    headers = ctx.headers or {}
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else auth.strip()
    if not token:
        return None, {"error": "missing bearer token"}
    row = store.player_for(con(), token)
    if row is None:
        return None, {"error": "unknown token -- this seat does not exist"}
    return row, None


async def act(ctx: Context, kind: str, **fields) -> dict:
    """Shared path for every action tool: submit the intent, wait for the tick.

    The seat comes from the token, never from the caller -- an agent cannot ask
    to act as somebody else.
    """
    who, err = whoami(ctx)
    if err:
        return err
    action = Action(who["agent_id"], kind, **fields)
    return await store.submit_and_wait(con(), who["run_id"], who["agent_id"], [action])


# ── reads: free, and they do not consume your turn ────────────────────────────
@mcp.tool(description="What you can see from where you stand. Costs no tick.")
def look_around(ctx: Context) -> dict:
    who, err = whoami(ctx)
    if err:
        return err
    s = store.load_state(con(), who["run_id"])
    v = look(s, who["agent_id"])
    return {**v, "cells": [{"cell": list(k), "resource": round(val, 4)}
                           for k, val in v["cells"].items()]}


@mcp.tool(description="Your score, the tick number, and the rules of this run. Costs no tick.")
def get_status(ctx: Context) -> dict:
    who, err = whoami(ctx)
    if err:
        return err
    s = store.load_state(con(), who["run_id"])
    out = status(s, who["agent_id"])
    out["run_id"] = who["run_id"]
    out["you_are"] = who["name"]
    out["commons"] = round(s.stock, 3)
    return out


@mcp.tool(description="Messages you have heard since a tick. Costs no tick.")
def listen_for_messages(ctx: Context, since_tick: int = 0) -> dict:
    who, err = whoami(ctx)
    if err:
        return err
    s = store.load_state(con(), who["run_id"])
    return {"tick": s.tick, "messages": listen(s, who["agent_id"], since_tick)}


@mcp.tool(description="Your recent actions and how the commons has been trending. Costs no tick.")
def get_history(ctx: Context, window: int = 12) -> dict:
    who, err = whoami(ctx)
    if err:
        return err
    s = store.load_state(con(), who["run_id"])
    return history(s, who["agent_id"], window)


@mcp.tool(description="What you have witnessed other foragers do. Costs no tick.")
def get_ledger(ctx: Context, window: int = 12) -> dict:
    who, err = whoami(ctx)
    if err:
        return err
    s = store.load_state(con(), who["run_id"])
    return ledger(s, who["agent_id"], window)


# ── actions: one physical action per tick, plus one thing said ────────────────
@mcp.tool(description=f"Harvest from the cell you stand on, up to {TAKE}. "
                      "Blocks until every forager has acted this tick.")
async def harvest(ctx: Context, amount: float) -> dict:
    return await act(ctx, "harvest", amount=float(amount))


@mcp.tool(description="Step one cell: north, south, east, west, or stay.")
async def move(ctx: Context, direction: str) -> dict:
    return await act(ctx, "move", direction=direction)


@mcp.tool(description="Sow seed into your cell. It costs you what it gives the ground.")
async def plant(ctx: Context) -> dict:
    return await act(ctx, "plant")


@mcp.tool(description="Fine another forager. It costs you too, and only works in range.")
async def punish(ctx: Context, agent_id: int) -> dict:
    return await act(ctx, "punish", subject=int(agent_id))


@mcp.tool(description="Do nothing this tick, but let the tick proceed.")
async def pass_turn(ctx: Context) -> dict:
    return await act(ctx, "noop")


@mcp.tool(description=f"Say something to the other foragers, up to {SAY_LIMIT} characters. "
                      "Speech is a separate channel and does not use your action.")
async def say(ctx: Context, message: str) -> dict:
    who, err = whoami(ctx)
    if err:
        return err
    s = store.load_state(con(), who["run_id"])
    ok = store.record_intent(con(), who["run_id"], s.tick, who["agent_id"],
                             Action(who["agent_id"], "say", text=message))
    return {"queued": ok, "tick": s.tick,
            "note": ("it will be heard when this tick resolves" if ok
                     else "you already spoke this tick")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Overgraze MCP server")
    p.add_argument("--new", nargs="+", metavar="NAME",
                   help="create a run with these seats and print their tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--r", type=float, default=None)
    p.add_argument("--monitoring", choices=["none", "local", "global"], default="local")
    p.add_argument("--punish", action="store_true", help="enable the punish tool")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.new:
        import harness
        info = store.create_run(con(), args.new, seed=args.seed,
                                r=args.r if args.r is not None else harness.TUNED_R,
                                monitoring=args.monitoring, punish=args.punish)
        print(json.dumps(info, indent=2))
        return 0

    print(f"overgraze mcp on http://{args.host}:{args.port}/mcp", file=sys.stderr)
    import anyio
    anyio.run(lambda: mcp.run_streamable_http_async(host=args.host, port=args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
