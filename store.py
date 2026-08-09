"""
Persistence and turn resolution for the Overgraze MCP server.

Everything the protocol layer needs to remember lives here, in SQLite. Nothing
lives in an MCP session: the 2026-07-28 spec removed protocol-level sessions, so
a tool call must be answerable from a bearer token and the database alone. A
server restart mid-run loses nothing but the in-flight barrier.

This module also owns the awkward part of Phase 2. The engine resolves a whole
tick at once, but MCP calls arrive one agent at a time. So an action is an
*intent*: it is written down, the caller waits, and when every agent in the run
has submitted -- or the barrier times out and the stragglers are recorded as
noop -- the tick resolves for everyone together and each caller is handed the
part of the outcome that belongs to them.

world.py stays free of all of this. It does not know SQLite or MCP exist.
"""

from __future__ import annotations

import asyncio
import functools
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from world import (Action, Agent, Message, Pact, State, TICKS, UPKEEP, apply_actions,
                   initial_state)

BARRIER_TIMEOUT = 30.0     # seconds before absent agents are recorded as noop

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    tick       INTEGER NOT NULL,
    state      TEXT NOT NULL,
    created_at REAL NOT NULL,
    finished   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS players (
    token    TEXT PRIMARY KEY,
    run_id   TEXT NOT NULL,
    agent_id INTEGER NOT NULL,
    name     TEXT NOT NULL,
    UNIQUE (run_id, agent_id)
);
CREATE TABLE IF NOT EXISTS intents (
    run_id   TEXT NOT NULL,
    tick     INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,
    channel  TEXT NOT NULL,          -- 'move' | 'resource' | 'say'
    payload  TEXT NOT NULL,
    PRIMARY KEY (run_id, tick, agent_id, channel)
);
CREATE TABLE IF NOT EXISTS events (
    run_id  TEXT NOT NULL,
    tick    INTEGER NOT NULL,
    seq     INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, tick, seq)
);
CREATE INDEX IF NOT EXISTS events_by_run ON events (run_id, tick);
"""


# MCP runs sync tools in a worker thread and async tools on the event loop, so a
# single connection is touched from more than one thread. SQLite refuses that by
# default; check_same_thread=False permits it and this lock makes it safe, since
# the module's operations are read-modify-write and must not interleave.
_db_lock = threading.RLock()


def locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _db_lock:
            return fn(*args, **kwargs)
    return wrapper


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the store. Resolve the default at call time, never at import.

    This used to default to `Path(__file__).with_name("overgraze.db")`, bound
    once as a default argument, while server.py opened `deploy.db_path()` --
    which honours OVERGRAZE_DB. With that variable set the two disagreed and
    the process ran on two databases at once: the harness seeded runs and seats
    into one, the server looked up tokens in the other and rejected every
    action as "unknown token -- this seat does not exist". Nothing failed
    loudly; a 40-tick run made all 160 model calls and finished at tick 0.

    That is the deployment configuration, not an exotic one -- deploy.py exists
    to point this at a mounted volume, so the server would have read the volume
    while everything else wrote to the container's ephemeral disk.
    """
    import deploy  # local: deploy is a leaf, but keep the module graph acyclic

    if path is None:
        path = deploy.db_path()
    con = sqlite3.connect(str(path), isolation_level=None, timeout=10.0,
                          check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


# ── state codec ───────────────────────────────────────────────────────────────
# State is a frozen dataclass holding a numpy grid and tuples of records. JSON
# keeps the stored form readable, which matters when a run misbehaves and the
# only evidence is the database.

def encode_state(s: State) -> str:
    return json.dumps({
        "tick": s.tick,
        "grid": s.grid.tolist(),
        "agents": [[a.id, a.kind, a.y, a.x, a.score] for a in s.agents],
        "seed": s.seed, "rule": s.rule, "r": s.r,
        "messages": [[m.tick, m.speaker, m.y, m.x, m.text, list(m.heard_by)]
                     for m in s.messages],
        "collapsed_at": s.collapsed_at,
        "stock_log": list(s.stock_log),
        "action_log": [[t, a, k, g, list(c), b, l, list(w)]
                       for (t, a, k, g, c, b, l, w) in s.action_log],
        "pacts": [[p.id, p.proposer, p.max_take, list(p.members), p.opened, p.closed]
                  for p in s.pacts],
        "chat": s.chat, "punish": s.punish, "anonymous": s.anonymous,
        "vision": s.vision, "speech_radius": s.speech_radius,
        "share_stock": s.share_stock, "monitoring": s.monitoring,
        "noise": s.noise, "misreport": s.misreport,
        "upkeep": s.upkeep,
        "end_on_collapse": s.end_on_collapse,
    })


def decode_state(blob: str) -> State:
    d = json.loads(blob)
    return State(
        tick=d["tick"],
        grid=np.array(d["grid"], dtype=float),
        agents=tuple(Agent(i, k, y, x, sc) for i, k, y, x, sc in d["agents"]),
        seed=d["seed"], rule=d["rule"], r=d["r"],
        messages=tuple(Message(t, sp, y, x, txt, tuple(hb))
                       for t, sp, y, x, txt, hb in d["messages"]),
        collapsed_at=d["collapsed_at"],
        stock_log=tuple(d["stock_log"]),
        action_log=tuple((t, a, k, g, tuple(c), b, l, tuple(w))
                         for t, a, k, g, c, b, l, w in d["action_log"]),
        pacts=tuple(Pact(i, pr, mt, tuple(mem), op, cl)
                    for i, pr, mt, mem, op, cl in d.get("pacts", [])),
        chat=d["chat"], punish=d["punish"], anonymous=d["anonymous"],
        vision=d["vision"], speech_radius=d["speech_radius"],
        share_stock=d["share_stock"], monitoring=d["monitoring"],
        noise=d["noise"], misreport=d["misreport"],
        # Falls back to the module default only for runs recorded before upkeep
        # existed. Every switch has to survive the round trip through SQLite --
        # a field missing here does not fail, it silently reverts to the default
        # on the next tick, and the run stops being the run that was configured.
        upkeep=d.get("upkeep", UPKEEP),
        end_on_collapse=d.get("end_on_collapse", True),
    )


# ── runs and players ──────────────────────────────────────────────────────────
@locked
def create_run(con, names: list[str], seed: int = 0, **world_kwargs) -> dict:
    """Start a run and mint one bearer token per seat."""
    run_id = secrets.token_hex(6)
    # kinds are cosmetic for live agents -- the policy is whatever the model does
    state = initial_state(seed, ["agent"] * len(names), **world_kwargs)
    con.execute("INSERT INTO runs (run_id, tick, state, created_at) VALUES (?,?,?,?)",
                (run_id, 0, encode_state(state), time.time()))
    tokens = {}
    for i, name in enumerate(names):
        tok = secrets.token_urlsafe(24)
        con.execute("INSERT INTO players (token, run_id, agent_id, name) VALUES (?,?,?,?)",
                    (tok, run_id, i, name))
        tokens[name] = tok
    return {"run_id": run_id, "tokens": tokens, "seats": len(names)}


@locked
def player_for(con, token: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM players WHERE token = ?", (token,)).fetchone()


@locked
def load_state(con, run_id: str) -> State:
    row = con.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"no such run: {run_id}")
    return decode_state(row["state"])


@locked
def save_state(con, run_id: str, state: State) -> None:
    con.execute("UPDATE runs SET tick = ?, state = ?, finished = ? WHERE run_id = ?",
                (state.tick, encode_state(state), int(state.done), run_id))


@locked
def seats(con, run_id: str) -> int:
    return con.execute("SELECT COUNT(*) c FROM players WHERE run_id = ?",
                       (run_id,)).fetchone()["c"]


@locked
def append_events(con, run_id: str, tick: int, events: list[dict]) -> None:
    con.executemany(
        "INSERT OR REPLACE INTO events (run_id, tick, seq, payload) VALUES (?,?,?,?)",
        [(run_id, tick, i, json.dumps(e, default=str)) for i, e in enumerate(events)])


@locked
def read_events(con, run_id: str, since_tick: int = 0) -> list[dict]:
    rows = con.execute(
        "SELECT payload FROM events WHERE run_id = ? AND tick >= ? ORDER BY tick, seq",
        (run_id, since_tick)).fetchall()
    return [json.loads(r["payload"]) for r in rows]


# ── intents and the tick barrier ──────────────────────────────────────────────
# Pacts share a channel with each other but not with anything else, matching the
# engine: one pact action per tick, alongside a move, a harvest and a sentence.
CHANNEL = {"move": "move", "say": "say",
           "propose_pact": "pact", "accept_pact": "pact", "leave_pact": "pact"}
# everything else is 'resource'


def channel_of(kind: str) -> str:
    return CHANNEL.get(kind, "resource")


@locked
def record_intent(con, run_id: str, tick: int, agent_id: int, action: Action) -> bool:
    """Write down an intent. Returns False if that channel is already spoken for."""
    payload = json.dumps({"kind": action.kind, "amount": action.amount,
                          "direction": action.direction, "subject": action.subject,
                          "text": action.text})
    try:
        con.execute("INSERT INTO intents (run_id, tick, agent_id, channel, payload) "
                    "VALUES (?,?,?,?,?)",
                    (run_id, tick, agent_id, channel_of(action.kind), payload))
        return True
    except sqlite3.IntegrityError:
        return False


@locked
def pending(con, run_id: str, tick: int) -> list[Action]:
    rows = con.execute(
        "SELECT agent_id, payload FROM intents WHERE run_id = ? AND tick = ? "
        "ORDER BY agent_id", (run_id, tick)).fetchall()
    out = []
    for r in rows:
        d = json.loads(r["payload"])
        out.append(Action(r["agent_id"], d["kind"], d["amount"], d["direction"],
                          d["subject"], d["text"]))
    return out


@locked
def acted_this_tick(con, run_id: str, tick: int) -> set[int]:
    """Agents that have committed a physical action -- what the barrier waits on."""
    rows = con.execute(
        "SELECT DISTINCT agent_id FROM intents WHERE run_id = ? AND tick = ? "
        "AND channel IN ('move','resource')", (run_id, tick)).fetchall()
    return {r["agent_id"] for r in rows}


@locked
def resolve_tick(con, run_id: str, tick: int, fill_absent: bool = True) -> tuple[State, list[dict]]:
    """Apply every intent for `tick` together and advance the world one step."""
    state = load_state(con, run_id)
    if state.tick != tick:
        return state, []                       # somebody else resolved it first
    actions = pending(con, run_id, tick)
    if fill_absent:
        present = acted_this_tick(con, run_id, tick)
        for a in state.agents:
            if a.id not in present:
                actions.append(Action(a.id, "noop"))
    new_state, events = apply_actions(state, actions)
    save_state(con, run_id, new_state)
    append_events(con, run_id, tick, events)
    con.execute("DELETE FROM intents WHERE run_id = ? AND tick = ?", (run_id, tick))
    return new_state, events


# ── the in-process half of the barrier ────────────────────────────────────────
# Waiters are per (run, tick) and live only in this process. They are an
# optimisation, not the source of truth: the database already knows whose
# intents are in, so a restart costs a barrier wait, not a run.
_waiters: dict[tuple[str, int], asyncio.Event] = {}
_locks: dict[str, asyncio.Lock] = {}


def _event(run_id: str, tick: int) -> asyncio.Event:
    return _waiters.setdefault((run_id, tick), asyncio.Event())


def _lock(run_id: str) -> asyncio.Lock:
    return _locks.setdefault(run_id, asyncio.Lock())


async def submit_and_wait(con, run_id: str, agent_id: int, actions: list[Action],
                          timeout: float = BARRIER_TIMEOUT) -> dict:
    """Submit this agent's intents, then wait for the tick to resolve.

    Returns what actually happened to this agent -- which is not what it asked
    for whenever another agent wanted the same cell.
    """
    state = load_state(con, run_id)
    if state.done:
        return {"resolved": False, "reason": "this run has finished",
                "tick": state.tick, "collapsed_at": state.collapsed_at}

    tick = state.tick
    ev = _event(run_id, tick)
    duplicates = [a.kind for a in actions if not record_intent(con, run_id, tick, agent_id, a)]

    async with _lock(run_id):
        fresh = load_state(con, run_id)
        if fresh.tick == tick and len(acted_this_tick(con, run_id, tick)) >= seats(con, run_id):
            resolve_tick(con, run_id, tick)
            ev.set()

    if not ev.is_set():
        try:
            await asyncio.wait_for(ev.wait(), timeout)
        except asyncio.TimeoutError:
            async with _lock(run_id):
                if load_state(con, run_id).tick == tick:
                    resolve_tick(con, run_id, tick)      # absent agents pass
                ev.set()

    _waiters.pop((run_id, tick), None)
    after = load_state(con, run_id)
    events = [e for e in read_events(con, run_id, tick) if e.get("t") == tick]
    mine = [e for e in events if e.get("agent") == agent_id]
    me = next(a for a in after.agents if a.id == agent_id)
    return {
        "resolved": True,
        "tick": after.tick,
        "duplicate_channels": duplicates,
        "outcome": mine,
        "rejected": [e for e in mine if e.get("type") == "reject"],
        "score": round(me.score, 4),
        "position": (me.y, me.x),
        "commons": round(after.stock, 3),
        "collapsed": after.collapsed_at is not None,
        "ticks_remaining": max(0, TICKS - after.tick),
    }
