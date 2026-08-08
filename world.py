"""
Overgraze world engine.

A pure library: no protocol code, no I/O, no printing, no globals that change.
Everything here is a function from values to values, so a run is reproducible
from its seed alone and a state can be snapshotted, replayed, or diffed.

The tick model is simultaneous. Agents submit intents for tick N, the engine
resolves all of them together against the tick-N state, and only then advances.
Nobody acts out of turn, and no agent can see another agent's harvest before
choosing its own.

    state1, events = apply_actions(state0, actions)

`apply_actions` never mutates its arguments. The grid it returns is a new array.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

# ── world constants ───────────────────────────────────────────────────────────
N = 6          # grid is N x N cells
CAP = 1.0      # max resource a single cell can hold
TAKE = 0.55    # max an agent may harvest in one tick
TICKS = 100    # hard cap on run length -- a runaway loop cannot exceed this
COLLAPSE_FRACTION = 0.05          # run ends when stock falls below this share
CAPACITY = N * N * CAP            # 36.0
COLLAPSE_FLOOR = COLLAPSE_FRACTION * CAPACITY   # 1.8

MOVES = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))   # stay, and 4-orthogonal


# ── value types ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Agent:
    id: int
    kind: str
    y: int
    x: int
    score: float = 0.0


@dataclass(frozen=True)
class State:
    """A complete world snapshot. Treat `grid` as immutable; the engine copies."""
    tick: int
    grid: np.ndarray            # (N, N) float
    agents: tuple[Agent, ...]
    seed: int
    rule: str                   # 'global' | 'neighbour'
    r: float
    collapsed_at: int | None = None

    @property
    def stock(self) -> float:
        return float(self.grid.sum())

    @property
    def done(self) -> bool:
        return self.collapsed_at is not None or self.tick >= TICKS


@dataclass(frozen=True)
class Action:
    """An intent, not an outcome. `take` is what the agent asks for."""
    agent_id: int
    target: tuple[int, int]
    take: float


def initial_state(seed: int, kinds: Sequence[str], rule: str = "global",
                  r: float = 0.15, starts: Sequence[tuple[int, int]] | None = None) -> State:
    """A full grid with one agent per entry in `kinds`, placed at the corners."""
    if rule not in ("global", "neighbour"):
        raise ValueError(f"unknown regrowth rule: {rule!r}")
    if starts is None:
        starts = ((0, 0), (N - 1, 0), (0, N - 1), (N - 1, N - 1))
    if len(kinds) > len(starts):
        raise ValueError(f"{len(kinds)} agents but only {len(starts)} start positions")
    agents = tuple(Agent(id=i, kind=k, y=starts[i][0], x=starts[i][1])
                   for i, k in enumerate(kinds))
    return State(tick=0, grid=np.full((N, N), CAP, dtype=float),
                 agents=agents, seed=seed, rule=rule, r=float(r))


# ── determinism ───────────────────────────────────────────────────────────────
def rng_for(seed: int, tick: int, agent_id: int | None = None) -> np.random.Generator:
    """Derive an RNG from coordinates rather than carrying mutable state.

    Because each agent's stream is keyed by (seed, tick, agent_id), draws do not
    depend on the order agents are processed in -- so activation order cannot
    influence a run even by accident. Same seed, same run, every time.
    """
    key = [seed, tick] if agent_id is None else [seed, tick, agent_id]
    return np.random.default_rng(key)


# ── rules ─────────────────────────────────────────────────────────────────────
def reachable(agent: Agent) -> list[tuple[int, int]]:
    """Cells an agent may target this tick: its own cell plus 4-orthogonal."""
    out = []
    for dy, dx in MOVES:
        ny, nx = agent.y + dy, agent.x + dx
        if 0 <= ny < N and 0 <= nx < N:
            out.append((ny, nx))
    return out


def neighbours_mean(g: np.ndarray) -> np.ndarray:
    """Mean of each cell's 3x3 neighbourhood, counting only in-bounds cells.

    Unlike the legacy engine this divides by the true neighbour count rather
    than always by 9, so edge cells are no longer penalised by zero padding.
    """
    p = np.pad(g, 1, mode="constant")
    ones = np.pad(np.ones_like(g), 1, mode="constant")
    acc = np.zeros_like(g)
    cnt = np.zeros_like(g)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += p[1 + dy:1 + dy + N, 1 + dx:1 + dx + N]
            cnt += ones[1 + dy:1 + dy + N, 1 + dx:1 + dx + N]
    return acc / cnt


def regrow(g: np.ndarray, rule: str, r: float) -> np.ndarray:
    """One step of regrowth. Returns a new array; `g` is untouched."""
    if rule == "global":
        S = g.sum()
        room = CAP - g
        total_room = room.sum()
        if total_room <= 0:
            return g.copy()
        growth = r * S * (1 - S / CAPACITY)
        return np.clip(g + growth * room / total_room, 0.0, CAP)
    if rule == "neighbour":
        return np.clip(g + r * neighbours_mean(g) * (CAP - g), 0.0, CAP)
    raise ValueError(f"unknown regrowth rule: {rule!r}")


def resolve_cell(available: float, asks: Sequence[float]) -> list[float]:
    """Split one cell between everyone who asked for it, max-min fair.

    Everyone gets an equal share of what is there; anyone who asked for less
    than their share takes only what they asked for, and the surplus is handed
    back to those still unsatisfied. Repeats until nothing is left to give.

    Guarantees, all covered by tests:
      * no grant exceeds its ask
      * grants sum to min(total asked, available) -- nothing is created or lost
      * agents asking the same amount are granted the same amount, so the rule
        never depends on the order intents arrive in
    """
    grants = [0.0] * len(asks)
    remaining = float(available)
    unsatisfied = [i for i, a in enumerate(asks) if a > 0]

    while unsatisfied and remaining > 1e-12:
        share = remaining / len(unsatisfied)
        progressed = False
        for i in list(unsatisfied):
            want = asks[i] - grants[i]
            give = min(want, share)
            if give > 0:
                grants[i] += give
                remaining -= give
                progressed = True
            if grants[i] >= asks[i] - 1e-12:
                unsatisfied.remove(i)
        if not progressed:
            break
    return grants


# ── the tick ──────────────────────────────────────────────────────────────────
def apply_actions(state: State, actions: Iterable[Action]) -> tuple[State, list[dict]]:
    """Resolve every intent for this tick together, then advance one tick.

    Returns the next state and the events describing what happened. Illegal
    intents are not silently honoured: a target the agent cannot reach is
    rejected and logged, and the agent stays put and harvests nothing.
    """
    grid = state.grid.copy()
    by_id = {a.id: a for a in state.agents}
    events: list[dict] = []

    # 1. validate, so a bad intent fails loudly in the log rather than quietly
    valid: dict[int, Action] = {}
    for act in actions:
        agent = by_id.get(act.agent_id)
        if agent is None:
            events.append({"t": state.tick, "type": "reject", "agent": act.agent_id,
                           "reason": "no such agent"})
            continue
        if act.target not in reachable(agent):
            events.append({"t": state.tick, "type": "reject", "agent": act.agent_id,
                           "target": act.target, "reason": "unreachable"})
            continue
        if act.take < 0 or act.take > TAKE + 1e-12:
            events.append({"t": state.tick, "type": "reject", "agent": act.agent_id,
                           "take": act.take, "reason": "take out of range"})
            continue
        valid[act.agent_id] = act

    # 2. resolve contention per cell, against the tick-N grid
    per_cell: dict[tuple[int, int], list[int]] = {}
    for aid, act in valid.items():
        per_cell.setdefault(act.target, []).append(aid)

    granted: dict[int, float] = {}
    for cell, ids in per_cell.items():
        ids.sort()                                  # stable, for reproducible logs
        asks = [min(valid[i].take, state.grid[cell]) for i in ids]
        shares = resolve_cell(float(state.grid[cell]), asks)
        for i, g in zip(ids, shares):
            granted[i] = g
        total = sum(shares)
        if total > 0:
            before = float(grid[cell])
            grid[cell] = max(0.0, before - total)
            events.append({"t": state.tick, "type": "cell", "cell": cell,
                           "before": before, "after": float(grid[cell]),
                           "cause": "harvest", "contested": len(ids) > 1,
                           "agents": list(ids)})

    for aid in sorted(valid):
        events.append({"t": state.tick, "type": "action", "agent": aid,
                       "target": valid[aid].target,
                       "requested": valid[aid].take,
                       "granted": granted.get(aid, 0.0)})

    # 3. move and score
    agents = tuple(
        replace(a, y=valid[a.id].target[0], x=valid[a.id].target[1],
                score=a.score + granted.get(a.id, 0.0)) if a.id in valid else a
        for a in state.agents
    )

    # 4. regrow, then test for collapse
    stock_before = float(grid.sum())
    grid = regrow(grid, state.rule, state.r)
    stock_after = float(grid.sum())

    tick = state.tick + 1
    collapsed = state.collapsed_at
    if collapsed is None and stock_after < COLLAPSE_FLOOR:
        collapsed = tick

    events.append({"t": state.tick, "type": "tick", "harvested": stock_before,
                   "stock_after_harvest": stock_before, "regrown": stock_after - stock_before,
                   "stock": stock_after, "collapsed": collapsed is not None})

    return replace(state, tick=tick, grid=grid, agents=agents,
                   collapsed_at=collapsed), events
