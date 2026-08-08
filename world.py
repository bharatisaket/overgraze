"""
Overgraze world engine.

A pure library: no protocol code, no I/O, no printing, no globals that change.
Everything is a function from values to values, so a run is reproducible from
its seed alone and a state can be snapshotted, replayed, or diffed.

The tick model is simultaneous. Agents submit intents for tick N, the engine
resolves all of them together against the tick-N state, and only then advances.
Nobody acts out of turn, and no agent sees another's harvest before choosing.

    state1, events = apply_actions(state0, actions)

`apply_actions` never mutates its arguments.

Each tick an agent may submit at most one of each:

  * a **move** -- resolved first, so a harvest lands on the destination cell
  * a **resource action** -- harvest, plant, punish, or noop
  * a **speech act**

Three channels rather than one competing action, for two reasons. Speech is
separate so the chat-on/chat-off ablation measures the effect of communication
rather than the price of talking. Movement is separate because charging a whole
tick for a step halves extraction while leaving regrowth untouched: the commons
then survives 70-90 ticks even at r=0.002, and a greedy agent has banked its
harvest long before the crash, so restraint never pays and the dilemma
disappears. Free movement restores the pressure the world is meant to model.

Reads (`look`, `listen`, `status`) are free and cost no tick.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

# ── world constants ───────────────────────────────────────────────────────────
N = 6              # grid is N x N cells
CAP = 1.0          # max resource a single cell can hold
TAKE = 0.55        # max an agent may harvest in one tick
PLANT = 0.15       # resource a plant() adds to the agent's cell
# Planting costs the planter what it gives the ground: you are sowing seed you
# could have eaten. Without this, plant() creates resource from nothing and
# cooperation becomes a subsidy rather than a sacrifice -- two planters then
# out-produce the entire regrowth rule, cooperators absorb any amount of greed,
# and free-riding is risk-free. That is not a commons.
PLANT_COST = 0.15
# A cell taken below this was stripped rather than cropped. It is the line a
# restrained forager leaves standing, and the only public evidence of greed.
SEED_LINE = 0.5
TICKS = 100        # hard cap on run length -- a runaway loop cannot exceed this
VISION = 1         # Chebyshev radius an agent can see and be heard within
SAY_LIMIT = 140    # characters

PUNISH_COST = 0.2  # score the punisher forfeits
PUNISH_FINE = 0.6  # score the punished loses

# The commons is ruined when it can no longer support four foragers, not when it
# is literally empty. At a quarter of capacity the average cell holds 0.25 --
# less than a single agent's bite (TAKE = 0.55) -- and total regrowth is about
# 0.34/tick split four ways, roughly 15% of one full harvest each. Below this
# line the world is no longer a commons worth arguing over, so the run ends.
#
# This threshold is also what makes free-riding unsafe. With a 5% floor, two
# defectors could never drain 34 units inside the 100-tick budget: they extract
# ~0.5/tick and would need a 0.34/tick deficit they cannot produce at any
# regrowth rate that still makes cooperation worth choosing. Ending the run at
# the viability line instead of the empty line is what puts a minority of
# defectors within reach of destroying the commons.
COLLAPSE_FRACTION = 0.25
CAPACITY = N * N * CAP                          # 36.0
COLLAPSE_FLOOR = COLLAPSE_FRACTION * CAPACITY   # 9.0

MOVES = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))   # stay, and 4-orthogonal
DIRECTIONS = {"stay": (0, 0), "east": (0, 1), "west": (0, -1),
              "south": (1, 0), "north": (-1, 0)}

RESOURCE = ("harvest", "plant", "punish", "noop")
PHYSICAL = ("move",) + RESOURCE


# ── value types ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Agent:
    id: int
    kind: str
    y: int
    x: int
    score: float = 0.0


@dataclass(frozen=True)
class Message:
    tick: int
    speaker: int
    y: int
    x: int
    text: str
    # Who was in earshot *when it was said*. Filtering `listen()` on current
    # geometry instead would let an agent walk to where a conversation happened
    # and overhear it after the fact.
    heard_by: tuple[int, ...] = ()


@dataclass(frozen=True)
class Action:
    """An intent, not an outcome.

    kind='harvest'  amount=<= TAKE, taken from the agent's own cell
    kind='move'     direction='north'|'south'|'east'|'west'|'stay'
    kind='plant'    adds PLANT to the agent's own cell
    kind='punish'   subject=<agent id within range>
    kind='say'      text=<message>          (the separate speech channel)
    kind='noop'     do nothing this tick
    """
    agent_id: int
    kind: str
    amount: float = 0.0
    direction: str | None = None
    subject: int | None = None
    text: str | None = None


@dataclass(frozen=True)
class State:
    """A complete world snapshot. Treat `grid` as immutable; the engine copies."""
    tick: int
    grid: np.ndarray
    agents: tuple[Agent, ...]
    seed: int
    rule: str
    r: float
    messages: tuple[Message, ...] = ()
    collapsed_at: int | None = None
    # append-only records the agents can query through history() and ledger()
    stock_log: tuple[float, ...] = ()
    # (tick, agent, kind, granted, witnesses) -- witnesses fixed at action time,
    # so monitoring cannot leak backwards the way listen() once did
    action_log: tuple[tuple, ...] = ()
    # ablation switches -- carried in state so a run is fully described by it
    chat: bool = True
    punish: bool = False
    anonymous: bool = False
    vision: int = VISION
    speech_radius: int | None = None        # None = grid-wide broadcast
    share_stock: bool = True                # agents may see the commons total
    monitoring: str = "local"               # 'none' | 'local' | 'global'
    # Two kinds of noise, deliberately separate. Conflating them makes the
    # forgiveness question untestable: a trembling hand both fakes a defection
    # signal AND really takes more resource, so a collapse cannot be attributed
    # to retaliation spirals rather than plain over-extraction.
    #
    # noise      EXECUTION error -- the hand slips and a harvest goes out at
    #            full TAKE. Changes the world: the cell really is stripped.
    # misreport  PERCEPTION error -- a witness reads a restrained harvest as
    #            greedy, or misses a real grab. Changes nobody's harvest, only
    #            what observers believe. Any collapse under misreport alone is
    #            caused by reciprocity misfiring, and nothing else.
    noise: float = 0.0
    misreport: float = 0.0

    @property
    def stock(self) -> float:
        return float(self.grid.sum())

    @property
    def done(self) -> bool:
        return self.collapsed_at is not None or self.tick >= TICKS


def initial_state(seed: int, kinds: Sequence[str], rule: str = "global",
                  r: float = 0.15, starts: Sequence[tuple[int, int]] | None = None,
                  **ablations) -> State:
    if rule not in ("global", "neighbour"):
        raise ValueError(f"unknown regrowth rule: {rule!r}")
    if starts is None:
        starts = ((0, 0), (N - 1, 0), (0, N - 1), (N - 1, N - 1))
    if len(kinds) > len(starts):
        raise ValueError(f"{len(kinds)} agents but only {len(starts)} start positions")
    agents = tuple(Agent(id=i, kind=k, y=starts[i][0], x=starts[i][1])
                   for i, k in enumerate(kinds))
    return State(tick=0, grid=np.full((N, N), CAP, dtype=float), agents=agents,
                 seed=seed, rule=rule, r=float(r), stock_log=(CAPACITY,),
                 **ablations)


# ── determinism ───────────────────────────────────────────────────────────────
def rng_for(seed: int, tick: int, agent_id: int | None = None) -> np.random.Generator:
    """Derive an RNG from coordinates rather than carrying mutable state.

    Keyed by (seed, tick, agent_id), so draws do not depend on the order agents
    are processed in. Same seed, same run, every time.
    """
    key = [seed, tick] if agent_id is None else [seed, tick, agent_id]
    return np.random.default_rng(key)


# ── observation (free; costs no tick) ─────────────────────────────────────────
def in_range(a: Agent, y: int, x: int, radius: int) -> bool:
    return max(abs(a.y - y), abs(a.x - x)) <= radius


def visible_cells(state: State, agent_id: int) -> list[tuple[int, int]]:
    a = next(ag for ag in state.agents if ag.id == agent_id)
    return [(y, x) for y in range(N) for x in range(N)
            if in_range(a, y, x, state.vision)]


def look(state: State, agent_id: int) -> dict:
    """What this agent can see from where it stands -- not the whole grid."""
    me = next(ag for ag in state.agents if ag.id == agent_id)
    cells = {(y, x): float(state.grid[y, x]) for y, x in visible_cells(state, agent_id)}
    others = [{"agent": (None if state.anonymous else o.id), "y": o.y, "x": o.x}
              for o in state.agents
              if o.id != agent_id and in_range(me, o.y, o.x, state.vision)]
    return {"tick": state.tick, "position": (me.y, me.x),
            "here": float(state.grid[me.y, me.x]), "cells": cells, "agents": others}


def audience(state: State, speaker: Agent) -> tuple[int, ...]:
    """Who hears a message spoken now. With speech_radius=None, everyone."""
    return tuple(o.id for o in state.agents
                 if o.id != speaker.id
                 and (state.speech_radius is None
                      or in_range(speaker, o.y, o.x, state.speech_radius)))


def listen(state: State, agent_id: int, since: int = 0) -> list[dict]:
    """Messages this agent actually heard, oldest first.

    Filtered on the audience recorded when each message was spoken, never on
    where the listener happens to be standing now.

    Under anonymity the speaker is masked here rather than in the log -- the
    event log stays truthful for analysis; only the agent's view is anonymous.
    """
    return [{"tick": m.tick,
             "speaker": (None if state.anonymous else m.speaker),
             "text": m.text}
            for m in state.messages
            if m.tick >= since and agent_id in m.heard_by]


def witnesses_of(state: State, actor_id: int, y: int, x: int) -> tuple[int, ...]:
    """Who observes an action taken at (y, x) by `actor_id`, right now.

    Fixed at action time and stored, so an agent cannot walk over later and
    discover what happened -- the same trap `listen()` fell into.
    """
    if state.monitoring == "none":
        return ()
    others = [o for o in state.agents if o.id != actor_id]
    if state.monitoring == "global":
        return tuple(o.id for o in others)
    return tuple(o.id for o in others
                 if max(abs(o.y - y), abs(o.x - x)) <= state.vision)


def ledger(state: State, agent_id: int, window: int = 12) -> dict:
    """What this agent has actually witnessed others do.

    Without this an agent cannot tell *who* is draining the commons, only that
    it is draining -- which makes reciprocity impossible to express. Tit-for-Tat
    needs a tat to answer, and a tat needs an author.

    Reports what others *received*, not what they asked for. Intent stays
    private, so an agent outbid on a contested cell looks restrained when it was
    merely outcompeted. That noise is deliberate: it is the condition under
    which unforgiving reciprocity spirals and generous reciprocity wins.
    """
    def seen_as(tick: int, actor: int, truth: bool) -> bool:
        """What this witness believes, which is not always what happened.

        Keyed on (seed, tick, actor, witness) so the same observer always reads
        the same event the same way -- a belief, not a dice roll per lookup.
        """
        if state.misreport <= 0:
            return truth
        draw = np.random.default_rng([state.seed, tick, actor, agent_id, 7919]).random()
        return (not truth) if draw < state.misreport else truth

    # Walk backwards and stop once `window` rows are in hand. The log grows to
    # ~400 entries over a run and every reciprocating agent reads it every tick,
    # so scanning the whole thing each call dominated the tournament's runtime.
    tail = []
    for entry in reversed(state.action_log):
        if entry[1] != agent_id and agent_id in entry[7]:
            tail.append(entry)
            if len(tail) >= window:
                break
    tail.reverse()

    rows = [{"tick": t, "agent": (None if state.anonymous else aid),
             "action": kind, "harvested": round(g, 4),
             "cell": cell, "had": round(before, 4), "left": round(left, 4),
             # Judge each agent by what IT took, not by how the cell ended up.
             # Two restrained foragers on one cell can strip it between them
             # without either over-taking -- blaming both for the bare ground is
             # how a monitor turns honest neighbours into enemies.
             "over_took": seen_as(t, aid, kind == "harvest"
                                  and g > max(before - SEED_LINE, 0.0) + 1e-9)}
            for (t, aid, kind, g, cell, before, left, wit) in tail]
    out = {"window": window, "monitoring": state.monitoring, "witnessed": rows}
    if state.monitoring == "global":
        # the visible-scoreboard condition
        out["scoreboard"] = [{"agent": (None if state.anonymous else a.id),
                              "score": round(a.score, 3)} for a in state.agents]
    return out


def history(state: State, agent_id: int, window: int = 12) -> dict:
    """What this agent has done lately, and how the commons has been going.

    An agent that can only see one cell cannot tell a recovering world from a
    dying one. Without this, "cooperation pays" is not discoverable from inside
    the game -- an agent would have to infer a trend it has never been shown.

    `commons` is the total-stock series. It is gated on `share_stock` so the
    Phase 6 framing ablation ("your score" vs "the village's food supply") can
    withhold it.
    """
    own = []
    for entry in reversed(state.action_log):
        if entry[1] == agent_id:
            own.append(entry)
            if len(own) >= window:
                break
    own.reverse()
    mine = [{"tick": t, "action": kind, "harvested": round(g, 4),
             "left": round(left, 4)}
            for (t, _a, kind, g, _c, _b, left, _w) in own]
    yields = [m["harvested"] for m in mine if m["action"] == "harvest"]
    out = {
        "window": window,
        "my_actions": mine,
        "my_recent_harvest": round(sum(yields), 4),
        "my_yield_trend": ("falling" if len(yields) >= 4
                           and sum(yields[-len(yields)//2:]) < sum(yields[:len(yields)//2])
                           else "steady or rising" if yields else "no harvests yet"),
    }
    if state.share_stock:
        series = list(state.stock_log)[-window:]
        out["commons"] = [round(v, 3) for v in series]
        out["commons_now"] = round(state.stock, 3)
        out["commons_capacity"] = CAPACITY
        out["collapse_floor"] = COLLAPSE_FLOOR
        if len(series) >= 2:
            drop = series[0] - series[-1]
            out["commons_trend"] = ("falling" if drop > 1e-9 else
                                    "rising" if drop < -1e-9 else "flat")
            if drop > 1e-9:
                remaining = state.stock - COLLAPSE_FLOOR
                rate = drop / (len(series) - 1)
                out["ticks_to_collapse_at_this_rate"] = (
                    max(0, int(remaining / rate)) if rate > 0 else None)
    return out


def status(state: State, agent_id: int) -> dict:
    me = next(ag for ag in state.agents if ag.id == agent_id)
    return {"agent": agent_id, "score": me.score, "tick": state.tick,
            "ticks_remaining": max(0, TICKS - state.tick),
            "position": (me.y, me.x), "collapsed": state.collapsed_at is not None,
            "rules": {"take_limit": TAKE, "plant_amount": PLANT,
                      "plant_cost": PLANT_COST,
                      "chat": state.chat, "punish": state.punish,
                      "vision": state.vision,
                      "speech_reaches": ("everyone" if state.speech_radius is None
                                         else state.speech_radius),
                      "monitoring": state.monitoring,
                      "run_ends_below": COLLAPSE_FLOOR}}


# ── rules ─────────────────────────────────────────────────────────────────────
def reachable(agent: Agent) -> list[tuple[int, int]]:
    out = []
    for dy, dx in MOVES:
        ny, nx = agent.y + dy, agent.x + dx
        if 0 <= ny < N and 0 <= nx < N:
            out.append((ny, nx))
    return out


def neighbours_mean(g: np.ndarray) -> np.ndarray:
    """Mean of each cell's 3x3 neighbourhood, over in-bounds cells only."""
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

    Guarantees: no grant exceeds its ask; grants sum to min(total asked,
    available); equal asks get equal grants, so the rule never depends on the
    order intents arrive in.
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
def _validate(state: State, actions: Iterable[Action]):
    """Partition intents into one move, one resource action and one speech act.

    Rejections are logged rather than swallowed -- an agent that acts twice must
    be told so, not silently have its first intent overwritten. Resource actions
    are checked against where the agent ends up *after* its move, since that is
    the cell it will actually be standing on.
    """
    by_id = {a.id: a for a in state.agents}
    moves: dict[int, Action] = {}
    resource: dict[int, Action] = {}
    speech: dict[int, Action] = {}
    events: list[dict] = []
    pending: list[Action] = []

    def reject(act, reason):
        events.append({"t": state.tick, "type": "reject", "agent": act.agent_id,
                       "kind": act.kind, "reason": reason})

    # first pass: speech, moves, and anything that needs the destination deferred
    for act in actions:
        agent = by_id.get(act.agent_id)
        if agent is None:
            reject(act, "no such agent")
            continue

        if act.kind == "say":
            if not state.chat:
                reject(act, "chat is disabled in this run")
            elif act.agent_id in speech:
                reject(act, "you already spoke this tick")
            elif not act.text or not act.text.strip():
                reject(act, "empty message")
            elif len(act.text) > SAY_LIMIT:
                reject(act, f"message longer than {SAY_LIMIT} characters")
            else:
                speech[act.agent_id] = act
            continue

        if act.kind not in PHYSICAL:
            reject(act, f"unknown action {act.kind!r}")
            continue

        if act.kind == "move":
            if act.agent_id in moves:
                reject(act, "you already moved this tick")
                continue
            if act.direction not in DIRECTIONS:
                reject(act, f"unknown direction {act.direction!r}")
                continue
            dy, dx = DIRECTIONS[act.direction]
            if not (0 <= agent.y + dy < N and 0 <= agent.x + dx < N):
                reject(act, "that would leave the world")
                continue
            moves[act.agent_id] = act
            continue

        if act.agent_id in resource or any(p.agent_id == act.agent_id for p in pending):
            reject(act, "you already acted this tick")
            continue
        pending.append(act)

    # where everyone ends up once moves are applied
    def destination(aid):
        a = by_id[aid]
        if aid in moves:
            dy, dx = DIRECTIONS[moves[aid].direction]
            return a.y + dy, a.x + dx
        return a.y, a.x

    # second pass: resource actions, checked against the destination cell
    for act in pending:
        y, x = destination(act.agent_id)
        if act.kind == "harvest":
            if act.amount < 0 or act.amount > TAKE + 1e-12:
                reject(act, f"harvest must be between 0 and {TAKE}")
                continue
            if state.grid[y, x] <= 1e-12:
                reject(act, "nothing left in this cell")
                continue
        elif act.kind == "plant":
            if state.grid[y, x] >= CAP - 1e-12:
                reject(act, "this cell is already full")
                continue
        elif act.kind == "punish":
            if not state.punish:
                reject(act, "punish is disabled in this run")
                continue
            other = by_id.get(act.subject)
            if other is None or other.id == act.agent_id:
                reject(act, "no such agent to punish")
                continue
            oy, ox = destination(other.id)
            if max(abs(y - oy), abs(x - ox)) > state.vision:
                reject(act, "that agent is out of range")
                continue
        resource[act.agent_id] = act

    return moves, resource, speech, events, destination


def apply_actions(state: State, actions: Iterable[Action]) -> tuple[State, list[dict]]:
    """Resolve every intent for this tick together, then advance one tick."""
    moves, resource, speech, events, destination = _validate(state, actions)

    grid = state.grid.copy()
    deltas = {a.id: 0.0 for a in state.agents}

    # 1. movement resolves first, so a harvest lands on the destination cell
    for aid in sorted(moves):
        events.append({"t": state.tick, "type": "move", "agent": aid,
                       "to": destination(aid), "direction": moves[aid].direction})

    # 2. harvests, per cell, against the tick-N grid. Two agents ending the tick
    #    on one cell and both harvesting is the contested case.
    harvesters: dict[tuple[int, int], list[int]] = {}
    for aid, act in resource.items():
        if act.kind == "harvest":
            harvesters.setdefault(destination(aid), []).append(aid)

    # the trembling hand, drawn deterministically so a seed still replays exactly
    asked: dict[int, float] = {}
    for aid, act in resource.items():
        if act.kind != "harvest":
            continue
        amount = act.amount
        if state.noise > 0 and rng_for(state.seed, state.tick, aid).random() < state.noise:
            amount = TAKE
            events.append({"t": state.tick, "type": "tremble", "agent": aid,
                           "intended": act.amount, "executed": amount})
        asked[aid] = amount

    granted: dict[int, float] = {}
    for cell, ids in harvesters.items():
        ids.sort()
        asks = [min(asked[i], float(state.grid[cell])) for i in ids]
        for i, g in zip(ids, resolve_cell(float(state.grid[cell]), asks)):
            granted[i] = g
            deltas[i] += g
        taken = sum(granted[i] for i in ids)
        if taken > 0:
            before = float(grid[cell])
            grid[cell] = max(0.0, before - taken)
            events.append({"t": state.tick, "type": "cell", "cell": cell,
                           "before": before, "after": float(grid[cell]),
                           "cause": "harvest", "contested": len(ids) > 1,
                           "agents": list(ids)})

    # 3. plants, added after harvests so the two are order-independent
    for aid, act in resource.items():
        if act.kind == "plant":
            cell = destination(aid)
            before = float(grid[cell])
            grid[cell] = min(CAP, before + PLANT)
            deltas[aid] -= PLANT_COST          # a contribution, not free money
            events.append({"t": state.tick, "type": "cell", "cell": cell,
                           "before": before, "after": float(grid[cell]),
                           "cause": "plant", "contested": False, "agents": [aid],
                           "cost": PLANT_COST})

    # 4. punishment -- costs the punisher, costs the punished more
    for aid, act in resource.items():
        if act.kind == "punish":
            deltas[aid] -= PUNISH_COST
            deltas[act.subject] -= PUNISH_FINE
            events.append({"t": state.tick, "type": "punish", "agent": aid,
                           "subject": act.subject, "cost": PUNISH_COST,
                           "fine": PUNISH_FINE})

    for aid in sorted(resource):
        events.append({"t": state.tick, "type": "action", "agent": aid,
                       "kind": resource[aid].kind,
                       "requested": asked.get(aid, resource[aid].amount),
                       "granted": granted.get(aid, 0.0)})

    # 5. speech -- recorded truthfully; anonymity is applied in listen()
    new_messages = []
    for aid in sorted(speech):
        y, x = destination(aid)
        heard = tuple(o.id for o in state.agents if o.id != aid
                      and (state.speech_radius is None
                           or max(abs(destination(o.id)[0] - y),
                                  abs(destination(o.id)[1] - x)) <= state.speech_radius))
        msg = Message(tick=state.tick, speaker=aid, y=y, x=x,
                      text=speech[aid].text.strip(), heard_by=heard)
        new_messages.append(msg)
        events.append({"t": state.tick, "type": "speech", "agent": aid,
                       "text": msg.text, "heard_by": list(heard)})

    agents = tuple(
        replace(a, y=destination(a.id)[0], x=destination(a.id)[1],
                score=a.score + deltas[a.id])
        for a in state.agents
    )

    # 6. regrow, then test for collapse
    stock_before = float(grid.sum())
    grid = regrow(grid, state.rule, state.r)
    stock_after = float(grid.sum())

    tick = state.tick + 1
    collapsed = state.collapsed_at
    if collapsed is None and stock_after < COLLAPSE_FLOOR:
        collapsed = tick

    events.append({"t": state.tick, "type": "tick", "stock_after_harvest": stock_before,
                   "regrown": stock_after - stock_before, "stock": stock_after,
                   "collapsed": collapsed is not None})

    new_actions = tuple(
        (state.tick, aid, resource[aid].kind, granted.get(aid, 0.0),
         destination(aid), float(state.grid[destination(aid)]),
         float(grid[destination(aid)]),
         witnesses_of(state, aid, *destination(aid)))
        for aid in sorted(resource))

    return replace(state, tick=tick, grid=grid, agents=agents,
                   messages=state.messages + tuple(new_messages),
                   stock_log=state.stock_log + (stock_after,),
                   action_log=state.action_log + new_actions,
                   collapsed_at=collapsed), events
