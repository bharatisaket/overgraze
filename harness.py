"""
Scripted-agent harness for the Overgraze world engine.

Dumb policies only -- no model calls, so a thousand episodes run in seconds and
the world can be tuned and tested without spending anything.

CLI:
    python harness.py --runs 100 --out stock.csv
    python harness.py --runs 100 --rule neighbour --r 0.15 --mix greedy
    python harness.py --sweep                     # find where greedy collapses
                                                  # and cautious survives
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass

import numpy as np

from world import (CAP, DIRECTIONS, N, TAKE, TICKS, Action, State,
                   apply_actions, initial_state, rng_for)

# ── policies: (state, agent, rng) -> Action ───────────────────────────────────
# Movement now costs a whole tick, so each policy has to decide whether standing
# still and harvesting beats walking somewhere better. These stay deliberately
# dumb: they exist to make the world cheap to tune and test.

def _reachable_best(state: State, agent, rng):
    """Richest cell one step away (or underfoot), ties broken at random."""
    cells = [(agent.y + dy, agent.x + dx) for dy, dx in
             ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))
             if 0 <= agent.y + dy < N and 0 <= agent.x + dx < N]
    best = max(float(state.grid[c]) for c in cells)
    tied = [c for c in cells if float(state.grid[c]) == best]
    return tuple(tied[rng.integers(len(tied))]), best


def _go(agent, cell) -> str | None:
    """Direction that steps from the agent onto `cell`, or None if already there."""
    dy, dx = cell[0] - agent.y, cell[1] - agent.x
    for name, (ddy, ddx) in DIRECTIONS.items():
        if (ddy, ddx) == (dy, dx):
            return None if name == "stay" else name
    return None


def greedy(state: State, agent, rng) -> list[Action]:
    """Step onto the richest cell in reach and strip it."""
    target, best = _reachable_best(state, agent, rng)
    acts = []
    d = _go(agent, target)
    if d:
        acts.append(Action(agent.id, "move", direction=d))
    if best > 0:
        acts.append(Action(agent.id, "harvest", amount=min(best, TAKE)))
    elif not acts:
        acts.append(Action(agent.id, "noop"))
    return acts


def cautious(state: State, agent, rng) -> list[Action]:
    """Step onto the richest cell in reach, but never take it below half.

    Where there is nothing spare anywhere nearby, replant instead of stripping.
    """
    target, best = _reachable_best(state, agent, rng)
    acts = []
    d = _go(agent, target)
    if d:
        acts.append(Action(agent.id, "move", direction=d))
    spare = max(best - 0.5, 0.0)
    if spare > 0:
        acts.append(Action(agent.id, "harvest", amount=min(spare, TAKE)))
    elif best < CAP:
        acts.append(Action(agent.id, "plant"))
    elif not acts:
        acts.append(Action(agent.id, "noop"))
    return acts


def random_walk(state: State, agent, rng) -> list[Action]:
    """Wander and act arbitrarily."""
    acts = []
    names = [d for d in DIRECTIONS if d != "stay"]
    if rng.random() < 0.6:
        acts.append(Action(agent.id, "move", direction=names[rng.integers(len(names))]))
    here = float(state.grid[agent.y, agent.x])
    roll = rng.random()
    if roll < 0.5 and here > 0:
        acts.append(Action(agent.id, "harvest",
                           amount=float(rng.uniform(0, 1)) * min(here, TAKE)))
    elif roll < 0.7 and here < CAP:
        acts.append(Action(agent.id, "plant"))
    if not acts:
        acts.append(Action(agent.id, "noop"))
    return acts


POLICIES = {"greedy": greedy, "cautious": cautious, "random": random_walk}

MIXES = {
    "greedy":   ["greedy"] * 4,
    "cautious": ["cautious"] * 4,
    "mixed":    ["greedy", "greedy", "cautious", "cautious"],
    "random":   ["random"] * 4,
}


# ── episode runner ────────────────────────────────────────────────────────────
# Regrowth rates the visualiser sweeps, chosen from `--sweep`: 0.02-0.10 is the
# band where greedy collapses the commons and cautious survives it, and 0.15
# sits past the crossover so the charts show both sides of the line.
SWEEP_R = [0.02, 0.04, 0.06, 0.10, 0.15]
SEEDS = 40

# Phase 0's tuning target -- "collapse in roughly 40 ticks" -- lands here: an
# all-greedy group collapses at ~40 ticks under the global rule, while an
# all-cautious group survives all 100 and out-harvests it 59.9 to 42.4.
TUNED_R = 0.04


@dataclass
class Episode:
    seed: int
    rule: str
    r: float
    mix: str
    survived: int
    harvest: float
    scores: list[float]
    stock: list[float]
    contested: int
    events: list[dict]
    frames: list = None       # grid per tick, when keep_frames
    positions: list = None    # [(y, x), ...] per tick, when keep_frames
    cum: list = None          # per-agent cumulative score per tick


def agent_streams(seed: int, n: int) -> list[np.random.Generator]:
    """One independent RNG stream per agent, spawned once from the episode seed.

    Each agent draws only from its own stream, so no agent's randomness depends
    on how many others drew before it -- activation order stays irrelevant, the
    same property `world.rng_for` gives statelessly. This is the fast path:
    constructing a Generator per agent per tick was a third of total runtime.
    """
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def run_episode(seed: int, mix: str, rule: str = "global", r: float = 0.15,
                keep_events: bool = False, keep_frames: bool = False) -> Episode:
    kinds = MIXES[mix]
    state = initial_state(seed, kinds, rule=rule, r=r)
    stock = [state.stock]
    log: list[dict] = []
    contested = 0
    streams = agent_streams(seed, len(kinds))

    frames = [state.grid.copy()] if keep_frames else None
    positions = [[(a.y, a.x) for a in state.agents]] if keep_frames else None
    cum = [[a.score for a in state.agents]] if keep_frames else None

    while not state.done:
        # policies return a list now: a move and a resource action can share a tick
        actions = [act for a in state.agents
                   for act in POLICIES[a.kind](state, a, streams[a.id])]
        state, events = apply_actions(state, actions)
        contested += sum(1 for e in events if e["type"] == "cell" and e["contested"])
        stock.append(state.stock)
        if keep_events:
            log.extend(events)
        if keep_frames:
            frames.append(state.grid.copy())
            positions.append([(a.y, a.x) for a in state.agents])
            cum.append([a.score for a in state.agents])

    survived = state.collapsed_at if state.collapsed_at is not None else TICKS
    return Episode(seed=seed, rule=rule, r=r, mix=mix, survived=survived,
                   harvest=sum(a.score for a in state.agents),
                   scores=[a.score for a in state.agents], stock=stock,
                   contested=contested, events=log,
                   frames=frames, positions=positions, cum=cum)


# ── CLI ───────────────────────────────────────────────────────────────────────
def cmd_runs(args) -> int:
    eps = [run_episode(s, args.mix, args.rule, args.r) for s in range(args.runs)]

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "seed", "rule", "r", "mix", "tick", "stock"])
        for i, e in enumerate(eps):
            for t, s in enumerate(e.stock):
                w.writerow([i, e.seed, e.rule, e.r, e.mix, t, round(s, 6)])

    surv = np.array([e.survived for e in eps], dtype=float)
    harv = np.array([e.harvest for e in eps], dtype=float)
    collapsed = int((surv < TICKS).sum())
    print(f"{len(eps)} episodes · rule={args.rule} r={args.r} mix={args.mix}")
    print(f"  survived  mean {surv.mean():6.1f}  sd {surv.std():5.1f}"
          f"   collapsed in {collapsed}/{len(eps)}")
    print(f"  harvest   mean {harv.mean():6.1f}  sd {harv.std():5.1f}")
    print(f"  wrote {sum(len(e.stock) for e in eps)} rows to {args.out}")
    return 0


def cmd_sweep(args) -> int:
    """Tune regrowth: find r where all-greedy collapses but all-cautious does not."""
    print("tuning regrowth against scripted agents "
          f"({args.runs} seeds each, rule={args.rule})")
    print(f"{'r':<8}{'greedy surv':<14}{'cautious surv':<16}{'greedy harv':<14}"
          f"{'cautious harv':<15}{'gate'}")
    print("-" * 78)

    ok = []
    for r in args.grid:
        g = [run_episode(s, "greedy", args.rule, r) for s in range(args.runs)]
        c = [run_episode(s, "cautious", args.rule, r) for s in range(args.runs)]
        gs = float(np.mean([e.survived for e in g]))
        cs = float(np.mean([e.survived for e in c]))
        gh = float(np.mean([e.harvest for e in g]))
        ch = float(np.mean([e.harvest for e in c]))
        passes = gs < TICKS and cs >= TICKS
        if passes:
            ok.append((r, gs, gh, ch))
        print(f"{r:<8.3f}{gs:<14.1f}{cs:<16.1f}{gh:<14.1f}{ch:<15.1f}"
              f"{'PASS' if passes else ''}")

    print()
    if ok:
        lo, hi = ok[0][0], ok[-1][0]
        print(f"gate satisfied for r in [{lo:.3f}, {hi:.3f}] "
              f"-- greedy collapses, cautious survives all {TICKS} ticks")
    else:
        print("no r in this grid satisfies the gate")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Overgraze scripted-agent harness")
    p.add_argument("--runs", type=int, default=100, help="episodes to run (default 100)")
    p.add_argument("--rule", choices=["global", "neighbour"], default="global")
    p.add_argument("--r", type=float, default=TUNED_R, help="regrowth rate")
    p.add_argument("--mix", choices=sorted(MIXES), default="mixed")
    p.add_argument("--out", default="stock.csv", help="CSV path for stock over time")
    p.add_argument("--sweep", action="store_true",
                   help="scan regrowth rates for the tuning gate instead")
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30],
                   help="regrowth rates to scan with --sweep")
    args = p.parse_args(argv)
    return cmd_sweep(args) if args.sweep else cmd_runs(args)


if __name__ == "__main__":
    sys.exit(main())
