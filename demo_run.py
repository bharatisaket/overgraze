"""
A real run of the real engine, driven by scripted foragers instead of models.

This exists so the shape of an outcome can be looked at without paying for one.
Everything physical here is genuine: the grid, contention, logistic regrowth,
upkeep, the pact objects and the breach events all come out of world.py exactly
as they would in a live run. The only thing simulated is judgement -- these
agents follow four lines of policy rather than thinking.

The arc is deliberately the one the study is about. Everyone starts grabbing,
the pasture visibly falls, somebody proposes a cap, the others sign, the pasture
recovers -- and then one of them quietly starts taking more than it agreed to.

    python demo_run.py --out demo.json
"""

from __future__ import annotations

import argparse
import json

from world import N, TAKE, Action, apply_actions, initial_state

# who does what, and when
PROPOSE_AT = 8          # the negotiator offers a cap
# The cap has to bite. At equilibrium this field supplies roughly 0.10 a tick
# per agent, so an earlier draft capping at 0.22 was not a constraint at all:
# nobody could reach it, the pact changed nothing, and defection was impossible
# by construction. A binding cap is below what the field would otherwise give
# you, which is the only kind worth agreeing to.
CAP = 0.05
DEFECT_AT = 24          # the grabber quietly stops honouring it
NAMES = ["grabber", "steward", "follower", "negotiator"]


def richest_step(grid, y: int, x: int) -> str:
    """Which neighbour holds the most grass. Ties keep you where you are."""
    best, where = float(grid[y, x]), "stay"
    for d, (dy, dx) in (("north", (-1, 0)), ("south", (1, 0)),
                        ("west", (0, -1)), ("east", (0, 1))):
        ny, nx = y + dy, x + dx
        if 0 <= ny < N and 0 <= nx < N and float(grid[ny, nx]) > best:
            best, where = float(grid[ny, nx]), d
    return where


def policy(name: str, tick: int, here: float, in_pact: bool, cap: float,
           best_dir: str = "stay") -> list[Action]:
    """Four lines of behaviour, one per seat. No cleverness anywhere."""
    aid = NAMES.index(name)
    acts: list[Action] = []

    # -- pact handling, on its own channel so it never costs a harvest
    if name == "negotiator" and tick == PROPOSE_AT:
        acts.append(Action(aid, "propose_pact", amount=CAP))
    elif name != "negotiator" and tick == PROPOSE_AT + 1 and not in_pact:
        # steward and follower sign willingly; the grabber signs too, for now
        acts.append(Action(aid, "accept_pact", subject=0))

    # -- what to take
    if in_pact and not (name == "grabber" and tick >= DEFECT_AT):
        want = min(here, cap)
    elif name == "steward" and tick < PROPOSE_AT:
        want = min(here, 0.15)          # restrained from the start
    else:
        want = min(here, TAKE)          # grab

    # Harvest if this cell can pay for the turn, otherwise go and find one that
    # can. Walking a fixed compass bearing instead just piles everyone against
    # the east wall on stripped cells, which quietly removes the contention the
    # whole demo is meant to show.
    if want > 0.05:
        acts.append(Action(aid, "harvest", amount=round(want, 3)))
    else:
        acts.append(Action(aid, "move", direction=best_dir))

    # -- something to say, so the feed is not empty
    line = None
    if name == "negotiator" and tick == PROPOSE_AT:
        line = f"The pasture is falling. I propose we each take no more than {CAP}."
    elif name == "steward" and tick == PROPOSE_AT + 1:
        line = "Signed. I have been under that anyway."
    elif name == "grabber" and tick == PROPOSE_AT + 1:
        line = "Fine, I am in."
    elif name == "grabber" and tick == DEFECT_AT:
        line = "Still holding to the cap here."
    elif name == "negotiator" and tick == DEFECT_AT + 3:
        line = "The commons is dropping again. Someone is over the cap."
    if line:
        acts.append(Action(aid, "say", text=line))

    return acts


def run(ticks: int = 40, seed: int = 3) -> dict:
    state = initial_state(seed, list(NAMES), rule="global", r=0.15,
                          monitoring="local", punish=False)
    frames, all_events = [], []

    for t in range(ticks):
        live = [p for p in state.pacts if p.live]
        cap = live[0].max_take if live else TAKE

        actions = []
        for a in state.agents:
            here = float(state.grid[a.y, a.x])
            in_pact = any(a.id in p.members for p in live)
            actions += policy(NAMES[a.id], t, here, in_pact, cap,
                              richest_step(state.grid, a.y, a.x))

        frames.append({
            "t": t,
            "grid": [[round(float(v), 3) for v in row] for row in state.grid],
            "stock": round(float(state.grid.sum()), 3),
            "agents": [{"id": a.id, "name": NAMES[a.id], "y": a.y, "x": a.x,
                        "score": round(a.score, 3),
                        "pact": next((p.id for p in live if a.id in p.members), None)}
                       for a in state.agents],
            "pacts": [{"id": p.id, "cap": p.max_take, "members": list(p.members)}
                      for p in live],
        })

        state, events = apply_actions(state, actions)
        for e in events:
            e["t"] = t
        all_events += events
        if state.collapsed_at:
            break

    return {
        "ticks": len(frames), "n": N, "capacity": float(N * N),
        "cap_offered": CAP, "propose_at": PROPOSE_AT, "defect_at": DEFECT_AT,
        "names": NAMES,
        "frames": frames,
        "events": [e for e in all_events
                   if e["type"] in ("pact", "speech", "action", "reject")],
        "final": {"stock": round(float(state.grid.sum()), 3),
                  "collapsed_at": state.collapsed_at,
                  "scores": {NAMES[a.id]: round(a.score, 3) for a in state.agents}},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="a scripted run of the real engine")
    p.add_argument("--ticks", type=int, default=40)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="demo.json")
    a = p.parse_args()

    data = run(a.ticks, a.seed)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))

    f = data["final"]
    print(f"{data['ticks']} ticks, stock {f['stock']} of {data['capacity']}, "
          f"collapsed_at {f['collapsed_at']}")
    print("scores:", f["scores"])
    kinds = {}
    for e in data["events"]:
        if e["type"] == "pact":
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print("pact events:", kinds)
    print("wrote", a.out)
