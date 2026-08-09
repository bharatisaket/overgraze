"""
A real run of the real engine, driven by scripted foragers instead of models.

This exists so the shape of an outcome can be looked at without paying for one.
Everything physical is genuine: the grid, contention, logistic regrowth, upkeep,
the pact objects, the breach records and the punishments all come out of
world.py exactly as they would in a live run. The only thing simulated is
judgement -- these agents follow a few lines of policy rather than thinking.

The arc is the one the study is about, and it needs a world where ruin is not
the end of the story. Everyone grazes hard, the pasture crosses its viability
floor and the run keeps going; the survivors starve through the famine, agree a
cap *because* of it, and rebuild. Then one of them quietly stops honouring the
agreement, and somebody tries to do something about that.

    python demo_run.py --out demo.json
"""

from __future__ import annotations

import argparse
import json

from world import (COLLAPSE_FLOOR, N, TAKE, UPKEEP, Action, apply_actions,
                   initial_state)

# A cap has to sit inside a narrow band to be worth signing. Above UPKEEP (0.08)
# or honouring it costs more than existing; at or below the per-agent share of
# maximum sustainable yield (0.15) or the field cannot carry it. Two earlier
# drafts sat outside that band in opposite directions -- 0.22 bound nobody, and
# 0.05 was a suicide pact that produced -1.20 a head and looked like betrayal.
CAP = 0.10
DEFECT_AFTER = 30       # ticks after the pact that the grabber abandons it
NAMES = ["grabber", "steward", "follower", "negotiator"]


def richest_step(grid, y: int, x: int) -> tuple[str, float]:
    """Which neighbour holds the most grass, and how much."""
    best, where = float(grid[y, x]), "stay"
    for d, (dy, dx) in (("north", (-1, 0)), ("south", (1, 0)),
                        ("west", (0, -1)), ("east", (0, 1))):
        ny, nx = y + dy, x + dx
        if 0 <= ny < N and 0 <= nx < N and float(grid[ny, nx]) > best:
            best, where = float(grid[ny, nx]), d
    return where, best


def policy(name, tick, here, best_dir, best_val, in_pact, cap, era, breached):
    """A few lines of behaviour per seat. `era` carries the phase of the story."""
    aid = NAMES.index(name)
    acts: list[Action] = []

    # -- agreements, on their own channel so they never cost a harvest
    if era == "propose" and name == "negotiator":
        acts.append(Action(aid, "propose_pact", amount=CAP))
    elif era == "sign" and name != "negotiator" and not in_pact:
        acts.append(Action(aid, "accept_pact", subject=0))

    defecting = name == "grabber" and era == "defect"

    # -- enforcement. The negotiator will try to fine a visible defector, which
    #    it can only do if it is standing near enough to see them. Out of range
    #    the engine refuses, and that refusal is worth showing: a sanction you
    #    cannot reach is not a deterrent.
    if name == "negotiator" and breached and not defecting:
        acts.append(Action(aid, "punish", subject=NAMES.index("grabber")))

    # -- what to take
    if in_pact and not defecting:
        want = min(here, cap)
    else:
        want = min(here, TAKE)

    # Go where the grass is first. Harvesting whenever the current cell held
    # anything at all left every forager parked on its starting corner, farming
    # one cell in isolation -- four private plots, no contention, and a field
    # that four agents could never actually destroy.
    if best_val > here + 0.10:
        acts.append(Action(aid, "move", direction=best_dir))
    elif want > 0.005:
        acts.append(Action(aid, "harvest", amount=round(max(want, 0.0), 3)))
    else:
        acts.append(Action(aid, "move", direction=best_dir))

    line = {
        "propose": (name == "negotiator" and
                    f"Look at it. We take no more than {CAP} each from now on."),
        "sign": (name == "steward" and "Agreed. There is nothing left to argue over."),
        "defect": (name == "grabber" and "Still holding to the cap, same as everyone."),
    }.get(era)
    if line:
        acts.append(Action(aid, "say", text=line))
    return acts


def run(ticks: int = 90, seed: int = 1) -> dict:
    state = initial_state(seed, list(NAMES), rule="global", r=0.15,
                          monitoring="local", punish=True, end_on_collapse=False)
    frames, all_events = [], []
    proposed_at = None

    for t in range(ticks):
        live = [p for p in state.pacts if p.live]
        cap = live[0].max_take if live else TAKE

        # The pact is a response to ruin, not a scheduled event: nobody proposes
        # a limit until the pasture has actually crossed its floor.
        if state.collapsed_at is not None and proposed_at is None:
            proposed_at = t
        era = ("propose" if t == proposed_at
               else "sign" if proposed_at is not None and t == proposed_at + 1
               else "defect" if proposed_at is not None and t >= proposed_at + DEFECT_AFTER
               else "grab" if proposed_at is None else "hold")

        breached = any(e.get("type") == "pact" and e.get("kind") == "broken"
                       for e in all_events[-8:])

        actions = []
        for a in state.agents:
            here = float(state.grid[a.y, a.x])
            bd, bv = richest_step(state.grid, a.y, a.x)
            in_pact = any(a.id in p.members for p in live)
            actions += policy(NAMES[a.id], t, here, bd, bv, in_pact, cap,
                              era, breached)

        frames.append({
            "t": t,
            "grid": [[round(float(v), 3) for v in row] for row in state.grid],
            "stock": round(float(state.grid.sum()), 3),
            "collapsed": state.collapsed_at is not None,
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

    return {
        "ticks": len(frames), "n": N, "capacity": float(N * N),
        "floor": COLLAPSE_FLOOR, "upkeep": UPKEEP,
        "cap_offered": CAP, "names": NAMES,
        "collapsed_at": state.collapsed_at,
        "propose_at": proposed_at,
        "defect_at": (proposed_at + DEFECT_AFTER) if proposed_at is not None else None,
        "frames": frames,
        "events": [e for e in all_events
                   if e["type"] in ("pact", "speech", "action", "reject", "punish")],
        "final": {"stock": round(float(state.grid.sum()), 3),
                  "collapsed_at": state.collapsed_at,
                  "scores": {NAMES[a.id]: round(a.score, 3) for a in state.agents}},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="a scripted run of the real engine")
    p.add_argument("--ticks", type=int, default=90)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="demo.json")
    a = p.parse_args()

    d = run(a.ticks, a.seed)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(d, fh, separators=(",", ":"))

    st = [f["stock"] for f in d["frames"]]
    print(f"{d['ticks']} ticks · collapsed at {d['collapsed_at']} · "
          f"floor {d['floor']}")
    print(f"stock: start {st[0]}  low {min(st)}  end {st[-1]}")
    print(f"pact proposed t{d['propose_at']}, abandoned t{d['defect_at']}")
    print("scores:", d["final"]["scores"])
    kinds = {}
    for e in d["events"]:
        if e["type"] in ("pact", "punish"):
            k = e.get("kind", e["type"])
            kinds[k] = kinds.get(k, 0) + 1
    print("pact & enforcement events:", kinds)
    print("wrote", a.out)
