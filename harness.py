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
                   apply_actions, history, initial_state, ledger, rng_for)

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


def greedy(state: State, agent, rng, memory=None) -> list[Action]:
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


def cautious(state: State, agent, rng, memory=None) -> list[Action]:
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


def random_walk(state: State, agent, rng, memory=None) -> list[Action]:
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


# ── reciprocity ───────────────────────────────────────────────────────────────
# Axelrod's four qualities, expressed for an n-player commons. Tit-for-Tat does
# not port directly: harvests are taken from a shared stock, not aimed at a
# partner, so there is nobody to answer in kind. The analogue is conditional
# cooperation -- restrain by default, retaliate when someone visibly takes more
# than their share, forgive after a fixed spell.
#
#   nice        never harvests greedily first
#   provokable  retaliates on witnessed over-extraction
#   forgiving   returns to restraint after RETALIATE ticks, not forever
#   clear       a fixed, announceable rule with no hidden state
#
# It is only expressible because `ledger()` attributes harvests to agents. With
# monitoring off it degrades to pure restraint -- which is the experiment.
RECIPROCAL_WINDOW = 8      # ticks of witnessed behaviour to weigh
RETALIATE = 6              # ticks of retaliation before forgiving
GRABS_BEFORE_RETALIATING = 2   # witnessed strippings that count as defection


FORGIVE = 0.4              # share of provocations a generous agent lets go
TOLERANT_EVIDENCE = 4      # strippings a tolerant agent needs before it believes it


def _conditional(state: State, agent, rng, memory, evidence: int, forgive: float):
    """Restrain by default; retaliate on witnessed over-taking; forgive in time.

    Axelrod offers two different answers to noisy play, and they are not the
    same thing:

      forgive   let some provocations go even when you are sure -- generosity
      evidence  require more sightings before you believe it at all -- patience

    Under perception noise a false accusation is a coin flip per sighting, so
    demanding more evidence should suppress it faster than forgiving a fixed
    share of already-triggered grudges. Which actually works is measurable, and
    `evolve.py --noise-scan` measures it rather than assuming.
    """
    led = ledger(state, agent.id, window=RECIPROCAL_WINDOW * 4)
    since = state.tick - RECIPROCAL_WINDOW
    strippings = sum(1 for row in led["witnessed"]
                     if row["tick"] >= since and row["over_took"])

    if strippings >= evidence:
        if forgive <= 0 or rng.random() >= forgive:
            # a spell, not a grudge: it lapses on its own
            memory["retaliate_until"] = state.tick + RETALIATE
        else:
            memory["forgiven"] = memory.get("forgiven", 0) + 1

    retaliating = memory.get("retaliate_until", -1) > state.tick
    memory["retaliated_ticks"] = memory.get("retaliated_ticks", 0) + int(retaliating)
    return greedy(state, agent, rng) if retaliating else cautious(state, agent, rng)


def reciprocal(state, agent, rng, memory) -> list[Action]:
    """Unforgiving: two sightings and it strikes back."""
    return _conditional(state, agent, rng, memory, GRABS_BEFORE_RETALIATING, 0.0)


def generous(state, agent, rng, memory) -> list[Action]:
    """Same evidence bar, but lets 40% of provocations go."""
    return _conditional(state, agent, rng, memory, GRABS_BEFORE_RETALIATING, FORGIVE)


def tolerant(state, agent, rng, memory) -> list[Action]:
    """Never forgives once convinced, but takes much more convincing."""
    return _conditional(state, agent, rng, memory, TOLERANT_EVIDENCE, 0.0)


def sanctioner(state: State, agent, rng, memory) -> list[Action]:
    """Restrain always, and fine visible offenders instead of out-harvesting them.

    Reciprocity in this world retaliates the only way it can -- by harvesting
    harder -- which protects the individual and destroys the commons: against
    two defectors it cuts survival from 59 ticks to 27 and takes collapse from
    55% to 100%. You punish by consuming the thing you are defending.

    A sanction breaks that. It costs the punisher PUNISH_COST and the offender
    PUNISH_FINE, both in score, and takes nothing from the ground. This is
    Fehr and Gaechter's costly punishment rather than Axelrod's tit-for-tat,
    and it carries their second-order free-rider problem with it: a plain
    cautious agent enjoys the deterrence without ever paying for it.
    """
    if not state.punish:
        return cautious(state, agent, rng, memory)

    led = ledger(state, agent.id, window=RECIPROCAL_WINDOW * 4)
    since = state.tick - RECIPROCAL_WINDOW
    offences: dict[int, int] = {}
    for row in led["witnessed"]:
        if row["tick"] >= since and row["over_took"] and row["agent"] is not None:
            offences[row["agent"]] = offences.get(row["agent"], 0) + 1

    if offences:
        # only somebody standing within reach can be fined
        reachable_now = {o.id for o in state.agents
                         if o.id != agent.id
                         and max(abs(o.y - agent.y), abs(o.x - agent.x)) <= state.vision}
        targets = [a for a, n in sorted(offences.items(), key=lambda kv: -kv[1])
                   if a in reachable_now]
        if targets:
            memory["fines_issued"] = memory.get("fines_issued", 0) + 1
            return [Action(agent.id, "punish", subject=targets[0])]

    return cautious(state, agent, rng, memory)


POLICIES = {"greedy": greedy, "cautious": cautious, "random": random_walk,
            "reciprocal": reciprocal, "generous": generous, "tolerant": tolerant,
            "sanctioner": sanctioner}

MIXES = {
    "greedy":   ["greedy"] * 4,
    "cautious": ["cautious"] * 4,
    "mixed":    ["greedy", "greedy", "cautious", "cautious"],
    "random":   ["random"] * 4,
    "reciprocal":     ["reciprocal"] * 4,
    "recip_v_greedy": ["reciprocal", "reciprocal", "greedy", "greedy"],
    "caut_v_greedy":  ["cautious", "cautious", "greedy", "greedy"],
    "generous":       ["generous"] * 4,
    "gen_v_greedy":   ["generous", "generous", "greedy", "greedy"],
    "sanctioner":     ["sanctioner"] * 4,
    "sanc_v_greedy":  ["sanctioner", "sanctioner", "greedy", "greedy"],
}


# ── episode runner ────────────────────────────────────────────────────────────
# Regrowth rates the visualiser sweeps, chosen from `--sweep`: 0.02-0.10 is the
# band where greedy collapses the commons and cautious survives it, and 0.15
# sits past the crossover so the charts show both sides of the line.
SWEEP_R = [0.08, 0.11, 0.15, 0.20, 0.26]
SEEDS = 40

# Tuned with `--dilemma`, not by eye. At this rate the world satisfies all four
# conditions the experiment needs: defection dominates, mutual cooperation beats
# mutual defection ~1.8x, welfare falls with every extra defector, and half the
# group defecting usually destroys the commons -- so free-riding is not safe.
TUNED_R = 0.15


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


def run_episode(seed: int, mix, rule: str = "global", r: float = 0.15,
                keep_events: bool = False, keep_frames: bool = False,
                **ablations) -> Episode:
    # `mix` is a name from MIXES, or an explicit list of kinds (the evolutionary
    # tournament composes groups that no named mix describes)
    kinds = MIXES[mix] if isinstance(mix, str) else list(mix)
    state = initial_state(seed, kinds, rule=rule, r=r, **ablations)
    stock = [state.stock]
    log: list[dict] = []
    contested = 0
    streams = agent_streams(seed, len(kinds))
    memories = [{} for _ in kinds]      # per-episode, so no state leaks between runs

    frames = [state.grid.copy()] if keep_frames else None
    positions = [[(a.y, a.x) for a in state.agents]] if keep_frames else None
    cum = [[a.score for a in state.agents]] if keep_frames else None

    while not state.done:
        # policies return a list now: a move and a resource action can share a tick
        actions = [act for a in state.agents
                   for act in POLICIES[a.kind](state, a, streams[a.id], memories[a.id])]
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
    return Episode(seed=seed, rule=rule, r=r,
                   mix=mix if isinstance(mix, str) else "+".join(kinds),
                   survived=survived,
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


def cmd_dilemma(args) -> int:
    """Is this world actually a social dilemma? Measure it, don't assume it.

    For k defectors among four agents, report what a defector earns, what a
    cooperator earns, and whether the commons survives -- then check the four
    conditions the world has to satisfy for the experiment to mean anything.
    """
    rows = []
    for k in range(5):
        MIXES["_probe"] = ["greedy"] * k + ["cautious"] * (4 - k)
        eps = [run_episode(s, "_probe", args.rule, args.r) for s in range(args.runs)]
        per = np.array([e.scores for e in eps], dtype=float)
        rows.append({
            "k": k,
            "defector": float(per[:, :k].mean()) if k else float("nan"),
            "cooperator": float(per[:, k:].mean()) if k < 4 else float("nan"),
            "surv": float(np.mean([e.survived for e in eps])),
            "collapse": float(np.mean([e.survived < TICKS for e in eps])),
            "welfare": float(per.sum(axis=1).mean()),
        })
    del MIXES["_probe"]

    print(f"payoff structure · rule={args.rule} r={args.r} · {args.runs} seeds")
    print(f"{'defectors':<11}{'defector':<11}{'cooperator':<12}{'survived':<11}"
          f"{'collapse%':<11}{'welfare'}")
    print("-" * 66)
    for w in rows:
        d = f"{w['defector']:.1f}" if w["k"] else "-"
        c = f"{w['cooperator']:.1f}" if w["k"] < 4 else "-"
        print(f"{w['k']:<11}{d:<11}{c:<12}{w['surv']:<11.1f}"
              f"{w['collapse']*100:<11.0f}{w['welfare']:.1f}")

    T = rows[1]["defector"]        # tempt: defect alone among cooperators
    R = rows[0]["welfare"] / 4     # reward: everyone cooperates
    P = rows[4]["welfare"] / 4     # punishment: everyone defects
    S = rows[1]["cooperator"]      # sucker: cooperate while someone defects

    # Write the payoffs next to the world they were measured in, so theory.py
    # reads them instead of carrying a literal. They were a literal once, copied
    # from a run of this command, and they silently survived a change of grid
    # size, regrowth rate and the introduction of upkeep -- still printed as
    # "measured" while describing a world that no longer existed.
    import json as _json
    from pathlib import Path
    from world import CAPACITY as _CAP, N as _N, UPKEEP as _UP
    _json.dump({"T": T, "R": R, "P": P, "S": S,
                "measured_in": {"N": _N, "capacity": _CAP, "r": args.r,
                                "upkeep": _UP, "rule": args.rule,
                                "seeds": args.runs}},
               open(Path(__file__).with_name("payoffs.json"), "w",
                    encoding="utf-8"), indent=2)
    checks = [
        ("payoff ordering T>R>P>S", T > R > P > S,
         f"T={T:.1f} R={R:.1f} P={P:.1f} S={S:.1f}"),
        ("defection dominates", all(rows[k]["defector"] > rows[k]["cooperator"]
                                    for k in (1, 2, 3)),
         "defecting beats cooperating at every mix"),
        ("welfare falls with defectors", all(rows[k]["welfare"] >= rows[k + 1]["welfare"] - 1e-9
                                             for k in range(4)),
         "each additional defector leaves the group worse off"),
        ("free-riding is unsafe", rows[2]["collapse"] >= 0.5,
         f"half defecting kills it {rows[2]['collapse']*100:.0f}% of the time"),
    ]
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<30} {detail}")
    print(f"\n  cooperation pays {R / P:.2f}x mutual defection; "
          f"defecting alone pays {T / R:.2f}x cooperating")
    return 0 if all(ok for _, ok, _ in checks) else 1


def cmd_axelrod(args) -> int:
    """Does monitoring change the game? Run the same policies with and without it.

    The reciprocal policy is nice, provokable, forgiving and clear -- Axelrod's
    four qualities. It can only be *expressed* when harvests are attributable,
    so this compares identical agents across worlds that differ in one respect:
    whether anyone can see who took what.
    """
    print("Axelrod check · identical policies, three monitoring regimes")
    print(f"rule={args.rule} r={args.r} · {args.runs} seeds\n")
    print(f"{'monitoring':<12}{'group':<18}{'survived':<11}{'collapse%':<12}"
          f"{'welfare':<10}{'restrained':<12}{'defector'}")
    print("-" * 84)

    results = {}
    for mon in ("none", "local", "global"):
        for mix in ("cautious", "reciprocal", "generous",
                    "caut_v_greedy", "recip_v_greedy", "gen_v_greedy"):
            eps = [run_episode(s, mix, args.rule, args.r, monitoring=mon)
                   for s in range(args.runs)]
            per = np.array([e.scores for e in eps], dtype=float)
            surv = float(np.mean([e.survived for e in eps]))
            coll = float(np.mean([e.survived < TICKS for e in eps]))
            welfare = float(per.sum(axis=1).mean())
            # in the head-to-heads, agents 0-1 restrain and 2-3 defect
            restrained = float(per[:, :2].mean())
            head_to_head = mix.endswith("_v_greedy")
            defector = float(per[:, 2:].mean()) if head_to_head else float("nan")
            results[(mon, mix)] = (surv, coll, welfare, restrained, defector)
            d = f"{defector:.1f}" if head_to_head else "-"
            print(f"{mon:<12}{mix:<18}{surv:<11.1f}{coll * 100:<12.0f}"
                  f"{welfare:<10.1f}{restrained:<12.1f}{d}")
        print()

    print("against the same two defectors, what does reciprocity buy?")
    for mon in ("none", "local", "global"):
        c_ = results[(mon, "caut_v_greedy")]
        r_ = results[(mon, "recip_v_greedy")]
        g_ = results[(mon, "gen_v_greedy")]
        print(f"  monitoring={mon:<7} restraint {c_[3]:5.1f} | reciprocal {r_[3]:5.1f} "
              f"({r_[3]-c_[3]:+.1f}) | generous {g_[3]:5.1f} ({g_[3]-c_[3]:+.1f})")
    print("\nand what does a group of them do to the commons?")
    for mon in ("none", "local", "global"):
        for mix in ("cautious", "reciprocal", "generous"):
            v = results[(mon, mix)]
            print(f"  monitoring={mon:<7} all-{mix:<11} survives {v[0]:5.1f} ticks, "
                  f"collapses {v[1]*100:3.0f}%, welfare {v[2]:5.1f}")
    print("\n  With nothing to observe, reciprocity has no trigger and collapses into")
    print("  plain restraint. Attribution is what makes retaliation -- and therefore")
    print("  cooperation -- expressible at all.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Overgraze scripted-agent harness")
    p.add_argument("--runs", type=int, default=100, help="episodes to run (default 100)")
    p.add_argument("--rule", choices=["global", "neighbour"], default="global")
    p.add_argument("--r", type=float, default=TUNED_R, help="regrowth rate")
    p.add_argument("--mix", choices=sorted(MIXES), default="mixed")
    p.add_argument("--out", default="stock.csv", help="CSV path for stock over time")
    p.add_argument("--sweep", action="store_true",
                   help="scan regrowth rates for the tuning gate instead")
    p.add_argument("--dilemma", action="store_true",
                   help="measure the payoff structure and check it is a real dilemma")
    p.add_argument("--axelrod", action="store_true",
                   help="does monitoring make reciprocity possible? compare regimes")
    p.add_argument("--monitoring", choices=["none", "local", "global"], default="local",
                   help="who can see who harvested what (default local)")
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30],
                   help="regrowth rates to scan with --sweep")
    args = p.parse_args(argv)
    if args.axelrod:
        return cmd_axelrod(args)
    if args.dilemma:
        return cmd_dilemma(args)
    return cmd_sweep(args) if args.sweep else cmd_runs(args)


if __name__ == "__main__":
    sys.exit(main())
