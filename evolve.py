"""
Evolutionary tournament: can cooperation invade?

Axelrod's tournament asked which strategy scores best. His evolutionary
simulation asked the harder question -- whether a strategy *spreads*. A lone
reciprocator among defectors is exploited and dies; a cluster of them, meeting
each other often enough, can take over a population of pure defectors.

That result depends on **assortment**: cooperators must interact preferentially
with other cooperators. In a pairwise tournament that comes free. In a commons
it does not -- everyone draws from the same stock, and you cannot choose to
share a resource only with people who deserve it. Whether Axelrod's result
survives that move is an open question this file is built to answer.

Two knobs stand in for the two ways a commons can supply assortment:

  --assort    how often a foraging group forms from one strategy (clustering)
  --monitoring  'local' means you only witness neighbours, so a cluster is
                also an information island; 'global' dissolves that structure

    python evolve.py --invasion            # can 10% reciprocators take over?
    python evolve.py --scan                # assortment x monitoring
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

import harness

GROUP = 4       # the world seats exactly four foragers


def make_groups(pop: list[str], assort: float, rng) -> list[list[int]]:
    """Partition the population into foraging groups.

    With probability `assort` a group is drawn from a single strategy -- the
    cluster that lets cooperation get a foothold. Otherwise it is a random mix.
    """
    remaining = list(rng.permutation(len(pop)))
    groups = []
    while len(remaining) >= GROUP:
        if rng.random() < assort:
            kind = pop[remaining[0]]
            same = [i for i in remaining if pop[i] == kind][:GROUP]
            group = same if len(same) == GROUP else remaining[:GROUP]
        else:
            group = remaining[:GROUP]
        for i in group:
            remaining.remove(i)
        groups.append(list(group))
    return groups


def generation(pop: list[str], rule: str, r: float, monitoring: str,
               noise: float, assort: float, rng, misreport: float = 0.0) -> tuple[list[float], dict]:
    """Play one generation. Returns per-individual payoff and some diagnostics."""
    fitness = [0.0] * len(pop)
    survived, collapses, episodes = 0.0, 0, 0
    for group in make_groups(pop, assort, rng):
        kinds = [pop[i] for i in group]
        ep = harness.run_episode(int(rng.integers(1 << 30)), kinds, rule, r,
                                 monitoring=monitoring, noise=noise,
                                 misreport=misreport)
        for slot, i in enumerate(group):
            fitness[i] = ep.scores[slot]
        survived += ep.survived
        collapses += int(ep.survived < harness.TICKS)
        episodes += 1
    return fitness, {"survived": survived / max(episodes, 1),
                     "collapse_rate": collapses / max(episodes, 1)}


def reproduce(pop: list[str], fitness: list[float], rng) -> list[str]:
    """Replicate proportional to payoff. Negative scores cannot reproduce."""
    w = np.array([max(f, 0.0) for f in fitness], dtype=float) + 1e-9
    picks = rng.choice(len(pop), size=len(pop), p=w / w.sum())
    return [pop[i] for i in picks]


def run(start: dict[str, int], generations: int, seed: int = 0,
        rule: str = "global", r: float | None = None, monitoring: str = "local",
        noise: float = 0.0, assort: float = 0.0, verbose: bool = False,
        misreport: float = 0.0):
    r = harness.TUNED_R if r is None else r
    rng = np.random.default_rng(seed)
    pop = [k for kind, n in start.items() for k in [kind] * n]
    history = [Counter(pop)]
    diags = []

    for g in range(generations):
        fitness, diag = generation(pop, rule, r, monitoring, noise, assort, rng,
                                   misreport)
        pop = reproduce(pop, fitness, rng)
        history.append(Counter(pop))
        diags.append(diag)
        if verbose:
            comp = " ".join(f"{k}:{v}" for k, v in sorted(history[-1].items()))
            print(f"  gen {g + 1:3d}  {comp:<44} collapse {diag['collapse_rate']*100:3.0f}%")
        if len(history[-1]) == 1:                      # fixation
            break
    return history, diags


def summarise(label: str, history, diags, target: str) -> None:
    first, last = history[0], history[-1]
    n = sum(last.values())
    start_share = first.get(target, 0) / sum(first.values())
    end_share = last.get(target, 0) / n
    verdict = ("FIXATED" if end_share == 1 else
               "invaded" if end_share > start_share + 0.05 else
               "died out" if end_share == 0 else
               "held on" if end_share >= start_share - 0.05 else "shrank")
    coll = np.mean([d["collapse_rate"] for d in diags]) if diags else 0.0
    print(f"{label:<38}{start_share * 100:5.0f}% -> {end_share * 100:5.0f}%   "
          f"{verdict:<9} commons collapsed {coll * 100:3.0f}% of episodes")


def cmd_invasion(args) -> int:
    n_recip = max(1, round(args.pop * args.share))
    start = {"reciprocal": n_recip, "greedy": args.pop - n_recip}
    print(f"invasion · {n_recip}/{args.pop} reciprocators seeded into a greedy population")
    print(f"rule={args.rule} r={args.r or harness.TUNED_R} monitoring={args.monitoring} "
          f"noise={args.noise} assort={args.assort}\n")
    hist, diags = run(start, args.generations, args.seed, args.rule, args.r,
                      args.monitoring, args.noise, args.assort, verbose=True,
                      misreport=args.misreport)
    print()
    summarise("result", hist, diags, "reciprocal")
    return 0


def cmd_scan(args) -> int:
    """Isolate assortment from monitoring -- the two ways a commons can cluster."""
    n_recip = max(1, round(args.pop * args.share))
    start = {"reciprocal": n_recip, "greedy": args.pop - n_recip}
    print(f"can {args.share * 100:.0f}% reciprocators invade a greedy population?")
    print(f"pop={args.pop} generations={args.generations} r={args.r or harness.TUNED_R} "
          f"noise={args.noise} · {args.seeds} seeds each\n")
    print(f"{'assortment':<14}{'monitoring':<12}{'share':<18}{'verdict':<10}{'commons'}")
    print("-" * 74)
    for assort in (0.0, 0.5, 0.9):
        for mon in ("none", "local", "global"):
            ends, colls = [], []
            for sd in range(args.seeds):
                hist, diags = run(start, args.generations, sd, args.rule, args.r,
                                  mon, args.noise, assort, misreport=args.misreport)
                ends.append(hist[-1].get("reciprocal", 0) / args.pop)
                colls.append(np.mean([d["collapse_rate"] for d in diags]))
            end = float(np.mean(ends))
            verdict = ("FIXATED" if end > 0.95 else "invaded" if end > args.share + 0.05
                       else "died out" if end < 0.02 else "held on")
            print(f"{assort:<14.1f}{mon:<12}{args.share * 100:3.0f}% -> {end * 100:5.1f}%      "
                  f"{verdict:<10}{np.mean(colls) * 100:3.0f}% collapse")
        print()
    return 0


def cmd_noise(args) -> int:
    """Which answer to noisy play actually works: generosity, or patience?

    Sweeps PERCEPTION noise, where the question is well posed -- a misreport
    moves no resource, so anything that happens is reciprocity misfiring and
    nothing else. Under execution noise the two effects are confounded.
    """
    n = max(1, round(args.pop * args.share))
    strategies = ("reciprocal", "generous", "tolerant")
    print(f"perception noise · {args.share * 100:.0f}% seeded into a greedy population")
    print(f"pop={args.pop} generations={args.generations} assort={args.assort} "
          f"monitoring={args.monitoring} execution-noise={args.noise} "
          f"· {args.seeds} seeds each\n")
    print(f"{'misreport':<11}{'strategy':<13}{'share':<20}{'commons':<16}{'retaliating'}")
    print("-" * 74)
    table = {}
    for mis in (0.0, 0.05, 0.10, 0.20):
        for strat in strategies:
            start = {strat: n, "greedy": args.pop - n}
            ends, colls = [], []
            for sd in range(args.seeds):
                hist, diags = run(start, args.generations, sd, args.rule, args.r,
                                  args.monitoring, args.noise, args.assort,
                                  misreport=mis)
                ends.append(hist[-1].get(strat, 0) / args.pop)
                colls.append(np.mean([d["collapse_rate"] for d in diags]))
            end, coll = float(np.mean(ends)), float(np.mean(colls))
            sd_end = float(np.std(ends))
            table[(mis, strat)] = (end, coll)
            print(f"{mis:<11.2f}{strat:<13}{end * 100:5.1f}% (sd {sd_end * 100:4.1f})   "
                  f"{coll * 100:3.0f}% collapse    -")
        print()

    print("does either kind of forgiveness beat unforgiving reciprocity?")
    for mis in (0.05, 0.10, 0.20):
        base = table[(mis, "reciprocal")]
        gen = table[(mis, "generous")]
        tol = table[(mis, "tolerant")]
        print(f"  misreport {mis:.2f}  generous {gen[0] * 100 - base[0] * 100:+5.1f}pp share "
              f"{gen[1] * 100 - base[1] * 100:+4.0f}pp collapse   |   "
              f"tolerant {tol[0] * 100 - base[0] * 100:+5.1f}pp share "
              f"{tol[1] * 100 - base[1] * 100:+4.0f}pp collapse")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Overgraze evolutionary tournament")
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--share", type=float, default=0.25,
                   help="starting share of reciprocators (default 0.25)")
    p.add_argument("--assort", type=float, default=0.0,
                   help="chance a group forms from one strategy (clustering)")
    p.add_argument("--monitoring", choices=["none", "local", "global"], default="local")
    p.add_argument("--noise", type=float, default=0.0,
                   help="execution error: a harvest slips to full TAKE")
    p.add_argument("--misreport", type=float, default=0.0,
                   help="perception error: witnesses misread a harvest")
    p.add_argument("--rule", choices=["global", "neighbour"], default="global")
    p.add_argument("--r", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=3, help="seeds per cell in --scan")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--invasion", action="store_true")
    p.add_argument("--noise-scan", dest="noise_scan", action="store_true")
    args = p.parse_args(argv)
    if args.noise_scan:
        return cmd_noise(args)
    return cmd_scan(args) if args.scan else cmd_invasion(args)


if __name__ == "__main__":
    raise SystemExit(main())
