"""
Game-theoretic benchmarks for Overgraze.

The simulation tells you what agents *did*. This tells you what they *should*
have done, so a result can be stated as a fraction of the achievable rather
than as a bare number.

Three benchmarks:

1. **The social planner's optimum.** One decision-maker owning the whole
   commons, maximising total harvest over the run. Solved by dynamic
   programming on total stock, ignoring geography -- so it is an upper bound no
   set of agents can beat.

2. **The open-access (Nash) path.** Every agent extracting at its private
   maximum with no regard for the stock: the tragedy, as a trajectory.

3. **The folk-theorem threshold.** How much agents must value the next tick for
   cooperation to survive as an equilibrium of the repeated game. Below it,
   defection is not a failure of reasoning -- it is correct play.

    python theory.py
"""

from __future__ import annotations

import numpy as np

from world import CAPACITY, COLLAPSE_FLOOR, TAKE, TICKS

MAX_HARVEST = 4 * TAKE          # what four agents can physically take in a tick
GRID = 400                      # stock discretisation for the DP


def growth(S: float, r: float) -> float:
    return r * S * (1 - S / CAPACITY)


def planner_optimum(r: float, ticks: int = TICKS) -> tuple[float, list[float]]:
    """Best total harvest one owner could extract, by backward induction.

    State is total stock; the run ends if stock falls below the viability floor,
    so the planner is not allowed to strip the commons and walk away.
    """
    stocks = np.linspace(0.0, CAPACITY, GRID + 1)
    V = np.zeros((ticks + 1, GRID + 1))          # value-to-go

    for t in range(ticks - 1, -1, -1):
        for i, S in enumerate(stocks):
            if S < COLLAPSE_FLOOR:
                V[t, i] = 0.0                     # dead: nothing more to take
                continue
            best = 0.0
            hi = min(MAX_HARVEST, S - COLLAPSE_FLOOR)
            for h in np.linspace(0.0, max(hi, 0.0), 45):
                nxt = min(CAPACITY, S - h + growth(S - h, r))
                j = int(round(nxt / CAPACITY * GRID))
                val = h + V[t + 1, min(max(j, 0), GRID)]
                if val > best:
                    best = val
            V[t, i] = best

    # replay the optimal policy to recover the path
    S, path, total = CAPACITY, [CAPACITY], 0.0
    for t in range(ticks):
        if S < COLLAPSE_FLOOR:
            break
        best, best_h = -1.0, 0.0
        hi = min(MAX_HARVEST, S - COLLAPSE_FLOOR)
        for h in np.linspace(0.0, max(hi, 0.0), 45):
            nxt = min(CAPACITY, S - h + growth(S - h, r))
            j = int(round(nxt / CAPACITY * GRID))
            val = h + V[t + 1, min(max(j, 0), GRID)]
            if val > best:
                best, best_h = val, h
        total += best_h
        S = min(CAPACITY, S - best_h + growth(S - best_h, r))
        path.append(S)
    return total, path


def open_access(r: float, ticks: int = TICKS) -> tuple[float, list[float], int]:
    """Everyone takes the maximum every tick. The tragedy, as a trajectory."""
    S, total, path = CAPACITY, 0.0, [CAPACITY]
    for t in range(ticks):
        h = min(MAX_HARVEST, S)
        total += h
        S = min(CAPACITY, S - h + growth(S - h, r))
        path.append(S)
        if S < COLLAPSE_FLOOR:
            return total, path, t + 1
    return total, path, ticks


def msy(r: float) -> tuple[float, float]:
    """Maximum sustainable yield of the logistic rule, and the stock it sits at."""
    return r * CAPACITY / 4, CAPACITY / 2


def folk_threshold(T: float, R: float, P: float) -> float:
    """Minimum discount factor for grim-trigger cooperation to be an equilibrium.

    Deviating pays T now and P forever after; cooperating pays R forever. The
    deviation is unprofitable when  T + dP/(1-d) <= R/(1-d), i.e.

        d >= (T - R) / (T - P)

    A high threshold means cooperation is fragile: agents must care about the
    future almost as much as the present for restraint to be rational.
    """
    return (T - R) / (T - P) if T > P else 0.0


def report(r: float = 0.05) -> None:
    opt, opt_path = planner_optimum(r)
    oa, oa_path, oa_died = open_access(r)
    y, s_star = msy(r)

    print(f"benchmarks · logistic r={r}, capacity {CAPACITY:.0f}, "
          f"viability floor {COLLAPSE_FLOOR:.0f}, {TICKS} ticks")
    print("-" * 68)
    print(f"  maximum sustainable yield   {y:.2f}/tick at stock {s_star:.0f}")
    print(f"  four agents can take        {MAX_HARVEST:.2f}/tick "
          f"({MAX_HARVEST / y:.1f}x the sustainable rate)")
    print()
    print(f"  social planner's optimum    {opt:6.1f}   (upper bound, one owner)")
    print(f"  open access, all-out        {oa:6.1f}   "
          f"(commons dead at tick {oa_died})")
    print(f"  the tragedy costs           {opt - oa:6.1f}   "
          f"= {100 * (1 - oa / opt):.0f}% of the achievable")
    print()
    print("  Extraction capacity exceeding sustainable yield by this much is what")
    print("  makes the commons destroyable at all. If four agents could not")
    print("  out-take the regrowth, there would be no dilemma to study.")


def report_equilibrium(T: float, R: float, P: float, S: float) -> None:
    d = folk_threshold(T, R, P)
    print()
    print("repeated game · can cooperation hold?")
    print("-" * 68)
    print(f"  T={T:.1f} (defect alone)  R={R:.1f} (all cooperate)  "
          f"P={P:.1f} (all defect)  S={S:.1f} (be the sucker)")
    print(f"  grim-trigger threshold      d >= {d:.2f}")
    print(f"  meaning: an agent must value the next tick at {d * 100:.0f}% of this")
    print("  one for restraint to beat defection. Below that, defecting is not a")
    print("  failure of reasoning -- it is correct play.")


if __name__ == "__main__":
    report(0.05)
    # measured payoffs from `python harness.py --dilemma`
    report_equilibrium(T=38.7, R=15.6, P=8.8, S=6.9)
