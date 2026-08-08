# Overgraze

An agent-based study of the tragedy of the commons. Four foragers share a 6×6
grid of renewable cells. Each tick they move to the richest cell in reach and
harvest it; then the grid regrows. Greedy foragers strip a cell to nothing.
Cautious ones leave half behind.

The question the code exists to answer: **when does restraint actually pay?**
Not as a moral claim — as a measurable one. A configuration is a genuine
dilemma only when both things are true at once:

1. A greedy forager out-earns a cautious one (the individual temptation), and
2. A group of greedy foragers out-harvests itself into collapse, ending up
   worse off than a group of cautious ones would have been (the collective loss).

Either one alone is not a dilemma. Most parameter settings turn out not to be.

## Quick start

```bash
python harness.py --runs 100 --out stock.csv
```

Runs 100 scripted episodes and writes stock over time, plus survival and
harvest summaries to stdout.

```bash
python harness.py --sweep
```

Scans regrowth rates for the band where a group of greedy foragers collapses
the commons while a group of cautious ones survives.

```bash
python -m unittest test_world -v
```

30 tests, most of them on the rules for resolving what happens when two
foragers want the same cell on the same tick.

## What's in here

| File | What it is |
|---|---|
| `world.py` | The engine. Pure functions, no I/O: `apply_actions(state, actions) -> (state, events)`. |
| `harness.py` | Scripted policies, the episode runner, and the CLI. Imports `world`. |
| `test_world.py` | Unit tests for the engine, weighted toward the contention rules. |
| `export_viz.py` | Runs the harness recording every tick → `viz_data.json`. |
| `viz_template.html` | The visualiser page, with a `/*__VIZ_DATA__*/` placeholder. |
| `build_viz.py` | Injects the data into the template → self-contained `overgraze.html`. |

`viz_data.json`, `overgraze.html` and any `.csv` are build outputs and are
gitignored.

## The model

**Grid.** 6×6 cells, each holding up to `CAP = 1.0`. Starts completely full.

**Foragers.** Four, starting in the four corners. Each tick a forager considers
staying put plus its four orthogonal neighbours and moves to whichever holds the
most, breaking ties at random. Then it harvests, up to `TAKE = 0.55` per tick:

| kind | takes |
|---|---|
| `greedy` | `min(cell, TAKE)` — will strip a cell to zero |
| `cautious` | `min(max(cell - 0.5, 0), TAKE)` — leaves half a cell as seed stock |

**Regrowth**, applied after all foragers have acted, at rate `r`:

- **`global`** — logistic growth on the *total* stock, `r × S × (1 − S/capacity)`,
  computed once from the whole-grid sum and then distributed across cells in
  proportion to how much empty room each has. A stripped corner is refilled by
  the health of the map as a whole, so local overharvesting is subsidised by
  everyone else.
- **`neighbour`** — local growth. Each cell regrows at `r × (its neighbourhood
  mean) × (its own empty room)`. A cell stripped to zero in a stripped region
  has nothing to recover from and stays dead.

**Collapse.** A run stops early if total resource falls below 5% of capacity
(1.8 of 36). `survived < 100` is the signature of a collapse.

**Randomness.** Two draws, both seeded: movement tie-breaks, and the order
foragers act in each tick. The activation shuffle matters — acting first means
harvesting before the others see the cell, and under a fixed order that
advantage always fell to the two greedy foragers, biasing the very comparison
this project makes.

## Results

From `world.py`, means over 40 seeds. `4G` = four greedy, `4C` = four cautious,
`2/2` = mixed. "contested" counts cells two or more foragers targeted on the
same tick — the case the engine has to arbitrate.

| rule | r | survived 4G | harvest 4G | harvest 4C | greedy each (2/2) | cautious each (2/2) | contested | dilemma |
|---|---|---|---|---|---|---|---|---|
| global | 0.035 | 34.5 | 41.0 | 49.7 | 16.1 | 6.5 | 56 | **yes** |
| global | 0.05 | 46.5 | 46.2 | 61.8 | 20.1 | 7.1 | 76 | no |
| global | 0.08 | 67.6 | 63.6 | 85.6 | 26.1 | 13.4 | 94 | no |
| global | 0.12 | 100.0 | 107.6 | 111.5 | 35.4 | 21.0 | 115 | no |
| global | 0.20 | 100.0 | 158.0 | 142.3 | 40.3 | 34.7 | 122 | no |
| neighbour | 0.035 | 31.9 | 39.8 | 48.9 | 14.3 | 6.8 | 44 | **yes** |
| neighbour | 0.05 | 42.9 | 44.0 | 59.8 | 17.1 | 7.7 | 66 | **yes** |
| neighbour | 0.08 | 55.6 | 56.5 | 80.6 | 23.1 | 10.8 | 84 | no |
| neighbour | 0.12 | 87.6 | 85.5 | 103.9 | 31.2 | 18.1 | 114 | no |
| neighbour | 0.20 | 100.0 | 147.3 | 136.6 | 39.3 | 31.9 | 118 | no |

**A greedy forager always out-earns a cautious one.** Across every rate, in
every mixed group, the individual temptation holds — 2.5× at r = 0.035, still
1.2× at r = 0.20. Defection is never individually irrational here.

**The collective picture flips at r ≈ 0.12–0.20.** Below it a group of cautious
foragers out-harvests a group of greedy ones (49.7 vs 41.0 at r = 0.035); above
it regrowth outruns extraction and greedy wins outright (158.0 vs 142.3 at
r = 0.20). Only where both conditions hold at once — the greedy advantage *and*
a collapsing commons — is this a dilemma: `global` at 0.035, and `neighbour` at
0.035 and 0.05.

**Local regrowth is less forgiving than pooled regrowth.** At the same rate the
`neighbour` rule collapses sooner and harvests less, because a stripped cell can
only recover from surviving neighbours. Pooled regrowth subsidises the damage
from the health of the whole map.

**Contention rises with abundance.** Foragers crowd the same cells more often
when there is more to crowd over — 44 contested cells per run at the lowest
rate, 122 at the highest.

## The visualiser

```bash
python export_viz.py && python build_viz.py
```

Produces `overgraze.html`, a self-contained page (no external scripts, fonts,
or fetches) with a grid player, sweep charts with per-seed error bars, and a
table of every number. The map draws each cell's resource as tufts of grass that
get eaten one by one, with greedy foragers as goats and cautious ones as sheep —
distinguished by silhouette as well as colour.

`export_viz.py` drives `world.apply_actions` through the scripted harness rather
than reimplementing anything, so the picture cannot drift from the engine.

Published (private): https://claude.ai/code/artifact/3a83f628-6372-451a-aceb-c07c7d6d7559

## History

An earlier engine (`compare.py`, `payoff.py`) ran agents sequentially within a
tick, mutating the grid in place, so the second forager chose against a grid the
first had already eaten from. It could not represent two foragers harvesting one
cell on the same tick at all, and it had no tests.

It was replaced rather than repaired, and deleted once `world.py` covered it —
`git log` has it if you need the old numbers. Two things worth carrying forward:

- **Its regrowth rate does not transfer.** The old dilemma band was r ≈ 0.11–0.13.
  Simultaneous choice makes foragers pile onto the same rich cell instead of the
  leader stripping it and the rest fanning out, and concentrated damage is easier
  for regrowth to repair — so the same rate collapses far less readily. `world.py`
  was retuned from scratch and lands at r ≈ 0.035–0.05.
- **Its edge cells were biased.** `neighbours_mean` divided by 9 everywhere,
  including at edges where part of the neighbourhood was zero padding, so rim
  cells systematically under-regrew. `world.py` divides by the true neighbour
  count.

## Design decisions and known deviations

**Performance is short of target.** The Phase 1 brief wants 1000 runs a second;
the engine does roughly 68 full-length episodes a second, so 1000 takes ~15s.
Profiling put a third of the cost in per-tick RNG construction, fixed by giving
each agent one spawned stream per episode (2.4× faster). What remains is numpy's
per-call overhead on a 6×6 grid, where 36 floats do not amortise a ~1µs call.
Closing the gap means dropping numpy for plain lists in the hot path, or
vectorising across episodes. The Phase 1 gate — 100 simulations to CSV — passes
in 1.2s regardless.

**The contention rule is a modelling choice.** When foragers target one cell,
`resolve_cell` splits it max-min fair. Proportional-to-ask, random priority, or
all-or-nothing would each give different dynamics. Max-min fair was chosen
because it is order-independent and conserves exactly, both of which are
testable properties; it is not the only defensible rule.

**Scripted policies are deliberately dumb.** `greedy`, `cautious` and `random`
hill-climb or wander; none of them model, negotiate, or anticipate. They exist
to make the world cheap to tune and test, not to be interesting agents.
