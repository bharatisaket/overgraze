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

**Foragers.** Four, starting in the four corners.

**A tick.** Every agent submits its intents for tick N; the engine resolves them
all against the same tick-N snapshot and only then advances. Each agent may
submit at most one of each, and a second is an error the agent is told about
(*"you already acted this tick"*) rather than a silent overwrite:

| channel | actions |
|---|---|
| move | `move(north\|south\|east\|west\|stay)` — resolves first, so a harvest lands on the destination |
| resource | `harvest(amount)` up to `TAKE = 0.55`, `plant()` adding `0.15`, `punish(agent)`, `noop` |
| speech | `say(text)` — heard by agents within vision |

Movement and speech are separate channels rather than competing actions, both
deliberately. Charging a tick for a step halves extraction while leaving
regrowth untouched — the commons then survives 70–90 ticks even at r = 0.002 and
restraint never pays, so the dilemma disappears. And if talking cost an agent
its harvest, the chat-on/chat-off ablation would measure the price of speaking
rather than the effect of communication.

**Reads are free.** `look()` shows only what is within vision (Chebyshev radius
1) — cells and nearby agents, not the whole grid. `listen()` returns messages
spoken in earshot. `status()` reports score, tick, and the run's rules. None of
them consume a tick.

**Contention.** When several agents harvest one cell, `resolve_cell` splits it
max-min fair: equal shares, surplus from a small ask flowing back to whoever
still wants more, never more than asked and never more than the cell holds.

**Ablation switches** live on the state, so a run is fully described by it:
`chat`, `punish`, `anonymous`, `vision`.

**Scripted policies:**

| kind | behaviour |
|---|---|
| `greedy` | step onto the richest cell in reach and strip it, `min(cell, TAKE)` |
| `cautious` | same step, but never below half — and replant when nothing is spare |
| `random` | wander, harvest arbitrary amounts, occasionally plant |

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

**Randomness.** One seeded draw: which cell a forager picks when several are
tied. Each agent has its own stream, so the order agents are processed in cannot
influence a run.

## Results

From `world.py`, means over 40 seeds. `4G` = four greedy, `4C` = four cautious,
`2/2` = mixed.

| rule | r | survived 4G | harvest 4G | harvest 4C | greedy each (2/2) | cautious each (2/2) |
|---|---|---|---|---|---|---|
| global | 0.02 | 29.1 | 37.8 | **49.3** | 21.7 | 6.1 |
| global | 0.04 | 39.3 | 42.6 | **60.1** | 26.7 | 7.6 |
| global | 0.06 | 52.9 | 50.9 | **73.8** | 29.7 | 10.5 |
| global | 0.10 | 91.4 | 83.4 | **98.7** | 32.8 | 18.9 |
| global | 0.15 | 100.0 | **135.5** | 119.5 | 37.5 | 27.9 |
| neighbour | 0.02 | 28.1 | 37.3 | **49.2** | 21.2 | 6.2 |
| neighbour | 0.04 | 34.4 | 41.0 | **61.3** | 25.3 | 7.7 |
| neighbour | 0.06 | 48.3 | 47.5 | **72.6** | 27.4 | 10.8 |
| neighbour | 0.10 | 67.4 | 68.2 | **95.7** | 31.6 | 17.2 |
| neighbour | 0.15 | 98.3 | 115.4 | **116.0** | 36.0 | 25.1 |

**The tuned rate is r = 0.04.** It is what Phase 0 asked for: an all-greedy
group collapses the commons at ~40 ticks, while an all-cautious group survives
all 100 and out-harvests it, 60.1 to 42.6. `harness.TUNED_R` holds it and the
CLI defaults to it.

**A greedy forager always out-earns a cautious one** in a mixed group — 3.6× at
r = 0.02, still 1.3× at r = 0.15. Defection is never individually irrational.

**The collective picture flips at r ≈ 0.15.** Below it a group of cautious
foragers out-harvests a group of greedy ones; at 0.15 regrowth outruns
extraction and greedy wins outright (135.5 vs 119.5), so there is no dilemma
left to study.

**Two cooperators are now enough to save the commons.** Unlike the earlier
engine, mixed groups do not collapse: the cautious pair replant when nothing is
spare, and `plant()` is a lever the old world did not have. The tragedy shows up
in the all-greedy runs, not the mixed ones.

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
