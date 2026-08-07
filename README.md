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
python compare.py
```

Sweeps both regrowth rules × five regrowth rates × three group compositions and
prints how long the commons lasted and how much the group harvested.

```bash
python payoff.py
```

Breaks the mixed group down to per-agent scores — does defection actually pay? —
and counts dead vs recovered cells after an all-greedy run.

## What's in here

| File | What it is |
|---|---|
| `compare.py` | The simulation, plus the headline sweep. Everything else imports from it. |
| `payoff.py` | Per-agent payoffs and dead-zone analysis. Imports `run`, `N`, `CAP`, `R_VALUES`. |
| `export_viz.py` | Replays `compare.run()` recording every tick → `viz_data.json`. |
| `viz_template.html` | The visualiser page, with a `/*__VIZ_DATA__*/` placeholder. |
| `build_viz.py` | Injects the data into the template → self-contained `overgraze.html`. |

`viz_data.json` and `overgraze.html` are build outputs and are gitignored.

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

Means over 40 seeds. `4G` = four greedy, `4C` = four cautious, `2/2` = mixed.

| rule | r | survived 4G | harvest 4G | harvest 4C | greedy each (2/2) | cautious each (2/2) | dilemma |
|---|---|---|---|---|---|---|---|
| global | 0.11 | 33.2 | 56.3 | 101.6 | 42.4 | 9.0 | no |
| global | 0.13 | 38.8 | 64.5 | 113.9 | 51.0 | 13.4 | no |
| global | 0.15 | 46.9 | 75.9 | 125.1 | 53.8 | 20.0 | no |
| global | 0.30 | 100.0 | 220.0 | 175.3 | 55.0 | 42.8 | no |
| global | 0.50 | 100.0 | 220.0 | 193.9 | 55.0 | 48.3 | no |
| neighbour | 0.11 | 25.9 | 46.7 | 85.6 | 23.2 | 7.1 | **yes** |
| neighbour | 0.13 | 28.2 | 50.4 | 96.4 | 29.2 | 8.1 | **yes** |
| neighbour | 0.15 | 30.6 | 54.7 | 106.5 | 38.9 | 9.5 | no |
| neighbour | 0.30 | 100.0 | 192.4 | 160.0 | 54.9 | 37.0 | no |
| neighbour | 0.50 | 100.0 | 220.0 | 187.8 | 55.0 | 46.5 | no |

**The dilemma occupies a narrow band.** At r ≥ 0.30 regrowth simply outruns
extraction: greedy wins individually *and* collectively (220 vs 175), so
defection is just correct and there is no tension to study. Under the `global`
rule the pooled regrowth bails out the mixed group, so partial cooperation is a
stable escape hatch. Only `neighbour` at r = 0.11 and r = 0.13 produces the full
structure.

**`neighbour` / r = 0.13 is the sharpest case.** All-cautious survives all 100
ticks. All-greedy collapses at tick 28. And the mixed group *also* collapses, at
tick 77 — so cooperation has a critical mass: two cooperators cannot offset two
defectors, and each greedy forager still personally earns ~3.6× what a cautious
one does. Restraint there earns those two foragers nothing and saves nothing.

**The damage is permanent within a run.** After an all-greedy run at r = 0.15,
22 of the 36 cells (`global`) or 27 (`neighbour`) sit at ~0, and *zero* cells
have recovered above 0.5 under either rule.

## The visualiser

```bash
python export_viz.py && python build_viz.py
```

Produces `overgraze.html`, a self-contained page (no external scripts, fonts,
or fetches) with a grid player, sweep charts with per-seed error bars, and a
table of every number. The map draws each cell's resource as tufts of grass that
get eaten one by one, with greedy foragers as goats and cautious ones as sheep —
distinguished by silhouette as well as colour.

`export_viz.py` replays `compare.run()` rather than reimplementing it, and its
recorded loop is asserted identical to the original across all 30 configurations.
The picture cannot drift from the code.

Published (private): https://claude.ai/code/artifact/3a83f628-6372-451a-aceb-c07c7d6d7559

## Design decisions and known deviations

**The tick model is sequential, not simultaneous.** Foragers act one at a time
and mutate the grid in place, so the second forager chooses against a grid the
first has already eaten from. A design spec calling for simultaneous resolution
(all agents read the same tick-N snapshot, submit intents, the server resolves
them together) is *not* implemented, deliberately. It was prototyped and
measured, and it is not a cosmetic fairness fix — under simultaneous choice the
foragers pile onto the same richest cell instead of the leader stripping it and
the rest fanning out. Concentrated damage is far easier for regrowth to repair,
so collapses largely stop happening: all-greedy at `global`/0.11 goes from
surviving 33 ticks to 99, and `neighbour`/0.13 stops being a dilemma at all.
Adopting it would be defensible, but it invalidates every number above.

**No default `r` is designated.** Tuning by hand against a scalar pool under a
constant drain of 4 × 0.55 gives r ≈ 0.187 for a ~40-tick collapse. The agent
sim needs r ≈ 0.133 for the same result, because foragers realise only ~1.64 of
the assumed 2.20 drain per tick — cells fall below `TAKE`, and hill-climbing
crowds them onto the same cells. Rather than pick one, the sweep brackets both.

**Seed counts are inconsistent** between scripts: `compare.py` uses 12,
`payoff.py` 20, `export_viz.py` 40. Numbers therefore differ slightly between
outputs. Worth unifying.

**`Agent.act` has an unreachable third branch** (a `cell > 0.3` threshold
policy). The sweep only ever builds `greedy` and `cautious` foragers.

**`neighbours_mean` divides by 9 everywhere**, including at edges where part of
the neighbourhood is zero padding — so edge cells systematically under-regrow
under the `neighbour` rule, biasing the map toward a rich centre and dead rim.
Left as-is because it is load-bearing for the spatial story, but it is a
modelling choice, not a neutral one.
