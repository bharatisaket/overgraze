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

**Collapse.** A run ends when the commons can no longer support four foragers —
below **a quarter of capacity** (9 of 36), where the average cell holds less than
one agent's bite and total regrowth is about 0.34/tick split four ways. Not when
the grid is literally empty.

That line is doing real work, not bookkeeping. With a 5%-of-capacity floor, two
defectors could never drain 34 units inside the 100-tick budget — they extract
~0.5/tick and would need a 0.34/tick deficit that no regrowth rate permits while
still making cooperation worth choosing. Ending at the viability line is what
puts a minority of defectors within reach of destroying the commons, which is
the difference between a dilemma and a world that quietly absorbs greed.

**Randomness.** One seeded draw: which cell a forager picks when several are
tied. Each agent has its own stream, so the order agents are processed in cannot
influence a run.

## Is it actually a dilemma?

The point of the world is the incentive structure, not the scripted policies —
once agents connect over MCP they decide for themselves, and the payoffs have to
do the teaching. So the structure is measured rather than assumed:

```bash
python harness.py --dilemma
```

For k defectors among four agents, at the tuned rate (`global`, r = 0.05):

| defectors | a defector earns | a cooperator earns | collapse % | total welfare |
|---|---|---|---|---|
| 0 | — | **15.8** | 0 | 63.1 |
| 1 | 37.9 | 6.9 | 10 | 58.7 |
| 2 | 17.2 | 5.9 | **68** | 46.0 |
| 3 | 11.3 | 5.5 | 88 | 39.5 |
| 4 | **8.7** | — | 100 | 34.9 |

Four conditions, all checked by that command:

- **T > R > P > S** — 37.9 > 15.8 > 8.7 > 6.9. The canonical dilemma ordering.
- **Defection dominates** at every mix, so aggression is where a self-interested
  agent starts. It should be: that is the whole tension.
- **Welfare falls with every extra defector**, so the group's best outcome is
  universal restraint.
- **Free-riding is unsafe** — half the group defecting destroys the commons 68%
  of the time (87% under the `neighbour` rule).

The arc that produces: defecting alone pays **2.4×** what cooperating pays, so
agents start greedy. But if everyone follows that reasoning the commons dies at
tick ~20 and each agent walks away with 8.7, where mutual restraint would have
paid **1.81×** that. And because two defectors are usually enough to kill it,
restraint has to be near-universal to work — which is what makes agreements
worth negotiating rather than a nicety.

**`plant()` costs what it gives.** Sowing seed you could have eaten is a
contribution, not free money. Without that cost, two planters out-produce the
entire regrowth rule, cooperators absorb any amount of greed, and a lone
defector actually *raises* total welfare — the world rewards greed and teaches
the opposite lesson.

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

Build it yourself with the two commands above -- the page is self-contained,
so opening `overgraze.html` from disk works with no server and no network.

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

## The MCP server

A thin shell over `world.py`. The tools identify the caller, call the engine,
and return what happened; there is no game logic in the protocol layer.

```bash
pip install -r requirements.txt
python server.py --new alice bob carol dave     # prints a bearer token per seat
python server.py                                # serve on 127.0.0.1:8000/mcp
python play.py --self-test                      # drive a whole run end to end
```

| tool | costs a tick? |
|---|---|
| `look_around`, `get_status`, `listen_for_messages`, `get_history`, `get_ledger` | no |
| `harvest`, `move`, `plant`, `punish`, `pass_turn` | yes — one per tick |
| `say` | no — speech is its own channel |

**The awkward part is the tick.** The engine resolves a whole tick at once, but
MCP calls arrive one agent at a time. So an action is an *intent*: it is written
down, the caller blocks, and when every seat has committed — or the barrier
times out and the stragglers are recorded as `noop` — the tick resolves for
everyone together and each caller is handed the part of the outcome that belongs
to it. A harvest can therefore return less than it asked for, which is the
honest answer when somebody else wanted the same cell.

**Nothing lives in an MCP session.** State is in SQLite, keyed by run, and a
tool call is answerable from a bearer token and the database alone. The
2026-07-28 spec removed protocol-level sessions, so the server is written as
though there is none — because there is not. A restart costs an in-flight
barrier wait, not a run.

**Auth is a bearer token per seat**, mapped to a player row. That is fine for a
demo over TLS and nothing more. OAuth 2.1 is the production answer.

Errors come back as results rather than protocol faults, because they are things
an agent has to reason about: *you already acted this tick*, *nothing left in
this cell*, *that would leave the world*, *that agent is out of range*.

## What is hard about this MCP server

Most MCP servers are stateless wrappers: a tool call arrives, an API is called,
a result comes back, nothing is shared between callers. This one is not that,
and the differences are where the engineering is.

**Four authenticated clients share one world.** Every seat has its own bearer
token mapping to a player row, and all four are connected at once to the same
run. A tool call is answerable from the token and the database alone.

**A tool call blocks on other clients.** The engine resolves a whole tick at
once; MCP calls arrive one client at a time. So an action is an *intent* — it
is written down, the caller blocks, and the tick resolves only when every seat
has committed, or a barrier times out and the stragglers are recorded as `noop`.
Each caller is then handed the part of the outcome that belongs to it. A harvest
returns **less than it asked for** when someone else wanted the same cell, which
is the honest answer rather than an error.

**No session state, by design.** The 2026-07-28 spec removed protocol-level
sessions, so the server is written as though there is none — because there is
not. World state is JSON in SQLite; a restart costs an in-flight barrier wait,
not a run. There is a test asserting a fresh connection sees the same world.

**Errors are content, not faults.** *You already acted this tick*, *nothing left
in this cell*, *that agent is out of range* come back as results, because they
are things a model has to read and reason about rather than exceptions for a
client library to swallow.

**The consequence for scale:** the barrier lives in process memory, so the
server runs as exactly one instance. Two replicas would resolve their own ticks
against the same database and clients would desynchronise silently. Moving the
barrier into the database is the prerequisite for scaling out — see the note in
the Dockerfile.

Built on `mcp` 2.0 (`MCPServer`, streamable HTTP), with `server.py` deliberately
thin: it identifies the caller, calls the engine, and returns what happened.
There is no game logic in the protocol layer, which is the whole point of having
built the engine first.

## Deploying it

```bash
docker build -t overgraze . && docker run -p 8000:8000 -v og:/data \
  -e OVERGRAZE_ADMIN_TOKEN=$(openssl rand -hex 16) overgraze
```

`fly.toml` and `render.yaml` are both ready; TLS and the public hostname are the
platform's job, so the app speaks plain HTTP behind them.

| variable | what it does |
|---|---|
| `OVERGRAZE_DB` | path to the SQLite file — point it at the mounted disk |
| `OVERGRAZE_ADMIN_TOKEN` | secret for `/admin/*`. **Unset means those routes refuse everything** |
| `OVERGRAZE_RATE` | tool calls per token per minute (default 120) |
| `PORT` | set by the platform; its presence also switches the bind to `0.0.0.0` |

| endpoint | |
|---|---|
| `GET /healthz` | liveness — touches the database, so a green check means something |
| `POST /admin/new` | start a run and mint its tokens, without redeploying |
| `POST /admin/reset` | wipe every run. Requires `{"confirm": "reset"}` in the body |
| `POST /mcp` | the server itself |

**Run exactly one instance.** The tick barrier that makes agents act
simultaneously lives in process memory. A second replica would resolve its own
ticks against the same database and agents would quietly desynchronise — the
worst kind of bug, because nothing would error. Scaling out means moving the
barrier into the database first, and until then `numInstances: 1` and
`auto_stop_machines = false` are load-bearing, not defaults. A machine that
sleeps mid-run strands every agent waiting at the barrier.

Rate limiting is per token, in memory, and exists to stop one client looping
`harvest()` a thousand times. It is not a billing meter and does not survive a
restart.

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
