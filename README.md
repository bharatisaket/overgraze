# Overgraze

A shared pasture, four language-model agents, and an MCP server between them.

Agents connect as ordinary MCP clients, graze a common resource, talk to each
other, form agreements, and can fine each other for breaking them. The engine is
a pure function; the protocol layer is a thin shell over it.

**Watch two real runs, side by side:**
https://claude.ai/code/artifact/8c8f2ffa-6f4f-41ef-be73-d03469f86335

Built with Claude Code. I designed the experiments, chose the parameters and
directed the work; the implementation is Claude's. The commit history reflects
that, and most of what is in `NOTES.md` came out of arguing with it.

---

## The finding

Four experimental conditions, 1,040 model decisions, `claude-sonnet-5`.

**No agent ever punished another. Not once.**

Not when breaches were delivered to them itemised (`agreed 0.3, took 0.55`).
Not when breaking a promise was fined at double rate. Not when the objective was
changed to *finish ahead of the other three*, which makes pulling a rival down
directly profitable. Across 260 decisions in that condition, not one mentions
leading, trailing or beating anybody.

Standard theory predicts exactly this: punishing costs you, and the benefit is a
public good shared with the person you punished. The well-known result is that
**humans violate that prediction** — Fehr and Gächter's costly punishment. These
agents did not.

Punishment appears twice in the whole study. Both times an agent held back its
own harvest to avoid being punished, by an enforcer that never existed.

Everything else that looked like a finding is in `NOTES.md`, including the two
that did not survive.

## Quick start

```bash
pip install -r requirements.txt

python -m unittest test_world test_store test_deploy   # 156 tests
python theory.py                                       # the benchmarks
python harness.py --dilemma                            # is it a real dilemma?
python demo_run.py --out demo.json                     # a run, no API cost
python play.py --self-test                             # drive it over MCP, free
```

With `ANTHROPIC_API_KEY` set, a real run of four model agents:

```bash
python llm_agents.py --dry-run                         # wiring check, free
python llm_agents.py --uniform --ticks 65 --budget 4.00 --monitoring local
python roles.py --trace traces.jsonl                   # who turned out to be what
python lies.py traces.jsonl                            # claims the ledger contradicts
```

**Always dry-run first.** It exercises MCP, the database and the tick barrier for
nothing. Skipping it once cost $0.55 on two runs that made every model call and
finished at tick zero.

## The world

| | |
|---|---|
| Field | 4×4 cells, 16 units at capacity, one per cell |
| Regrowth | logistic, `r = 0.15`. Peaks at half capacity: 0.6/tick at stock 8 |
| Take limit | 0.55/tick per agent. Four at full tilt draw 2.2 against 0.6 of regrowth |
| Upkeep | 0.08/tick, charged whatever you do |
| Collapse floor | 4.0, a quarter of capacity. Recorded, and survivable |
| Sight | the whole field. Only nearby *actions* are witnessed |
| Punishment | costs 0.2, fines 0.6, or **1.2** for breaking a pact you signed |
| Pacts | an object with a cap and a membership. Members pool what they have seen |

It was 6×6 with no upkeep. At that size the dilemma never happened: across four
measured runs, two agents stood on the same cell in 0%, 0%, 0% and 48% of ticks.
They were farming separate plots. A common-pool resource has to be rival, and
restraint has to cost something, or agents simply hoard.

**A viable pact cap sits between 0.08 and 0.15.** Below upkeep, honouring it
starves you. Above the per-agent share of maximum sustainable yield, the field
cannot carry it. That band is the most important number here.

## Is it actually a dilemma?

Measured, not assumed — `harness.py --dilemma` writes `payoffs.json`, and
`theory.py` refuses to run against a file measured under different constants.

```
T=25.1  R=8.7  P=3.6  S=2.2          strict T > R > P > S
grim-trigger threshold  δ ≥ 0.76
half the group defecting kills the commons 51% of the time
open access dies at tick 8; the tragedy costs 75% of the achievable
```

## What's in here

```
world.py        the engine. Pure, frozen dataclasses, no I/O
store.py        SQLite persistence and the simultaneous-tick barrier
server.py       the MCP layer. 15 tools, no game logic
llm_agents.py   four model agents over MCP, with a hard budget ceiling
dispositions.py the briefings. The experiment's independent variable
harness.py      scripted policies, sweeps, the dilemma check
theory.py       planner optimum, open access, the folk threshold
evolve.py       evolutionary tournaments over scripted strategies
roles.py        names each seat from conduct, after the run
lies.py         statements the ledger contradicts
demo_run.py     the real engine, scripted judgement, no API cost
play.py         drives a full run over MCP. The Phase 2 gate
results/        the four paid runs: traces, event logs, comparison
```

## Results

Four 65-tick runs, seed 1, local monitoring, same world.

| | proposed | joined | broken | fines | end stock |
|---|---|---|---|---|---|
| identical briefing | 3 | 1 | 14 | **0** | 6.05 |
| identical, rank objective | 2 | 2 | 16 | **0** | 6.34 |
| identical, neutral wording | 4 | 0 | 12 | **0** | 8.11 |
| assigned roles | 2 | 3 | 4 | **0** | 11.23 |

**Assigning roles produced a materially better commons** — a quarter of the
breaches and nearly double the resource surviving — and the briefings were
honoured: the steward took a mean of 0.123 against the maximizer's 0.300 and
finished last. The commons survived because two agents paid for it.

Full comparison and caveats in [`results/README.md`](results/README.md).

## Honest limits

- **One seed per condition.** A horizon result looked textbook on seed 1 and
  reversed exactly on seed 2. It is written up as a failed replication.
- **The reskin leaked.** `get_status` returns a field named `commons`, so agents
  saw that word every tick whatever the framing. Whether the coordination is
  derived or recognised is untested.
- **One server instance only.** The tick barrier is in process memory; two
  replicas desynchronise silently.
- **Bearer token per seat.** Fine for a closed demo, not for anything public.

## Cost

$19.02 across four paid runs, about $0.0143 per decision on Sonnet 5. Everything
that is not a model call — tests, `theory.py`, `harness.py`, `demo_run.py`,
`play.py --self-test`, every dry run — is free.

MIT.
