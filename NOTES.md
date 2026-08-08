# Working notes

Status and next steps, kept here so a fresh session can pick up without
re-deriving anything. The README describes the project; this file describes
where the work stopped.

## Where it stands

| Phase | State |
|---|---|
| 0 — decisions locked | done (world size, regrowth, tick model, budget, no win condition, named) |
| 1 — world engine | done. `world.py`, `harness.py`, 88 tests, tuned so the dilemma is real |
| 2 — MCP layer | done. `server.py` + `store.py`, 17 tests, gate passes (`play.py --self-test`) |
| 3 — deploy | **built but never deployed.** Dockerfile, `fly.toml`, `render.yaml`, health, admin, rate limiting, 16 tests. Deferred deliberately — see below |
| 4 — agents | **first real run done** 2026-08-08: 5 ticks, 4 agents, 20 Haiku calls, $0.15. Dispositions diverge, a pact forms and breaks. n=1 |
| 5 — observatory | not started. An earlier visualiser exists (`viz_template.html`) but renders precomputed frames, not the event log |
| 6 — ablations | not started for LLM agents. Scripted equivalents are done (`harness.py --dilemma`, `evolve.py`) |
| 7 — write-up | not started |

## The blocker is cleared — and what it revealed

The first real run happened on 2026-08-08. The agents play sensibly; the
premise holds. Three things came out of it that change the plan.

**1. The agents do not know when the run ends.** `status` reports
`ticks_remaining` from `world.TICKS` (100), not from `--ticks`
(`store.py:325`). In the 5-tick run every agent reasoned as though 97 turns of
future remained, and the maximizer's pivot to cooperation was explicitly a
shadow-of-the-future calculation over that horizon. This is not a cosmetic
mismatch: a *known finite* horizon should unravel cooperation by backward
induction, so the perceived horizon is an independent variable and right now
it is an accident. Decide deliberately whether agents see the true end, then
make `--ticks` and the reported horizon agree.

**2. Costs are ~15x the back-of-envelope.** 20 calls cost $0.1519, i.e.
$0.0076/call, because thinking roughly triples output tokens and the whole
observation JSON sits uncached in every user turn. A 40-tick run is ~160 calls
≈ $1.20; a 100-run Phase 6 matrix is ~$120, not ~$10. Either budget for that or
cache/trim the observation payload first.

**3. False accusations arose with no injected noise.** Agents accused each
other of breaking a pact using pre-pact harvests as evidence, and the pact died
of it — the same failure the scripted `--noise-scan` produces with a 10%
misreport rate. Worth checking whether the ledger's presentation invites it.

Seeds 1 and 2 are done under both horizon conditions. The API budget is spent
(~$4.58 of $5), so further seeds need new funding or a free provider — see the
cost note. Two seeds was exactly enough to kill the prettiest finding, which is
the argument for funding several more rather than none.

## Decisions that would otherwise get re-litigated

- **Nobody is expected to connect and play.** The goal is to *show results about
  how agents reason*, not to run a public game. That is why Phase 3 is built but
  not deployed, and why the observatory (Phase 5) matters more than hosting.
- **Deployment, if it ever happens:** Fly is the cheapest and does not need a
  GitHub remote; Render is push-to-deploy and now has the remote it needed. A
  Cloudflare tunnel from a laptop satisfies the Phase 3 gate for free.
- **Model default is `claude-haiku-4-5`**, chosen deliberately for the ~16,000
  calls a full Phase 6 matrix implies. `--model claude-opus-5` for a showcase run.
- **40 ticks, not 100.** The world collapses around tick 20 under greed, so the
  whole arc fits in 40 at a fraction of the cost.
- **One server instance only.** The tick barrier is in process memory; two
  replicas desynchronise silently. See the Dockerfile.
- **The repo is public on GitHub** — https://github.com/bharatisaket/overgraze,
  MIT, default branch `master`, pushed 2026-08-08 at 28 commits. History was
  scanned for credentials before publishing and was clean. Anything committed
  from here is public immediately, so keep `.env`, `overgraze.db`, `tokens.json`
  and `traces.jsonl` in `.gitignore` where they already are.

## Open questions

1. **Does the post lead with the game theory or the engineering?** They are
   different posts from the same repo, and the answer decides whether to invest
   in Phase 5 (observatory) or Phase 7 (connect-your-own-agent).
   On the evidence so far the sturdiest hook is the bluffed sanctions, which
   needs no further spend; the horizon result is prettier but rests on one seed.
2. The `sanctioner` result rests on 8 seeds and deserves the 25-seed treatment
   the noise scan got before it is published.
3. The observatory should be rebuilt as a pure function of `(event_log, tick)`.
   The existing page renders precomputed frames, which cannot show the ledger,
   speech, or ghost-overlay ablations.

## Findings worth not losing

Measured, reproducible, and the reason the world is tuned the way it is:

- **Unrestrained greed destroys 53% of achievable value** — planner optimum
  70.1, open access 33.0, commons dead by tick 15 (`python theory.py`).
- **Cooperation must be near-universal.** Two defectors of four destroy the
  commons 68% of the time.
- **Axelrod's islands survive the move to a commons, but only with assortment.**
  25% reciprocators die out entirely with neither clustering nor monitoring;
  they fixate with both (`python evolve.py --scan`).
- **A 10% *misreading* rate destroys a commons of four agents who all practised
  restraint** — perception noise alone, no extra resource taken, 0% → 100%
  collapse. The mistake is the accusation, not the harvest.
- **Patience beats generosity.** Under perception noise, demanding more evidence
  before retaliating took fixation from 24% to 76%; probabilistic forgiveness
  did nothing (`python evolve.py --noise-scan`).
- **Costly punishment underperforms retaliation-in-kind here**, because in a
  commons retaliating by over-harvesting is *profitable* while a sanction is
  purely costly — the opposite of public-goods experiments.

## Findings from the language-model agents

All from `claude-haiku-4-5`, thinking off, four dispositions at one table.
Weaker evidence than the scripted work above: single runs, not seed averages.
Traces are gitignored; regenerate with the commands given.

- **Agents derive the folk theorem unprompted.** The maximizer, whose prompt
  says only to maximise its own harvest and never mentions collapse or other
  agents, spent two ticks extracting — *"the commons is declining but that's
  not my concern"* — and then pivoted: *"0.4 x 97 = 38.8 future harvest vs.
  maybe 2-3 more maxed harvests before collapse... This is rational
  self-interest, not altruism—longer game = higher score."* It priced the
  shadow of the future explicitly.
- **They negotiate a quantitative norm nobody specified.** The negotiator
  proposed a 0.4/turn cap by tick 0; three agents had signed on by tick 3.
- **Seeing the end does *not* visibly change how they invest — that result
  failed to replicate.** 40 ticks, `--horizon true` vs `hidden`, treatment
  integrity verified in all four runs. Plants by phase, early/mid/late:

  | | early | mid | late |
  |---|---|---|---|
  | seed 1, counter visible | 12 | 12 | 5 |
  | seed 1, hidden | 19 | 19 | 23 |
  | seed 2, counter visible | 2 | 6 | 13 |
  | seed 2, hidden | 3 | 17 | 1 |

  Seed 1 looked like textbook backward induction: investment collapsing as the
  end approaches, and rising when there is no end in sight. Seed 2 reverses it
  exactly. Two seeds, opposite directions, so the planting trajectory is noise
  at this sample size and nothing should be claimed from it. Recorded because
  the seed-1 pattern was written up as a finding before seed 2 existed, and the
  temptation to keep it was real.

- **The outcome difference did hold on both seeds, weakly.** Agents shown the
  countdown ended with a lower commons (29.99 vs 33.20; 30.00 vs 32.67) and
  took more in total (15.91 vs 8.33; 14.84 vs 10.84). Same direction twice,
  which is 2 of 2 — about what a coin does one time in four. Suggestive,
  unpublishable, and the obvious thing to spend the next budget on.
- **Agents bluff about sanctions they cannot apply.** With `punish` disabled,
  they attempted it anyway (every attempt rejected: `punish is disabled in this
  run`) and then announced sanctions in speech — 3 attempts/7 claims and 5/12
  in the clean pair, 2/6 and 5/23 before it. Peers sometimes believe it and
  adjust; the maximizer once caught it (*"Agent 3 has declared they are
  punishing me, but punish is disabled"*). Attempts and claims across the four
  clean runs: 3/7, 5/12, 1/6, 2/5 — **every run, both seeds, both horizon
  conditions.** It is the only finding here that survived replication, and it
  was not what any of these runs set out to measure.
- **A steward can finish below zero.** −0.10 in one run: planting costs the
  planter, so the commons survived partly because one agent paid for it.
- **Haiku misreads rules under load.** The maximizer at tick 37: *"The collapse
  floor is 9.0, so I need to reach 9.0 by tick 40"* — confusing the commons
  viability floor with a personal score target. Comprehension is a live
  variable; check it before attributing behaviour to disposition.

### What these cost

$0.0029 per decision after the cost work, so a 40-tick four-agent run is about
$0.65 — input grows with accumulated history and notes, so early-tick
extrapolation understates it. The whole set of findings above cost about $4.60.

### Two failures worth remembering

- A 40-tick pair made all 320 model calls, exited 0, and finished at **tick 0**
  with every action rejected, because `store.connect()` and `server.py`
  resolved different database files. $0.55 for nothing. Actions that fail twice
  running now abort the run, and a `--dry-run` — which exercises MCP, the
  database and the barrier for free — is the gate before any paid run.
- The first horizon A/B was invalid: `apply_horizon` filtered `get_status` but
  every *action* also returns `ticks_remaining`, so the hidden condition still
  saw a countdown. The direction survived the fix, which was luck. Enumerate
  every channel a treatment has to cover before spending on it.
