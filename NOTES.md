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
| 4 — agents | **wiring done, never run for real.** `dispositions.py`, `llm_agents.py`, verified only with `--dry-run` |
| 5 — observatory | not started. An earlier visualiser exists (`viz_template.html`) but renders precomputed frames, not the event log |
| 6 — ablations | not started for LLM agents. Scripted equivalents are done (`harness.py --dilemma`, `evolve.py`) |
| 7 — write-up | not started |

## The one thing blocking everything

**No real model call has ever been made.** `ANTHROPIC_API_KEY` was never set in
the environment where this was built. Before anything downstream, run:

```bash
python llm_agents.py --ticks 5 --budget 0.50
```

Then read `traces.jsonl`. Everything in Phases 5–7 assumes the agents produce
sensible play, and that assumption is currently unverified. If the reasoning
reads as thoughtless, raise `--effort` from `low` before concluding the prompts
are wrong.

## Decisions that would otherwise get re-litigated

- **Nobody is expected to connect and play.** The goal is to *show results about
  how agents reason*, not to run a public game. That is why Phase 3 is built but
  not deployed, and why the observatory (Phase 5) matters more than hosting.
- **Deployment, if it ever happens:** Fly is the cheapest and does not need a
  GitHub remote; Render is the push-to-deploy option but needs one. A Cloudflare
  tunnel from a laptop satisfies the Phase 3 gate for free.
- **Model default is `claude-haiku-4-5`**, chosen deliberately for the ~16,000
  calls a full Phase 6 matrix implies. `--model claude-opus-5` for a showcase run.
- **40 ticks, not 100.** The world collapses around tick 20 under greed, so the
  whole arc fits in 40 at a fraction of the cost.
- **One server instance only.** The tick barrier is in process memory; two
  replicas desynchronise silently. See the Dockerfile.
- **The repo is local-only** — 27 commits, no remote. Pushing to GitHub is a
  decision about visibility that has not been made.

## Open questions

1. **Does the post lead with the game theory or the engineering?** They are
   different posts from the same repo, and the answer decides whether to invest
   in Phase 5 (observatory) or Phase 7 (connect-your-own-agent).
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
