# Working notes

Status and next steps, kept here so a fresh session can pick up without
re-deriving anything. The README describes the project; this file describes
where the work stopped and what it cost to learn.

## Where it stands

| Phase | State |
|---|---|
| 0 — decisions locked | done |
| 1 — world engine | done, and **rebuilt this session**: 4×4 board, upkeep, pacts, evidence-based punishment, survivable collapse. 155 tests |
| 2 — MCP layer | done. `server.py` + `store.py`, pact tools added, gate passes (`play.py --self-test`) |
| 3 — deploy | built, never deployed. Deliberate — nobody is expected to connect and play |
| 4 — agents | **run for real.** Six paid runs on Haiku, one in the rebuilt world. Sonnet 5 is the default and has never been run |
| 5 — observatory | done differently than planned: an isometric canvas playback driven by real run data, published as an artifact |
| 6 — ablations | horizon A/B done and **failed to replicate**. Monitoring × punishment is the live one now |
| 7 — write-up | drafts exist; the lead keeps changing as results land |

## The world, as it now stands

Every number here is measured from the engine, not assumed.

- **4×4 board, four agents.** It was 6×6 and the dilemma never happened: across
  four measured runs two agents stood on the same cell in 0%, 0%, 0% and 48% of
  ticks. They were farming separate plots, not sharing a commons. Not 3×3 —
  with `VISION = 1` an agent in the centre of a 3×3 sees everything, and
  unverifiable claims are the basis of deception under local monitoring.
- **`UPKEEP = 0.08` per tick, charged to everyone.** Without it restraint was
  free and agents hoarded: they held the field near capacity and harvested
  9–18% of what they were allowed, parking the commons far above the stock
  where it grows fastest.
- **`r = 0.15`.** At 0.05 upkeep drains 0.32/tick against an MSY of 0.20 and the
  world is unsurvivable by construction.
- **Payoffs: T=25.1 R=8.7 P=3.6 S=2.2**, folk threshold δ ≥ 0.76. All four
  dilemma checks pass; half the group defecting kills the commons 51% of the
  time; open access dies at tick 8; the tragedy costs 75% of the achievable.
- **A viable pact cap sits between 0.08 and 0.15.** Below upkeep, honouring it
  starves you. Above the per-agent MSY share, the field cannot carry it. At
  0.18 it collapses. This band is the single most important number in the file.

## Findings from the language-model agents

Weaker evidence than the scripted work: single runs, and all of it on
`claude-haiku-4-5`. Traces are gitignored.

- **Agents derive the folk theorem unprompted.** The maximizer, told only to
  maximise its own harvest, spent two ticks extracting — *"the commons is
  declining but that's not my concern"* — then pivoted: *"0.4 × 97 = 38.8
  future harvest vs. maybe 2-3 more maxed harvests before collapse… This is
  rational self-interest, not altruism—longer game = higher score."*
- **They use the pact tools, and they ratchet the cap down.** Given
  `propose_pact` for the first time, they proposed 0.40, then 0.25, then 0.15,
  joined and left repeatedly, and recorded only one breach in 21 ticks. Nobody
  told them to tighten it. Every cap they chose was above upkeep — they avoided
  the trap the scripted demo fell into twice.
- **Upkeep fixed the over-conservation.** 0.134 per agent per tick, which is
  **89% of the sustainable optimum**, against 9–18% of cap in every earlier
  run. This is the clearest single effect of any change made this session.
- **Agents bluff about sanctions they cannot apply.** With punish disabled they
  attempted it anyway, were refused every time, and announced sanctions in
  speech regardless — 3/7, 5/12, 1/6, 2/5 attempts/claims across four runs.
  Every run, both seeds, both conditions. The only fully replicated result.
- **Seeing the end does *not* change how they invest — that failed to
  replicate.** Plants by phase, early/mid/late: seed 1 gave 12/12/5 visible and
  19/19/23 hidden, textbook backward induction. Seed 2 reversed it exactly.
  Two seeds, opposite directions. Recorded because it had already been written
  up as a finding and the temptation to keep it was real.
- **The outcome difference held on both seeds, weakly.** Agents shown a
  countdown finished with a lower commons and took more. Two of two is what a
  coin does one time in four.
- **Haiku misreads rules under load.** One agent read the pasture's viability
  floor as a personal score target and played against it for the rest of the
  run. That is why the default model is now `claude-sonnet-5`.

## What was built this session

- **Pacts as objects.** `propose_pact` / `accept_pact` / `leave_pact` over MCP,
  with terms, membership and breach events carrying who witnessed them.
  Compliance became arithmetic instead of a reading of the transcript.
- **Punishment on by default, reach follows evidence.** You may fine someone
  you can see, or someone you witnessed within `PUNISH_MEMORY` ticks. Adjacency
  alone made enforcement impossible: the first real run in this world attempted
  nine punishments and every one was refused for range. **This makes monitoring
  decide whether a norm can be enforced at all**, which is Ostrom's claim and
  was previously unrepresented.
- **Survivable collapse.** `end_on_collapse=False` records the floor and lets
  the world continue. `SEED_BANK` adds recruitment scaled to how empty the
  field is — logistic growth is zero at zero, so without it a stripped commons
  is dead for ever and any run reaching the floor could only flatline. Measured:
  stock 16.0 → 2.18, floor crossed at tick 21, back to 14.08 by tick 80.
  **Default stays `True`** so the calibration and every existing test still
  measure the world they were written against.
- **`lies.py`** — audits a trace for statements the ledger contradicts: false
  self-reports, false compliance claims, phantom sanctions. Accusations are
  counted and deliberately **not** scored, because a wrong accusation may be an
  honest misreading and nothing outside the speaker distinguishes them.
- **`demo_run.py`** — the real engine driven by scripted policy, so the shape
  of an outcome can be inspected without paying for one.

## Costs, measured

Per call rises as the observation payload grows; pacts added roughly 10%.

| | per call | 40 ticks (160 calls) | 90 ticks (360) |
|---|---|---|---|
| Haiku 4.5 | $0.0048 | $0.76 | $1.72 |
| Sonnet 5 | $0.0143 | $2.29 | **$5.15** |

**Everything that is not a model call costs nothing** — tests, `theory.py`,
`harness --dilemma`, `evolve.py`, `demo_run.py`, `lies.py --demo`, and every
`--dry-run` gate. The only spend in this project has ever been model calls.

**Spent: $4.98 of $5**, across seven paid runs. The full 90-tick arc on Sonnet 5
needs a fresh budget.

## Procedure that was learned the expensive way

- **A `--dry-run` before any paid run.** It exercises MCP, the database and the
  tick barrier for free. Skipping it cost $0.55 on a pair of 40-tick runs that
  made all 320 model calls and finished at tick 0 with every action rejected.
- **Actions that fail twice running abort the run.** A broken run is not a bad
  decision.
- **Enumerate every channel a treatment has to cover.** The first horizon A/B
  was invalid because `apply_horizon` filtered `get_status` and not the status
  blob every *action* returns.
- **A measured number must not become a literal.** `theory.py` carried four
  payoff constants with a comment saying which command produced them; they
  survived a change of grid size, regrowth rate and the arrival of upkeep,
  still printed as measurements. `harness --dilemma` now writes `payoffs.json`
  with the constants it measured under, and `theory.py` refuses stale files.
- **Rivalry has to be constructed; it never appears on its own.** Three times
  now: nine cells per agent on the 6×6, foragers walking a fixed bearing into
  the east wall, and foragers parked on their starting corners farming one cell
  each. Every time it silently removed the contention the world exists to have.

## Open questions

1. **The live Sonnet 5 run.** Everything is built for it and it has never been
   made. ~$5.15 for the full arc.
2. **Monitoring × punishment is now the experiment worth running.** Enforcement
   reach is bounded by observation reach, so `local` vs `global` tests whether
   a norm can be policed at all. This replaces the horizon A/B, which died.
3. **Does the enforcer always finish last?** In the scripted run the negotiator
   threw all 17 fines and ended below the defector it was policing, while the
   two who let someone else enforce finished first — the second-order
   free-rider problem, unstaged. Whether language models reproduce it is the
   sharpest question this world can now ask.
4. The `sanctioner` result still rests on 8 seeds.
5. Nothing carries between runs. The greedy→ruin→cooperate arc now fits inside
   one episode thanks to survivable collapse, so cross-episode memory is
   optional rather than required — but it is untested.

## Published

- Isometric playback of a real run — grass drawn blade by blade, pact
  pennants, sickle swings, live scorecard: `claude.ai/code/artifact/c68000ac-9546-4e78-84cd-1b93d09239c5`
- Four measured game-theory results with the agents' behaviour beside them:
  `claude.ai/code/artifact/cfd9eab7-31d6-4961-9a96-259aaf949975`
- The deception audit, demonstrated on deliberately fabricated events:
  `claude.ai/code/artifact/304cb00c-d6e9-4909-b1e4-2857be5026bb`

Artifacts are private until shared from the page's own share menu.
