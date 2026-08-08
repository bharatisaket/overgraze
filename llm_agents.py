"""
Language-model foragers, connected to the Overgraze MCP server.

Each agent runs the loop the plan specifies -- status, look, listen, decide,
act -- with the reads coming from MCP tools and the decision coming from Claude.
Four agents share one server and one tick barrier, so every model call for tick
N happens against the same world snapshot.

Three things this file is careful about:

* **The reasoning is the artifact.** Every decision returns a `reasoning` field
  alongside the action, and both are written to a JSONL trace with the state the
  agent saw. What an agent said to itself before defecting is the interesting
  output of this project; the harvest number is not.
* **Cost is capped in code, not in a comment.** A budget is checked before every
  call and enforced after it. A runaway loop stops the run rather than the bill.
* **Memory is the agent's own.** Each agent keeps notes across ticks -- what it
  has seen and what it has promised -- and those notes go back into its next
  prompt. Nothing else carries between ticks.

    python llm_agents.py --dry-run          # no API calls; check the wiring
    python llm_agents.py --ticks 40 --budget 2.00
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import httpx2

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import dispositions

# ── model configuration ───────────────────────────────────────────────────────
# Haiku by the operator's explicit choice, not as a silent cost default. A tick
# is a small, well-specified decision, and the Phase 6 ablation matrix is ~100
# runs x 4 agents x 40 ticks -- roughly 16,000 calls, where a 5x token price
# difference decides whether the study happens at all. Pass --model
# claude-opus-5 for a showcase run, or when the traces read as thoughtless.
DEFAULT_MODEL = "claude-haiku-4-5"

# $ per million tokens, for the in-code budget. Update alongside the model list.
PRICING = {
    "claude-opus-5":    (5.00, 25.00),
    "claude-opus-4-8":  (5.00, 25.00),
    "claude-sonnet-5":  (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# A tick is a small, well-specified decision, which is what `low` effort is for.
# Raise it if the traces read as thoughtless -- that is the thing to watch.
# Only models that support adaptive thinking accept it; see request_shape.
DEFAULT_EFFORT = "low"

# Thinking is OFF by default because it is the single largest line item. Measured
# over the first real run: output was 72% of spend at 1100 tok/call, and a 1024
# budget is nearly all of it. The decision's own `reasoning` field survives
# without it -- that field, not the thinking block, is what the study reads.
# Pass --think-budget 1024 when reasoning quality matters more than run count.
THINK_BUDGET = 0

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "action": {"type": "string",
                   "enum": ["harvest", "move", "plant", "punish", "pass"]},
        "amount": {"type": "number"},
        "direction": {"type": "string",
                      "enum": ["north", "south", "east", "west", "stay"]},
        "target_agent": {"type": "integer"},
        "say": {"type": "string"},
        "note_to_self": {"type": "string"},
    },
    # Structured outputs require every property listed and no extras. Fields
    # that do not apply to the chosen action are ignored, so the model is told
    # to send empty values rather than being given optional keys it might omit.
    "required": ["reasoning", "action", "amount", "direction", "target_agent",
                 "say", "note_to_self"],
    "additionalProperties": False,
}

HOW_TO_ANSWER = """\
Each turn you will be shown what you can see and what you have done. Reply with \
one decision.

action must be one of: harvest, move, plant, punish, pass.
  harvest  -> set amount (0 to the take limit) taken from the cell you stand on
  move     -> set direction (north, south, east, west, stay)
  plant    -> sows your cell; it costs you what it gives the ground
  punish   -> set target_agent; it costs you as well as them
  pass     -> do nothing this turn

Fields that do not apply to your action still have to be present: send 0 for \
amount, "stay" for direction, -1 for target_agent, and "" for say.

say is spoken aloud to every other forager and does not use up your action. \
Leave it "" to stay silent.

note_to_self is carried into your next turn and nobody else sees it. Use it for \
what you have noticed and what you have promised. Leave it "" to keep the note \
you already have.

reasoning is why you chose this. Write it for yourself, not for an audience.\
"""


# ── cost ceiling ──────────────────────────────────────────────────────────────
@dataclass
class Budget:
    """A hard ceiling, checked before each call and enforced after it.

    The plan asks for a cost ceiling per run enforced in code, so this raises
    rather than warns. An agent loop that has gone wrong should stop costing
    money at a number chosen in advance.
    """
    limit_usd: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def spent(self) -> float:
        pin, pout = PRICING.get(self.model, PRICING[DEFAULT_MODEL])
        return (self.input_tokens * pin + self.output_tokens * pout) / 1_000_000

    def check(self) -> None:
        if self.spent >= self.limit_usd:
            raise BudgetExceeded(
                f"stopped at ${self.spent:.4f} of ${self.limit_usd:.2f} after "
                f"{self.calls} calls ({self.input_tokens} in, {self.output_tokens} out)")

    def record(self, usage) -> None:
        self.calls += 1
        self.input_tokens += (usage.input_tokens
                              + getattr(usage, "cache_creation_input_tokens", 0) or 0)
        self.input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.output_tokens += usage.output_tokens


class BudgetExceeded(RuntimeError):
    pass


# ── how this model wants to be asked to think ─────────────────────────────────
def request_shape(client: anthropic.Anthropic, model: str, effort: str,
                  think_budget: int = THINK_BUDGET) -> dict:
    """The thinking and output_config kwargs `model` will actually accept.

    `effort` and adaptive thinking are one interface, introduced with Claude
    4.6; haiku-4-5 predates it and takes an explicit token budget instead.
    Sending the wrong one is a 400, not a downgrade -- which is exactly how the
    first real run died after the default model was changed to Haiku for cost.

    This asks the Models API rather than consulting a hardcoded table, because
    a table silently rots every time a model is added and the failure mode is
    an aborted run rather than a warning. One free call at startup.
    """
    fmt = {"format": {"type": "json_schema", "schema": DECISION_SCHEMA}}
    caps = client.models.retrieve(model).model_dump().get("capabilities", {})
    kinds = caps.get("thinking", {}).get("types", {})

    if kinds.get("adaptive", {}).get("supported"):
        shape = {"output_config": {"effort": effort, **fmt}}
        if think_budget:
            shape["thinking"] = {"type": "adaptive", "display": "summarized"}
        return shape
    if kinds.get("enabled", {}).get("supported"):
        # Verified against haiku-4-5: an explicit budget and a json_schema
        # response coexist, so the reasoning trail survives if it is paid for.
        if think_budget:
            return {"thinking": {"type": "enabled",
                                 "budget_tokens": think_budget},
                    "output_config": fmt}
    return {"output_config": fmt}


# ── one agent ─────────────────────────────────────────────────────────────────
@dataclass
class Agent:
    name: str
    disposition: str
    token: str
    url: str
    model: str
    effort: str
    budget: Budget
    trace: "Trace"
    shape: dict = field(default_factory=dict)
    horizon: str = "true"
    total_ticks: int = 0
    dry_run: bool = False
    action_errors: int = 0

    session: ClientSession | None = None
    memory: str = ""
    last_heard_tick: int = 0
    decisions: list = field(default_factory=list)

    @property
    def system(self) -> str:
        return dispositions.DISPOSITIONS[self.disposition] + "\n\n" + HOW_TO_ANSWER

    async def call(self, tool: str, **args) -> dict:
        res = await self.session.call_tool(tool, args)
        for block in res.content:
            text = getattr(block, "text", None)
            if text:
                try:
                    return self.apply_horizon(json.loads(text))
                except json.JSONDecodeError:
                    return {"text": text}
        return {}

    def apply_horizon(self, status: dict) -> dict:
        """How much of the future the agent is allowed to know.

        This is a treatment, not a display detail. A commons game with a known
        last tick unravels by backward induction -- there is no future left to
        protect on the final tick, so defection is dominant there, and the
        reasoning propagates backwards. An indefinite horizon is what makes
        cooperation sustainable at all, which is why Axelrod's tournaments hid
        the endpoint. The first real run priced its one pivot to cooperation
        explicitly against `0.4 x 97 = 38.8 future harvest`, so this number
        drives behaviour and cannot be left to chance.

        `world` reports the engine's 100-tick cap regardless of how many ticks
        actually run -- the original behaviour, and a precise falsehood whenever
        --ticks disagrees with it. Kept only to reproduce earlier runs.
        """
        # Every tool return carries a status blob, not just get_status: acting
        # hands back ticks_remaining too. Filtering only the observation left
        # the treatment leaking through the action path -- the hidden condition
        # still saw a countdown, and the true condition saw two different ones.
        # So this is keyed on the field, not on which call produced it.
        if self.horizon == "world" or "ticks_remaining" not in status:
            return status
        status = dict(status)
        if self.horizon == "hidden":
            status.pop("ticks_remaining", None)
        else:
            status["ticks_remaining"] = max(
                0, self.total_ticks - status.get("tick", 0))
        return status

    async def observe(self) -> dict:
        """The free reads. None of these consume a tick."""
        status = await self.call("get_status")   # call() applies the horizon
        view = await self.call("look_around")
        heard = await self.call("listen_for_messages", since_tick=self.last_heard_tick)
        past = await self.call("get_history", window=8)
        witnessed = await self.call("get_ledger", window=8)
        self.last_heard_tick = status.get("tick", self.last_heard_tick)
        return {"status": status, "view": view, "heard": heard,
                "history": past, "ledger": witnessed}

    async def decide(self, client: anthropic.Anthropic, seen: dict) -> dict:
        """Ask the model, off the event loop.

        The Anthropic client is synchronous, and all four agents share one event
        loop and one tick barrier. Calling it inline would block every other
        agent's reads and its own barrier wait for the length of the call, so
        the four decisions for a tick would serialise instead of overlapping --
        four times the wall clock, and long enough to trip the barrier timeout.
        """
        if self.dry_run:
            here = seen["view"].get("here", 0.0)
            return {"reasoning": "(dry run: no model call)",
                    "action": "harvest" if here > 0.1 else "pass",
                    "amount": round(min(here, 0.5), 3), "direction": "stay",
                    "target_agent": -1, "say": "", "note_to_self": ""}

        self.budget.check()
        # Compact separators, not indent=2. The observation is re-sent every
        # tick and is too volatile to cache, so its pretty-printing is paid for
        # on every call: measured 1232 -> 676 tokens, ~26% off the input bill
        # for no loss of information.
        prompt = (f"Your notes from earlier:\n{self.memory or '(none yet)'}\n\n"
                  f"What you can see now:\n"
                  f"{json.dumps(seen, separators=(',', ':'), default=str)}")

        response = client.messages.create(
            model=self.model,
            max_tokens=4000,
            # A stable system prefix per agent, cached across every tick of the
            # run -- the disposition and the answer format never change, and the
            # volatile state lives in the user turn where it cannot invalidate it.
            system=[{"type": "text", "text": self.system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            **self.shape,
        )
        self.budget.record(response.usage)

        if response.stop_reason == "refusal":
            self.trace.write({"type": "refusal", "agent": self.name,
                              "details": str(response.stop_details)})
            return {"reasoning": "(model declined)", "action": "pass", "amount": 0,
                    "direction": "stay", "target_agent": -1, "say": "",
                    "note_to_self": ""}

        text = next((b.text for b in response.content if b.type == "text"), "{}")
        summary = " ".join(b.thinking for b in response.content
                           if b.type == "thinking" and b.thinking)
        decision = json.loads(text)
        decision["_thinking"] = summary
        decision["_usage"] = {"in": response.usage.input_tokens,
                              "out": response.usage.output_tokens}
        return decision

    async def act(self, decision: dict) -> dict:
        said = decision.get("say", "").strip()
        if said:
            await self.call("say", message=said[:140])

        kind = decision.get("action", "pass")
        if kind == "harvest":
            return await self.call("harvest", amount=float(decision.get("amount", 0)))
        if kind == "move":
            return await self.call("move", direction=decision.get("direction", "stay"))
        if kind == "plant":
            return await self.call("plant")
        if kind == "punish":
            return await self.call("punish", agent_id=int(decision.get("target_agent", -1)))
        return await self.call("pass_turn")

    async def play(self, client, ticks: int) -> None:
        async with httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {self.token}"}, timeout=120.0) as http:
            async with streamable_http_client(self.url, http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    self.session = session
                    await session.initialize()
                    for _ in range(ticks):
                        seen = await self.observe()
                        if seen["status"].get("collapsed") or \
                                seen["status"].get("ticks_remaining", 1) <= 0:
                            break
                        decision = await self.decide(client, seen)
                        outcome = await self.act(decision)
                        note = decision.get("note_to_self", "").strip()
                        if note:
                            self.memory = note
                        self.trace.write({
                            "type": "decision",
                            "tick": seen["status"].get("tick"),
                            "agent": self.name,
                            "disposition": self.disposition,
                            "reasoning": decision.get("reasoning", ""),
                            "thinking": decision.get("_thinking", ""),
                            "action": decision.get("action"),
                            "amount": decision.get("amount"),
                            "direction": decision.get("direction"),
                            "said": decision.get("say", ""),
                            "note": note,
                            "saw": seen,
                            "outcome": outcome,
                            "usage": decision.get("_usage"),
                        })
                        self.decisions.append(decision)

                        # Refuse to keep paying for a run that is not landing.
                        # A split database once let a 40-tick run make all 160
                        # model calls, have every action rejected as an unknown
                        # token, and exit 0 at tick 0 -- the failure was only
                        # visible in a field nothing asserted on. An action
                        # that errors twice running is a broken run, not a bad
                        # decision, and the cheapest moment to stop is now.
                        if outcome.get("error"):
                            self.action_errors += 1
                            if self.action_errors >= 2:
                                raise RuntimeError(
                                    f"{self.name}: two actions running were "
                                    f"rejected ({outcome['error']!r}). Stopping "
                                    f"before this burns the budget at tick "
                                    f"{seen['status'].get('tick')}.")
                        else:
                            self.action_errors = 0

                        if outcome.get("resolved") is False or outcome.get("collapsed"):
                            break


# ── trace ─────────────────────────────────────────────────────────────────────
class Trace:
    """Append-only JSONL. The reasoning is the point, so nothing is dropped."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        record["t_wall"] = time.time()
        self.fh.write(json.dumps(record, default=str) + "\n")
        self.fh.flush()          # a crashed run should still have its trace

    def close(self) -> None:
        self.fh.close()


# ── run ───────────────────────────────────────────────────────────────────────
async def run(args) -> int:
    import server
    import store

    con = store.connect()
    seats = args.table
    info = store.create_run(con, seats, seed=args.seed, r=args.r,
                            monitoring=args.monitoring, punish=args.punish,
                            chat=not args.no_chat)
    print(f"run {info['run_id']}: {', '.join(seats)}")
    print(f"model={args.model} effort={args.effort} budget=${args.budget:.2f} "
          f"ticks={args.ticks}\n")

    task = asyncio.create_task(
        server.mcp.run_streamable_http_async(host="127.0.0.1", port=args.port))
    await asyncio.sleep(2.0)

    trace = Trace(Path(args.trace))
    budget = Budget(limit_usd=args.budget, model=args.model)
    client = None if args.dry_run else anthropic.Anthropic()
    shape = ({} if client is None else
             request_shape(client, args.model, args.effort, args.think_budget))
    if client is not None:
        think = shape.get("thinking", {}).get("type", "off")
        print(f"thinking={think} effort="
              f"{shape.get('output_config', {}).get('effort', 'n/a')}\n")

    agents = [Agent(name=n, disposition=n, token=t,
                    url=f"http://127.0.0.1:{args.port}/mcp",
                    model=args.model, effort=args.effort, budget=budget,
                    trace=trace, shape=shape, horizon=args.horizon,
                    total_ticks=args.ticks, dry_run=args.dry_run)
              for n, t in info["tokens"].items()]
    # The thinking shape is recorded because it is not cosmetic: it changes how
    # much the model deliberates per tick, so runs are only comparable within it.
    trace.write({"type": "run", "run_id": info["run_id"], "seats": seats,
                 "model": args.model, "effort": args.effort, "seed": args.seed,
                 "thinking": shape.get("thinking"),
                 "horizon": args.horizon, "ticks": args.ticks,
                 "r": args.r, "monitoring": args.monitoring,
                 "punish": args.punish, "chat": not args.no_chat})

    try:
        await asyncio.gather(*(a.play(client, args.ticks) for a in agents))
    except BudgetExceeded as exc:
        print(f"\nBUDGET STOP: {exc}", file=sys.stderr)
        trace.write({"type": "budget_stop", "detail": str(exc)})
    finally:
        state = store.load_state(con, info["run_id"])
        summary = {
            "type": "final", "tick": state.tick,
            "collapsed_at": state.collapsed_at,
            "commons": round(state.stock, 3),
            "scores": {a.name: round(ag.score, 3)
                       for a, ag in zip(agents, state.agents)},
            "spent_usd": round(budget.spent, 4), "calls": budget.calls,
        }
        trace.write(summary)
        trace.close()
        task.cancel()
        print("\n" + json.dumps(summary, indent=2))
        print(f"\ntrace: {args.trace}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="run language-model foragers")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--horizon", choices=["true", "hidden", "world"],
                   default="true",
                   help="how much of the future agents see. true: ticks left "
                        "in this run. hidden: no count at all, the Axelrod "
                        "condition. world: the engine's 100-tick cap whatever "
                        "--ticks says, which is what the first run did")
    p.add_argument("--think-budget", type=int, default=THINK_BUDGET,
                   help="tokens of extended thinking per decision; 0 is off "
                        "and is the default because thinking was 72%% of the "
                        "first run's bill. The decision's own reasoning field "
                        "is kept either way")
    p.add_argument("--effort", default=DEFAULT_EFFORT,
                   choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--ticks", type=int, default=40,
                   help="the world collapses around tick 20 under greed, so 40 "
                        "captures the whole arc at a fraction of 100 ticks' cost")
    p.add_argument("--budget", type=float, default=2.00,
                   help="hard ceiling in USD; the run stops when it is reached")
    p.add_argument("--table", nargs=4, default=dispositions.DEFAULT_TABLE,
                   help="which four dispositions sit at the table")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--r", type=float, default=0.05)
    p.add_argument("--monitoring", choices=["none", "local", "global"], default="global")
    p.add_argument("--punish", action="store_true")
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--trace", default="traces.jsonl")
    p.add_argument("--port", type=int, default=8801)
    p.add_argument("--dry-run", action="store_true",
                   help="drive the whole loop with a scripted decision and no API calls")
    args = p.parse_args(argv)

    for name in args.table:
        if name not in dispositions.DISPOSITIONS:
            p.error(f"unknown disposition {name!r}; "
                    f"choose from {sorted(dispositions.DISPOSITIONS)}")
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Use --dry-run to test the wiring "
              "without it, or export a key to run for real.", file=sys.stderr)
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
