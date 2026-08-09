"""
Name the foragers after the run, from what they did.

Every result in this project so far has been reported against labels the agents
were given before they acted: an agent told it was a steward showed restraint,
and that was written up as restraint. It is circular. The disposition causes the
behaviour and then the behaviour is offered as evidence about the disposition.

So the labels move to the end. Four agents start indistinguishable, the run
happens, and this reads the event log and works out who turned out to be what.
A role here is a description of conduct with the evidence attached, not an
instruction anybody was given.

The tests are deliberately crude and mostly relative -- "took more than the
others", not "took more than 0.3" -- because absolute thresholds are another
way of deciding the answer in advance.

    python roles.py --events demo.json
    python roles.py --trace traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def tally(events: list[dict], n_agents: int | None = None) -> dict[int, dict]:
    """Per-agent conduct, counted straight off the event log."""
    who = set()
    t = defaultdict(lambda: defaultdict(float))
    for e in events:
        a = e.get("agent")
        if a is None:
            continue
        who.add(a)
        kind, typ = e.get("kind"), e.get("type")
        if typ == "action" and kind == "harvest":
            t[a]["harvests"] += 1
            t[a]["taken"] += float(e.get("granted") or 0.0)
        elif typ == "action" and kind == "plant":
            t[a]["planted"] += 1
        elif typ == "punish":
            t[a]["fines_thrown"] += 1
            if e.get("subject") is not None:
                t[e["subject"]]["fines_taken"] += 1
        elif typ == "pact":
            t[a][{"proposed": "proposals", "joined": "joins",
                  "left": "exits", "broken": "breaches"}.get(kind, "pact_other")] += 1
        elif typ == "speech":
            t[a]["spoke"] += 1
        elif typ == "reject" and kind == "punish":
            t[a]["fines_refused"] += 1
    for a in range(n_agents or 0):
        who.add(a)
    return {a: dict(t[a]) for a in sorted(who)}


def label(counts: dict[int, dict]) -> dict[int, dict]:
    """Assign each agent a role from its conduct, relative to the others.

    Order matters: the tests run from most to least specific, and the first that
    fits wins. An agent that both proposed a pact and broke it is a breaker --
    what somebody did with an agreement says more than that they made one.
    """
    if not counts:
        return {}
    n = len(counts)
    total_taken = sum(c.get("taken", 0.0) for c in counts.values()) or 1e-9
    avg_taken = total_taken / n
    fines = sum(c.get("fines_thrown", 0.0) for c in counts.values())
    plants = sum(c.get("planted", 0.0) for c in counts.values())

    out = {}
    for a, c in counts.items():
        took = c.get("taken", 0.0)
        share = took / total_taken
        why = []

        if c.get("breaches", 0) > 0:
            role = "breaker"
            why.append(f"broke an agreement {int(c['breaches'])}x")
        elif fines and c.get("fines_thrown", 0) >= max(1.0, fines / n * 1.5):
            role = "enforcer"
            why.append(f"threw {int(c['fines_thrown'])} of {int(fines)} fines")
        elif c.get("proposals", 0) > 0:
            role = "convener"
            why.append(f"proposed {int(c['proposals'])} agreement(s)")
        elif took > avg_taken * 1.25:
            role = "taker"
            why.append(f"took {share:.0%} of everything harvested")
        elif plants and c.get("planted", 0) >= max(1.0, plants / n * 1.5):
            role = "planter"
            why.append(f"planted {int(c['planted'])} of {int(plants)} times")
        elif took < avg_taken * 0.75:
            role = "abstainer"
            why.append(f"took {share:.0%}, well under an even share")
        elif c.get("joins", 0) > 0:
            role = "joiner"
            why.append(f"signed {int(c['joins'])} agreement(s), proposed none")
        else:
            role = "unremarkable"
            why.append("nothing stood out")

        if c.get("fines_refused", 0):
            why.append(f"{int(c['fines_refused'])} fines refused")
        if c.get("exits", 0):
            why.append(f"left a pact {int(c['exits'])}x")
        out[a] = {"role": role, "because": "; ".join(why), "counts": c}
    return out


def load_events(args) -> tuple[list[dict], list[str] | None]:
    if args.events:
        d = json.loads(Path(args.events).read_text(encoding="utf-8"))
        return d.get("events", []), d.get("names")
    rows = [json.loads(l) for l in
            Path(args.trace).read_text(encoding="utf-8").splitlines() if l.strip()]
    ev = []
    for r in rows:
        if r.get("type") != "decision":
            continue
        out = r.get("outcome") or {}
        for e in out.get("outcome", []) or []:
            ev.append(e)
    return ev, None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="work out who turned out to be what")
    p.add_argument("--events", help="a demo_run.py json")
    p.add_argument("--trace", help="a traces.jsonl from llm_agents.py")
    args = p.parse_args(argv)
    if not (args.events or args.trace):
        p.error("give --events or --trace")

    events, names = load_events(args)
    roles = label(tally(events))
    if not roles:
        print("no agent events found")
        return 1

    print(f"{len(events)} events\n")
    print(f"{'seat':<20}{'became':<14}{'on the evidence of'}")
    print("-" * 78)
    for a, r in roles.items():
        who = f"{a} ({names[a]})" if names and a < len(names) else str(a)
        print(f"{who:<20}{r['role']:<14}{r['because']}")
    print("\nRoles are descriptions of conduct, worked out after the fact.")
    print("Nobody was told to be any of these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
