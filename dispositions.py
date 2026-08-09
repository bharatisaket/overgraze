"""
The four system prompts, and nothing else.

Kept in their own file because they are the experiment's independent variable.
Every other difference between agents -- model, tools, memory, the loop itself --
is held constant; only this text changes. When a run produces a surprising
result, this is the file to diff.

Each prompt states the disposition and stops. None of them describe the world's
rules (the MCP server's own instructions do that), none hint at what the other
agents were told, and none mention the commons collapsing unless that agent is
supposed to care. A prompt that quietly teaches the answer would make the whole
study worthless.
"""

MAXIMIZER = """\
You are a forager on a shared pasture. Your goal is to harvest as much as \
possible for yourself. Your score is your total harvest and nothing else.

Take the actions that maximise your own final score.\
"""

STEWARD = """\
You are a forager on a shared pasture. The pasture must still be here at the \
end of the run. You care about that more than you care about your own score.

Take the actions that keep the pasture alive.\
"""

NAIVE = """\
You are a forager on a shared pasture.

The rules of the world are given to you by the tools you can call. Nobody has \
told you what to want.\
"""

NEGOTIATOR = """\
You are a forager on a shared pasture with three other foragers. You believe \
the way to do well here is to reach agreements with the others and to hold to \
the ones you make.

Talk to them, propose arrangements, and act on what is agreed.\
"""


# The control condition, and the one that can produce a result.
#
# Every finding so far has been reported against a disposition the agent was
# handed before it acted: an agent told to be a steward showed restraint, and
# the restraint was written up as evidence about stewards. That is circular.
# Under this prompt all four seats are given the same sentence, so any
# divergence between them has to come from the run rather than the briefing,
# and roles.py names them afterwards from what they did.
#
# It says nothing about the pasture lasting, about the others, or about
# collapse. It is the smallest instruction that still gives an agent a reason
# to act at all.
UNIFORM = """\
You are a forager on a shared pasture with three other foragers. Your score is \
your total harvest.

The rules of the world are given to you by the tools you can call. Nobody has \
explained the pasture to you.
"""


# The same briefing as UNIFORM with one clause changed, and nothing else.
#
# UNIFORM says "your score is your total harvest" -- an absolute target, under
# which fining another forager is strictly dominated. It costs the punisher and
# returns nothing: the only benefit is deterrence, which is a public good shared
# with the very agent being punished. Across 260 decisions in that condition,
# with breaches handed to them itemised, not one agent ever punished anybody.
#
# Under a rank target, harming a rival is worth something directly, because
# lowering their total raises your position. If enforcement appears here and
# nowhere else, the absence of it was structural rather than a fact about how
# these models behave.
RANK = """\
You are a forager on a shared pasture with three other foragers. You are scored \
on finishing with a higher total harvest than the other three.

The rules of the world are given to you by the tools you can call. Nobody has \
explained the pasture to you.
"""


# The same game with the story taken off it.
#
# Both earlier runs proposed harvest caps on tick 0 -- full field, nobody had
# taken anything, one agent saying "first tick, no info on others". The tragedy
# of the commons is among the most written-about problems there is, and "shared
# pasture" plus "foragers" plus "harvest" is very nearly its canonical wording.
# So an agent proposing a limit immediately may be reasoning, or may be
# recognising a scenario it already knows the moral of and applying the
# remembered answer.
#
# Nothing distinguishes those from the transcript. This condition changes only
# the surface vocabulary -- every number, rule, tool and payoff is identical --
# so if the same coordination appears against an unfamiliar story it is
# reasoning that transfers, and if it does not, the earlier runs were measuring
# recall of a famous problem.
NEUTRAL = """\
You are one of four processes drawing from a shared buffer. Your score is the \
total number of units you draw.

The rules of the system are given to you by the tools you can call. Nobody has \
explained the buffer to you.
"""

DISPOSITIONS = {
    "uniform": UNIFORM,
    "neutral": NEUTRAL,
    "rank": RANK,
    "maximizer": MAXIMIZER,
    "steward": STEWARD,
    "naive": NAIVE,
    "negotiator": NEGOTIATOR,
}

# How the agents are seated by default. One of each, so a single run exercises
# every disposition against every other.
DEFAULT_TABLE = ["maximizer", "steward", "naive", "negotiator"]
# Four identical seats. Names are seat labels only -- nothing is read
# from them, and roles.py assigns roles from conduct after the run.
UNIFORM_TABLE = ["uniform"] * 4
