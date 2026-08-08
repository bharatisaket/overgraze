"""
Export simulation data for the visualiser.

Runs the Phase 1 engine (`world.py`) through the scripted harness and records
every tick, then writes viz_data.json for build_viz.py to embed.

Nothing here reimplements the rules: episodes come from `harness.run_episode`,
which drives `world.apply_actions`. The picture cannot drift from the engine.

    python export_viz.py && python build_viz.py
"""

import base64
import json

import numpy as np

import harness
from world import CAP, N, TAKE, TICKS

REPLAY_SEED = 0   # the single run shown in the grid player

# (data key, label, harness mix name) -- the data keys are what the page's
# colour map keys off, so they stay stable across engine changes.
MIXES = [
    ("greedy4",   "4 greedy",              "greedy"),
    ("cautious4", "4 cautious",            "cautious"),
    ("mixed",     "2 greedy / 2 cautious", "mixed"),
]

RULES = ["global", "neighbour"]


def encode_grid(g):
    """36 cells -> 36 bytes -> base64, to keep the embedded payload small."""
    b = np.clip(np.round(g / CAP * 255), 0, 255).astype(np.uint8).ravel().tobytes()
    return base64.b64encode(b).decode("ascii")


def encode_pos(p):
    """Agent positions as a flat digit string, e.g. '00501554'. N<10 so 1 digit each."""
    return "".join(f"{y}{x}" for y, x in p)


data = {
    "n": N, "cap": CAP, "ticks": TICKS, "take": TAKE, "seeds": harness.SEEDS,
    "replaySeed": REPLAY_SEED,
    "rValues": harness.SWEEP_R,
    "rules": RULES,
    "mixes": [{"key": k, "label": lab, "kinds": harness.MIXES[m]}
              for k, lab, m in MIXES],
    "sweep": {},
    "runs": {},
}

for rule in RULES:
    data["sweep"][rule] = {}
    for r in harness.SWEEP_R:
        rk = str(r)
        data["sweep"][rule][rk] = {}
        for key, label, mix in MIXES:
            eps = [harness.run_episode(s, mix, rule, r) for s in range(harness.SEEDS)]
            sv = np.array([e.survived for e in eps], dtype=float)
            hv = np.array([e.harvest for e in eps], dtype=float)
            per = np.array([e.scores for e in eps], dtype=float)

            entry = {
                "survMean": round(float(sv.mean()), 2),
                "survSd":   round(float(sv.std()), 2),
                "harvMean": round(float(hv.mean()), 2),
                "harvSd":   round(float(hv.std()), 2),
                "collapseRate": round(float((sv < TICKS).mean()), 3),
                "contested": round(float(np.mean([e.contested for e in eps])), 1),
            }
            if key == "mixed":
                entry["greedyEach"] = round(float(per[:, :2].mean()), 2)
                entry["cautiousEach"] = round(float(per[:, 2:].mean()), 2)
            data["sweep"][rule][rk][key] = entry

            ep = harness.run_episode(REPLAY_SEED, mix, rule, r, keep_frames=True)
            data["runs"][f"{rule}|{rk}|{key}"] = {
                "survived": ep.survived,
                "scores": [round(float(v), 2) for v in ep.scores],
                "frames": [encode_grid(g) for g in ep.frames],
                "pos": [encode_pos(p) for p in ep.positions],
                "cum": [[round(float(v), 2) for v in row] for row in ep.cum],
            }

with open("viz_data.json", "w") as f:
    json.dump(data, f, separators=(",", ":"))

n_frames = sum(len(v["frames"]) for v in data["runs"].values())
print(f"wrote viz_data.json: {len(data['runs'])} runs, {n_frames} frames "
      f"(engine: world.py, simultaneous ticks)")
