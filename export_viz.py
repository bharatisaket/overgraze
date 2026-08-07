"""
Export simulation data for the HTML visualiser.

Replays the same simulation compare.run() does -- importing Agent, regrow and
the constants from compare so the rules live in exactly one place -- but
records the grid and agent positions at every tick instead of only the totals.
Writes viz_data.json, which gets embedded into the visualiser page.

The recorded loop below must stay a faithful copy of compare.run(): same rng
draw order (permutation, then one draw per agent action), same regrow-then-
check-collapse ordering. If run() changes, change this too.
"""

import json
import base64
import numpy as np

from compare import Agent, regrow, N, CAP, TICKS, TAKE, R_VALUES

SEEDS = 40        # for the summary statistics
REPLAY_SEED = 0   # the single run shown in the grid player

MIXES = [
    ("greedy4",   "4 greedy",              ["greedy"]*4),
    ("cautious4", "4 cautious",            ["cautious"]*4),
    ("mixed",     "2 greedy / 2 cautious", ["greedy", "greedy", "cautious", "cautious"]),
]


def run_recorded(rule, r, mix, seed):
    """compare.run(), but keeping every intermediate grid and agent position."""
    rng = np.random.default_rng(seed)
    g = np.full((N, N), CAP)
    starts = [(0, 0), (5, 0), (0, 5), (5, 5)]
    ags = [Agent(k, sx, sy, rng) for k, (sx, sy) in zip(mix, starts)]

    grids = [g.copy()]
    poss = [[(a.y, a.x) for a in ags]]
    scores = [[0.0]*len(ags)]

    survived = TICKS
    for t in range(TICKS):
        for i in rng.permutation(len(ags)):
            ags[i].act(g)
        g = regrow(g, rule, r)
        grids.append(g.copy())
        poss.append([(a.y, a.x) for a in ags])
        scores.append([a.score for a in ags])
        if g.sum() < 0.05*N*N and survived == TICKS:
            survived = t
            break
    return survived, [a.score for a in ags], grids, poss, scores


def encode_grid(g):
    """36 cells -> 36 bytes -> base64, to keep the embedded payload small."""
    b = np.clip(np.round(g/CAP*255), 0, 255).astype(np.uint8).ravel().tobytes()
    return base64.b64encode(b).decode('ascii')


def encode_pos(p):
    """Agent positions as a flat digit string, e.g. '00501554'. N<10 so 1 digit each."""
    return "".join(f"{y}{x}" for y, x in p)


data = {
    "n": N, "cap": CAP, "ticks": TICKS, "take": TAKE, "seeds": SEEDS,
    "replaySeed": REPLAY_SEED,
    "rValues": R_VALUES,
    "rules": ["global", "neighbour"],
    "mixes": [{"key": k, "label": lab, "kinds": kinds} for k, lab, kinds in MIXES],
    "sweep": {},
    "runs": {},
}

for rule in data["rules"]:
    data["sweep"][rule] = {}
    for r in R_VALUES:
        rk = str(r)
        data["sweep"][rule][rk] = {}
        for key, label, mix in MIXES:
            res = [run_recorded(rule, r, mix, s) for s in range(SEEDS)]
            sv = np.array([x[0] for x in res], dtype=float)
            hv = np.array([sum(x[1]) for x in res], dtype=float)
            per = np.array([x[1] for x in res], dtype=float)

            entry = {
                "survMean": round(float(sv.mean()), 2),
                "survSd":   round(float(sv.std()), 2),
                "harvMean": round(float(hv.mean()), 2),
                "harvSd":   round(float(hv.std()), 2),
                "collapseRate": round(float((sv < TICKS).mean()), 3),
            }
            if key == "mixed":
                entry["greedyEach"] = round(float(per[:, :2].mean()), 2)
                entry["cautiousEach"] = round(float(per[:, 2:].mean()), 2)
            data["sweep"][rule][rk][key] = entry

            # One recorded run per configuration for the grid player.
            sur, sc, grids, poss, scores = run_recorded(rule, r, mix, REPLAY_SEED)
            data["runs"][f"{rule}|{rk}|{key}"] = {
                "survived": sur,
                "scores": [round(float(v), 2) for v in sc],
                "frames": [encode_grid(x) for x in grids],
                "pos": [encode_pos(p) for p in poss],
                "cum": [[round(float(v), 2) for v in row] for row in scores],
            }

with open("viz_data.json", "w") as f:
    json.dump(data, f, separators=(",", ":"))

n_frames = sum(len(v["frames"]) for v in data["runs"].values())
print(f"wrote viz_data.json: {len(data['runs'])} runs, {n_frames} frames")
