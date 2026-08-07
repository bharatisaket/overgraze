"""
Common-pool resource simulation.

Four agents forage on a 6x6 grid of renewable resource cells. Each tick every
agent moves to the richest cell it can reach and harvests from it, then the
whole grid regrows. The question the sweep at the bottom asks is whether a
group of self-interested ("greedy") agents ends up better or worse off than a
group of restrained ("cautious") ones -- i.e. whether this is a tragedy of the
commons, and at which regrowth rates.
"""

import numpy as np
from itertools import product

N = 6        # grid is N x N cells
CAP = 1.0    # max resource a single cell can hold
TICKS = 100  # length of one simulation run
TAKE = 0.55  # max an agent can harvest in a single tick

# Regrowth rates to sweep. payoff.py imports this rather than keeping its own
# copy, so the two scripts always report on the same rates.
R_VALUES = [0.11, 0.13, 0.15, 0.30, 0.50]


def neighbours_mean(g):
    """Average resource level in each cell's 3x3 neighbourhood (including itself).

    Pads the grid with a border of zeros, then sums the nine shifted copies of
    the grid (the eight neighbours plus the centre) and divides by nine.

    Note: the divisor is always 9, even for edge and corner cells whose
    neighbourhood is partly the zero padding. So edge cells report a lower mean
    than their real neighbours justify, and regrow slower under the
    'neighbour' rule. That biases the map toward a rich centre and dead rim.
    """
    p = np.pad(g, 1, mode='constant')
    acc = np.zeros_like(g)
    cnt = np.zeros_like(g)
    for dy, dx in product([-1, 0, 1], [-1, 0, 1]):
        acc += p[1+dy:1+dy+N, 1+dx:1+dx+N]
        cnt += 1
    return acc / cnt


def regrow(g, rule, r):
    """Advance the grid one step of resource regrowth. Returns a new grid.

    Two competing models of how a commons replenishes:

    'global'    -- logistic growth on the *total* stock. Growth is
                   r * S * (1 - S/carrying_capacity), computed once from the
                   whole-grid sum S, then spread across cells in proportion to
                   how much empty room each has. A depleted corner is refilled
                   by the health of the map as a whole, so local overharvesting
                   is subsidised by everyone else.

    'neighbour' -- local growth. Each cell regrows at r * (its neighbourhood
                   mean) * (its own empty room). Regrowth is seeded by what is
                   physically nearby, so a cell stripped to zero in a stripped
                   region has nothing to recover from and stays dead.

    `r` is the regrowth rate. Both rules clamp the result to [0, CAP].
    """
    if rule == 'global':
        S = g.sum()
        total_growth = r * S * (1 - S/(N*N*CAP))
        room = (CAP - g)
        if room.sum() <= 0:      # grid already full, nothing to add
            return g
        return np.clip(g + total_growth * room/room.sum(), 0, CAP)
    else:
        nb = neighbours_mean(g)
        return np.clip(g + r * nb * (CAP - g), 0, CAP)


class Agent:
    """A forager with a fixed harvesting policy, tracking its own total take.

    `kind` selects the policy applied in act(). `score` accumulates everything
    this agent has harvested, which is what makes the individual-vs-collective
    comparison possible.
    """

    def __init__(self, kind, x, y, rng):
        self.kind, self.x, self.y, self.rng = kind, x, y, rng
        self.score = 0.0

    def act(self, g):
        """Move one step toward resource, harvest, and mutate the grid in place.

        Movement is a greedy hill-climb: consider staying put plus the four
        orthogonal neighbours, and move to whichever holds the most. Ties are
        broken uniformly at random via self.rng, which matters because ties are
        the common case, not an edge case -- the grid starts uniformly full, so
        on early ticks every candidate cell is identical and the move is a free
        choice among all of them.

        Harvest policy depends on kind:
          'greedy'   -- take up to TAKE, no matter how little is left. Will
                        strip a cell to zero.
          'cautious' -- take up to TAKE but only from the amount above 0.5,
                        leaving half a cell behind as seed stock.
          anything else -- take up to TAKE, but refuse to touch cells below 0.3.
                        (Unused: the sweep below only builds greedy/cautious
                        agents, so this branch never runs.)
        """
        best_val, best = None, []
        for dy, dx in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = self.y+dy, self.x+dx
            if 0 <= ny < N and 0 <= nx < N:          # stay on the grid
                val = g[ny, nx]
                if best_val is None or val > best_val:
                    best_val, best = val, [(ny, nx)]
                elif val == best_val:
                    best.append((ny, nx))
        self.y, self.x = best[self.rng.integers(len(best))]
        cell = g[self.y, self.x]
        if self.kind == 'greedy':
            take = min(cell, TAKE)
        elif self.kind == 'cautious':
            take = min(max(cell-0.5, 0), TAKE)
        else:
            take = min(cell, TAKE) if cell > 0.3 else 0.0
        g[self.y, self.x] -= take
        self.score += take


def run(rule, r, mix, seed):
    """Simulate one full game. Returns (survived, total_harvest, scores, grid).

    Starts from a completely full grid with four agents in the four corners,
    their policies given by `mix` (a list of four kind strings). Each tick all
    agents act, then the grid regrows.

    'survived' is how many ticks the commons lasted: it stays at TICKS if the
    grid never collapsed, or is set to the tick at which total resource fell
    below 5% of capacity, at which point the run stops early. So survived <
    TICKS is the signature of a collapse.

    `seed` drives two sources of randomness, so runs genuinely differ and
    averaging over seeds samples a distribution: agents break movement ties at
    random, and the order in which agents act is reshuffled every tick.

    Shuffling the activation order is not cosmetic. Agents share cells, and
    whoever acts first harvests before the others see the cell, so acting
    early is a real advantage. With a fixed order that advantage always went
    to the same agents -- and since the mixed group is spelled
    ['greedy','greedy','cautious','cautious'], it always went to the two
    greedy ones, inflating the very greedy-vs-cautious gap this script exists
    to measure.
    """
    rng = np.random.default_rng(seed)
    g = np.full((N, N), CAP)
    starts = [(0, 0), (5, 0), (0, 5), (5, 5)]
    ags = [Agent(k, sx, sy, rng) for k, (sx, sy) in zip(mix, starts)]
    survived = TICKS
    for t in range(TICKS):
        for i in rng.permutation(len(ags)):
            ags[i].act(g)
        g = regrow(g, rule, r)
        if g.sum() < 0.05*N*N and survived == TICKS:
            survived = t
            break
    return survived, sum(a.score for a in ags), [a.score for a in ags], g


# Sweep every combination of regrowth rule, regrowth rate, and group
# composition, and report how long the commons lasted and how much the group
# harvested in total. Compare the '4 greedy' and '4 cautious' rows at a given
# rule and r: where cautious harvests more, restraint pays collectively and the
# scenario is a genuine dilemma; where greedy harvests more, regrowth outpaces
# extraction and there is no dilemma to speak of.
print(f"{'rule':<8}{'r':<7}{'mix':<26}{'survived':<10}{'harvest':<10}")
print("-"*63)
for rule in ['global', 'neighbour']:
    for r in R_VALUES:
        for mix, label in [(['greedy']*4, '4 greedy'),
                           (['cautious']*4, '4 cautious'),
                           (['greedy', 'greedy', 'cautious', 'cautious'], '2 greedy / 2 cautious')]:
            res = [run(rule, r, mix, s) for s in range(12)]
            sv = np.mean([x[0] for x in res])
            hv = np.mean([x[1] for x in res])
            print(f"{rule:<8}{r:<7}{label:<26}{sv:<10.1f}{hv:<10.1f}")
    print()
