import numpy as np
from compare import run, N, CAP, R_VALUES

print("MIXED RUNS: does the greedy agent out-earn the cautious one?")
print(f"{'rule':<11}{'r':<7}{'greedy avg':<13}{'cautious avg':<15}{'survived':<10}{'dilemma?'}")
print("-"*72)
mix = ['greedy','greedy','cautious','cautious']
for rule in ['global','neighbour']:
    for r in R_VALUES:
        res = [run(rule, r, mix, s) for s in range(20)]
        gs = np.mean([np.mean(x[2][:2]) for x in res])
        cs = np.mean([np.mean(x[2][2:]) for x in res])
        sv = np.mean([x[0] for x in res])
        dil = "yes" if gs > cs and sv < 100 else ("weak" if gs > cs else "no")
        print(f"{rule:<11}{r:<7}{gs:<13.1f}{cs:<15.1f}{sv:<10.1f}{dil}")
    print()

print("\nDEAD ZONES: cells still at ~0 at end of a 4-greedy run (r=0.15)")
print(f"{'rule':<11}{'dead cells':<14}{'recovered cells':<18}{'spatial memory?'}")
print("-"*62)
for rule in ['global','neighbour']:
    deads, recs = [], []
    for s in range(20):
        sv, tot, sc, g = run(rule, 0.15, ['greedy']*4, s)
        deads.append(int((g < 0.05).sum()))
        recs.append(int((g > 0.5).sum()))
    print(f"{rule:<11}{np.mean(deads):<14.1f}{np.mean(recs):<18.1f}"
          f"{'strong' if np.mean(deads) > 8 else 'weak'}")
