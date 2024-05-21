from models_HMM import StepHMM_naive, StepHMM_better
import matplotlib.pyplot as plt

m=50
r=100
x0=0.2

N_trials = 5000
T = 100

# step_naive = StepHMM_naive(m,r,x0)
# step_better = StepHMM_better(m,r,x0)

# chains, spikes, jumps, rates = step_better.simulate(N_trials, T)
# plt.plot(chains[0])
# plt.show()

plt.xlim((0,T))
for m in [20,40,60,80]:
    step_better = StepHMM_better(m,r,x0)
    chains, spikes, jumps, rates = step_better.simulate(N_trials, T)
    plt.hist(jumps[jumps <= T], bins=int(T/10), label=f"{m=}")
plt.title(f"Jump times for better Step HMM, {r=}")
plt.legend()
plt.show()
