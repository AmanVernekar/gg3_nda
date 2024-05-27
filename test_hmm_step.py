from models_HMM import StepHMM_naive, StepHMM_better
import matplotlib.pyplot as plt

m=50
r=100
x0=0.2

N_trials = 10
T = 100

step_naive = StepHMM_naive(m,r,x0)
step_better = StepHMM_better(m,r,x0)

chains, spikes, jumps, rates = step_naive.simulate(N_trials, T)
for i in range(N_trials):
    chain = chains[i]
    chain[chain==0.2] = 0
    plt.plot(chain, label = f'Markov Chain {i}')
plt.title(f'Markov Chains for naive Step HMM {m = } {r = }')
# plt.ylim((0,r+10))
plt.legend()
plt.show()

for i in range(N_trials):
    plt.plot(rates[i], label = f'Firing rate in trial {i}')
plt.title(f'Firing rates for naive Step HMM {m = } {r = } {x0 = }')
plt.legend()
plt.show()

# plt.xlim((0,T))
# for r in [1,5,10,100,1000]:
#     step_naive = StepHMM_naive(m,r,x0)
#     chains, spikes, jumps, rates = step_naive.simulate(N_trials, T)
#     plt.hist(jumps[jumps <= T], bins=int(T/10), label=f"{r=}")
# plt.title(f"Jump times for naive Step HMM, {m=}")
# plt.legend()
# plt.show()
