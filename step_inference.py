import numpy as np
from inference import poisson_logpdf, hmm_expected_states
from models_HMM import StepHMM_better

m = 20
r = 100
x0 = 0.2
Rh = 50

N_trials = 1
T = 10

step_better = StepHMM_better(m, r, x0, Rh=Rh)

chains, spikes, jumps, _ = step_better.simulate(N_trials, T)
rates = np.array([x0]*(r) + [1])*Rh*(1/T)

lls = poisson_logpdf(spikes, rates)
print(lls)
Ps = step_better.trans_mtx
pi = np.array([1] + [0]*r) @ Ps**r

expected_states, normalizer = hmm_expected_states(pi, Ps, lls[0])
print(expected_states[-1])
