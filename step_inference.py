import numpy as np
from inference import poisson_logpdf, hmm_expected_states
from models_HMM import StepHMM_better

m = 50
r = 10
x0 = 0.2

N_trials = 500
T = 100

step_better = StepHMM_better(m, r, x0)

chains, spikes, jumps, rates = step_better.simulate(N_trials, T)

lls = poisson_logpdf(spikes, rates)

