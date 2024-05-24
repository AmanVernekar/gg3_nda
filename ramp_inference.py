import numpy as np
from numpy.typing import NDArray
from inference import hmm_expected_states, poisson_logpdf
from models_HMM import StepHMM_better, HMM_Ramp_Model

import matplotlib.pyplot as plt
import seaborn as sns

beta = 1.2
sigma = 0.2
x0 = 0.2
K = 50

N_trials = 20
T = 100

def mape(y:NDArray, y_hat:NDArray):
    return np.mean(np.abs((y-y_hat)/y))
def mae(y:NDArray, y_hat:NDArray):
    return np.mean(np.abs(y-y_hat))

ramp = HMM_Ramp_Model(
    beta=beta,
    sigma=sigma,
    x0= x0,
    K = K
)

spikes, xs, rates = ramp.simulate( N_trials, T)

ll = poisson_logpdf(counts=spikes, lambdas= ramp.Rh * ramp.state_space * ramp.dt)[:,:,:]
expected_xt = np.zeros( (N_trials, T) )
for trial in range(N_trials):
    posterior_prob, normalizer = hmm_expected_states(ramp.pi,ramp.trans_mtx,ll[trial,:,:])
    #Compute expectation
    expected_xt[trial,:] = posterior_prob @ ramp.state_space
    # print(np.argmax(posterior_prob,axis=1))
    #print(f"EM inference on xt: {expected_xt}")
    #print(f"Ground truth: {xs}")

print(f"MAPE: {mape(expected_xt, xs)}")
print(f"MAE: {mae(expected_xt, xs)}")