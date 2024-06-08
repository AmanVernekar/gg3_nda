import numpy as np
from numpy.typing import NDArray
from inference import hmm_expected_states, poisson_logpdf
from models_HMM import HMM_Ramp_Model
from heatmap import plot_heatmap
import matplotlib.pyplot as plt

def plot_x(x,b,s):
    fig, ax = plt.subplots()
    ts = np.arange(x.shape[1])
    for i in range(x.shape[0]):
        ax.plot(ts, x[i], label=f'spikes{i}')
    ax.legend()
    plt.grid(True)
    plt.title(f"x_t, beta={b}, sigma={s}")
    plt.show()

def mape(y:NDArray, y_hat:NDArray):
    return np.mean(np.abs((y-y_hat)/y))

def mae(y:NDArray, y_hat:NDArray):
    return np.mean(np.abs(y-y_hat))

name_s = "sigma"
name_l = "beta"
start_s= 0.1
end_s =  1.0
start_l= 0.0
end_l = 1.8
beta = 0.5
sigma = 0.2
x0 = 0.2
K = 50
Rh = 50
N_trials = 100
T = 100
num_samples = 10


S = np.linspace(start_s, end_s, num_samples)
L = np.flipud(np.linspace(start_l, end_l, num_samples))
grid = np.zeros((S.size,L.size))

for i in range(S.size):
    for j in range(L.size):
        ramp = HMM_Ramp_Model(
            beta=L[j],
            sigma=S[i],
            x0= x0,
            Rh= Rh,
            K = K
        )
        spikes, xs, rates = ramp.simulate( N_trials, T)
        ll = poisson_logpdf(counts = spikes, lambdas= ramp.Rh * ramp.state_space * ramp.dt)
        expected_xt = np.zeros( (N_trials, T) )
        expected_xt_filter = np.zeros( (N_trials, T) )
        for trial in range(N_trials):
            posterior_prob, normalizer = hmm_expected_states(ramp.pi,ramp.trans_mtx,ll[trial,:,:])
            posterior_prob_filter,normalizer_filter = hmm_expected_states(ramp.pi,ramp.trans_mtx,ll[trial,:,:],filter=True)
            expected_xt[trial,:] = posterior_prob @ ramp.state_space
            expected_xt_filter[trial, :] = posterior_prob_filter @ ramp.state_space
        grid[j,i] = mae(xs, expected_xt_filter)-mae(xs, expected_xt)
        #print(np.argmax(posterior_prob,axis=1))
        #print(f"EM inference on xt: {expected_xt}")
        #print(f"Ground truth: {xs}")
        #print(f"MAPE: {mape(expected_xt, xs)}")
        print(f"d MAE: {grid[j,i]}")

plot_heatmap(grid, S, L, name_s=name_s, name_l=name_l)