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
    #ax.legend()
    #plt.grid(True)
    #plt.title(f"x_t, beta={b}, sigma={s}")
    #plt.show()

def mape(y:NDArray, y_hat:NDArray):
    return np.mean(np.abs((y-y_hat)/y))

def mae(y:NDArray, y_hat:NDArray):
    return np.mean(np.abs(y-y_hat))


beta = 0.5
sigma = 0.4
x0 = 0.2
K = 50
Rh = 50
N_trials = 100
T = 100


ramp = HMM_Ramp_Model(
    beta=beta,
    sigma=sigma,
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

mu_expected_xt = np.mean(expected_xt,axis=0)
std_expected_xt = np.std(expected_xt,axis=0)


mu_expected_xt_filter = np.mean(expected_xt_filter,axis=0)
std_expected_xt_filter = np.std(expected_xt_filter,axis=0)

std_xs = np.std(xs,axis=0)

plt.plot(np.arange(T), mu_expected_xt_filter, c='g', label="filter")
plt.fill_between(
    np.arange(T),
    mu_expected_xt -  std_expected_xt,
    mu_expected_xt +  std_expected_xt,
    color='lightgreen'
)
plt.plot(np.arange(T), np.mean(xs,axis=0), c='black', label="true path")

#print(np.argmax(posterior_prob,axis=1))
plt.plot(np.arange(T), mu_expected_xt, c='r', label="smoother")
plt.fill_between(
    np.arange(T),
    mu_expected_xt -  std_expected_xt,
    mu_expected_xt +  std_expected_xt,
    color='#FFCCCC'
)
plt.title(f"x_t, b={beta}, s={sigma}")
plt.xlabel("timestep")
plt.ylabel("x_t")
plt.legend()
plt.grid()
plt.show()