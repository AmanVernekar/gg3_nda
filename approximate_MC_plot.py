import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from models_HMM import HMM_Ramp_Model
from models import RampModel

def mae(y:NDArray, y_hat:NDArray):
    return np.mean(np.abs((y-y_hat)))
def mase(y:NDArray, y_hat:NDArray):
    return np.mean(np.abs((y-y_hat))/y)

def plot_spike_raster(spikes_list, title=None):
  if title is None:
    title= f'Spike Raster Plot for HMM, beta={ramp.beta},K={ramp.K}'
  if len(spikes_list.shape)==2:
    # set different colors for each set of positions
    colors = ['C{}'.format(i) for i in range(spikes_list.shape[0])]
  else:
    colors = "black"
  # Define trial and time step indices where spikes occur
  trial_indices, time_indices = np.nonzero(spikes_list)

  # Plotting
  plt.figure(figsize=(10, 6))
  plt.scatter(time_indices/T * 1000, trial_indices, marker='|', color="black")
  plt.xlabel('Time (ms)')
  plt.ylabel('Spike Trains')
  plt.title(title)
  plt.grid(True)
  plt.show()

def plot_x(xs,beta,sigma):
    fig, ax = plt.subplots()
    ts = np.arange(xs.shape[1])
    for i in range(xs.shape[0]):
        ax.plot(ts, xs[i], label=f'spikes{i}')
    ax.legend()
    plt.grid(True)
    plt.title(f"x_t, beta={beta}, sigma={sigma}")
    plt.show()

N_trials = 8
T = 100
beta=0.5
sigma=0.4
#ramp_hmm = HMM_Ramp_Model(
#    beta=beta,
#    sigma=sigma,
#    x0=0.1,
#    K=50
#)
#np.random.seed(0)
#spikes_hmm, xs_hmm, rates_hmm = ramp_hmm.simulate(N_trials, T)
#ramp = RampModel(
#    beta=beta,
#    sigma=sigma,
#    x0=0.1,
#)
#np.random.seed(0)
#spikes, xs, rates = ramp.simulate(N_trials, T)

# plot_spike_raster(spikes)
#plot_x(xs,ramp.beta,ramp.sigma)
#plot_x(xs_hmm,ramp_hmm.beta,ramp_hmm.sigma)



# plot_spike_raster(spikes)
#plot_x(xs_hmm,ramp_hmm.beta,ramp_hmm.sigma)
#plot_x(xs,ramp.beta,ramp.sigma)

def plot_fano(hmm_model, model,legend=""):
    np.random.seed(0)
    spikes_s = hmm_model.simulate(10_000,25)[0]
    np.random.seed(0)
    spikes_r  = model.simulate(10_000,25)[0]
    mean_s = np.mean(spikes_s, axis=0)
    var_s = np.var( spikes_s, axis=0)
    fano_s = var_s/ mean_s
    mean_r = np.mean(spikes_r, axis=0)
    var_r = np.var( spikes_r, axis=0)
    fano_r = var_r/ mean_r
    plt.plot(np.arange(fano_s.size),fano_s,label="Hmm "+legend)
    plt.plot(np.arange(fano_r.size),fano_r,label="Cont. "+legend)

def moving_average(arr, window_size):
    # Define the kernel for convolution
    kernel = np.ones(window_size) / window_size

    # Perform convolution along axis 2
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=1, arr=arr)

#Plot PSTH
def plot_psth(spikes,window_size = 5,typed=""):
  # Compute the moving average along axis 2
  ma = moving_average(spikes, window_size)
  #Sample averaging
  psth=np.mean(ma,axis=0)
  plt.plot(np.arange(psth.size),psth,label=typed)
  return psth




#plt.legend()
#plt.xlabel("timestep")
#plt.ylabel("PSTH")
#plt.grid(True)
#plt.title(f"PSTH @ K={50}")
#plt.show()


for beta in [0.5,1.2]:
    for sigma in [0.3, 0.8]:
        ramp_hmm = HMM_Ramp_Model(
            beta=beta,
            sigma=sigma,
            x0=0.2,
            K=50
        )
        np.random.seed(0)
        spikes_hmm, xs_hmm, rates_hmm = ramp_hmm.simulate(N_trials, T)
        ramp = RampModel(
            beta=beta,
            sigma=sigma,
            x0=0.2,
        )
        np.random.seed(0)
        spikes, xs, rates = ramp.simulate(N_trials, T)
        plot_fano(ramp_hmm,ramp,f"b={beta} s={sigma}")

plt.title(f"Fano factor")
plt.legend()
plt.grid()
plt.show()
