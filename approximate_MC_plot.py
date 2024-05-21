import numpy as np
import matplotlib.pyplot as plt
from models_HMM import HMM_Ramp_Model


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

ramp = HMM_Ramp_Model(
    beta=1.2,
    sigma=0.25,
    x0=0.2,
    K=100
)

N_trials = 13
T = 100

spikes, xs, rates = ramp.simulate(N_trials, T)

# plot_spike_raster(spikes)
plot_x(xs,ramp.beta,ramp.sigma)