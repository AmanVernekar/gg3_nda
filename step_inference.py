import numpy as np
from inference import poisson_logpdf, hmm_expected_states
from models_HMM import StepHMM_better
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_step_model(m,r,x0,Rh,T,N_trials):
    step = StepHMM_better(m = m,r = r,x0= x0, Rh = Rh)
    _, spikes, jumps, _ = step.simulate(N_trials, T)
    rates = np.array([x0]*(r) + [1])*Rh*(1/T)
    ll = poisson_logpdf(spikes,rates)
    
    Ps = step.trans_mtx
    pi = np.array([1] + [0]*r) @ Ps**r
    
    marker_styles = [None, 's', 'x']
    line_styles = ['solid', None, None]
    labels = ['Posterior probability', 'Inferred jump time', 'True jump time']

    fig, ax = plt.subplots(1,2)
    plt.suptitle(f'Probability of being in the upper rate level\nm = {m}, r = {r}, x0 = {x0}, Rh = {Rh}')
    fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(10) 
    for j in range(N_trials):
        for i in range(2):
            expected_states ,_= hmm_expected_states(pi,Ps,ll = ll[j,:,:], filter=i)# each row is posterior distribution at a given time
        #shape(expected_states)= T,K
            high_state_probability = expected_states[:,r]
            ax[i].plot(np.arange(T),high_state_probability, color= "C{}".format(j))

            jump_time = np.where(high_state_probability>= 0.5)
            if len(jump_time[0]) > 0:
                jump_time = jump_time[0][0]
                # plt.axvline(jump_time, color= "C{}".format(j), linestyle='dotted', label='Inferred jump time')
                ax[i].scatter(jump_time, 0.5, color= "C{}".format(j), marker =marker_styles[1])

            # Only plot the simulated jump if we have actually jumped
            if jumps[j] != -1:
                # plt.axvline(jump[j], color = "C{}".format(j), linestyle = 'dashed', label='True jump time')
                ax[i].scatter(jumps[j], 0.5, color = "C{}".format(j), marker =marker_styles[2])

    dummy_lines = []
    for i in range(len(marker_styles)):
        dummy_lines.append(Line2D([], [], linestyle=line_styles[i], marker=marker_styles[i], color='black', label=labels[i]))
    for i in range(2):
        ax[i].legend(handles=dummy_lines, loc='lower right')
    ax[0].set_title('Smoothing')
    ax[1].set_title('Filtering')
    plt.show()


# plot_step_model(m=30,r=10,x0=0.2,Rh=50,T=100,N_trials=5)
# # plot_step_model(m=30,r=100,x0=0.2,Rh=50,T=100,N_trials=5)
# plot_step_model(m=50,r=1,x0=0.2,Rh=50,T=100,N_trials=5)
# plot_step_model(m=50,r=10,x0=0.2,Rh=50,T=100,N_trials=5)
# # plot_step_model(m=50,r=100,x0=0.2,Rh=50,T=100,N_trials=5)

# plot_step_model(m=30,r=10,x0=0.7,Rh=50,T=100,N_trials=5)
# # plot_step_model(m=30,r=100,x0=0.7,Rh=50,T=100,N_trials=5)
# plot_step_model(m=50,r=1,x0=0.7,Rh=50,T=100,N_trials=5)
# plot_step_model(m=50,r=10,x0=0.7,Rh=50,T=100,N_trials=5)
# # plot_step_model(m=50,r=100,x0=0.7,Rh=50,T=100,N_trials=5)


def calc_avg_error(m, r, x0, Rh, T, N_trials, _filter=False):
    step = StepHMM_better(m = m,r = r,x0= x0, Rh = Rh)
    _, spikes, jumps, _ = step.simulate(N_trials, T)
    jumps[jumps==np.inf] = T
    rates = np.array([x0]*(r) + [1])*Rh*(1/T)
    ll = poisson_logpdf(spikes,rates)
    
    Ps = step.trans_mtx
    pi = np.array([1] + [0]*r) @ Ps**r
    
    mse = np.zeros((N_trials))

    for j in range(N_trials):
        expected_states, _ = hmm_expected_states(pi,Ps,ll = ll[j], filter=_filter)
        high_state_probability = expected_states[:,r]
        jump_time = T
        for k, prob in enumerate(high_state_probability):
            if prob >= 0.5:
                jump_time = k
                break
        mse[j] = ((jump_time - jumps[j])/T)**2
    
    return np.mean(mse)

def step_heatmap(m, Rh=50, T=100, N_trials=10, _filter=False):
    num_vals = 10
    
    r = range(1,11,1)
    x0 = np.linspace(0.1, 1, num_vals)
    
    avg_errors = np.zeros([num_vals, num_vals])

    # Compute the return values of the inner function for each combination of varying parameters
    for i, val1 in enumerate(r):
        for j, val2 in enumerate(x0):
            avg_errors[i][j] = calc_avg_error(m, val1, val2, Rh, T, N_trials, _filter)
    
    


    fig, ax = plt.subplots()
    im = ax.imshow(avg_errors)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(r)), labels=r)
    ax.set_yticks(np.arange(len(x0)), labels=x0)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('yo', rotation=-90, va="bottom")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()

step_heatmap(m=30)
