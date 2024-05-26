import numpy as np
from inference import poisson_logpdf, hmm_expected_states
from models_HMM import StepHMM_better
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# m = 50
# r = 1
# x0 = 0.2
# Rh = 50

# N_trials = 5
# T = 100
# dt = 1/T

# step_better = StepHMM_better(m, r, x0, Rh=Rh)

# chains, spikes, jumps, _ = step_better.simulate(N_trials, T)
# rates = np.array([x0]*(r) + [1])*Rh*(1/T)

# lls = poisson_logpdf(spikes, rates)
# # print(lls)
# Ps = step_better.trans_mtx
# pi = np.array([1] + [0]*r) @ Ps**r

# expected_states, normalizer = hmm_expected_states(pi, Ps, lls[0])
# print(expected_states[-1])


def plot_step_model(m,r,x0,Rh,T,Ntrials):
    step = StepHMM_better(m = m,r = r,x0= x0, Rh = Rh)
    chains, spikes, jumps, _ = step.simulate(Ntrials, T)
    # pi0 = step.p0
    # Ps = step.transition_matrix
    # Ps = np.array([Ps])
    # xt_list = np.ones(r+1)*x0
    # xt_list[-1] = 1
    # rates_list = Rh * xt_list
    rates = np.array([x0]*(r) + [1])*Rh*(1/T)
    ll = poisson_logpdf(spikes,rates)#what matters is the mean count not the rate
    # print(ll)
    
    Ps = step.trans_mtx
    pi = np.array([1] + [0]*r) @ Ps**r
    
    marker_styles = [None, 'o', 'x']
    line_styles = ['solid', None, None]
    labels = [r'$P(s_t = 1 | n_{1:T})$', 'Inferred jump time', 'True jump time']

    fig, ax = plt.subplots(1,2)
    plt.suptitle('m = {}, r = {}, x0 = {}, Rh = {}'.format(m,r,x0,Rh))
    # fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(10) 
    for i_trial in range(Ntrials):
        for i in range(2):
            expected_states ,_= hmm_expected_states(pi,Ps,ll = ll[i_trial,:,:], filter=i)# each row is posterior distribution at a given time
        #shape(expected_states)= T,K
            high_state_probability = expected_states[:,r]
            ax[i].plot(np.arange(T),high_state_probability, color= "C{}".format(i_trial))

            jump_time = np.where(high_state_probability>= 0.5)
            if len(jump_time[0]) > 0:
                jump_time = jump_time[0][0]
                # plt.axvline(jump_time, color= "C{}".format(i_trial), linestyle='dotted', label='Inferred jump time')
                ax[i].scatter(jump_time, 0.5, color= "C{}".format(i_trial), marker =marker_styles[1])

            # Only plot the simulated jump if we have actually jumped
            if jumps[i_trial] != -1:
                # plt.axvline(jump[i_trial], color = "C{}".format(i_trial), linestyle = 'dashed', label='True jump time')
                ax[i].scatter(jumps[i_trial], 0.5, color = "C{}".format(i_trial), marker =marker_styles[2])

    dummy_lines = []
    for i in range(len(marker_styles)):
        dummy_lines.append(Line2D([], [], linestyle=line_styles[i], marker=marker_styles[i], color='black', label=labels[i]))
    for i in range(2):
        ax[i].legend(handles=dummy_lines, loc='lower right')
    ax[0].set_title('Smoothing')
    ax[1].set_title('Filtering')
    plt.show()


plot_step_model(m=20,r=10,x0=0.2,Rh=50,T=10,Ntrials=1)
plot_step_model(m=50,r=1,x0=0.2,Rh=50,T=100,Ntrials=5)
plot_step_model(m=50,r=10,x0=0.8,Rh=50,T=100,Ntrials=5)
plot_step_model(m=50,r=1,x0=0.8,Rh=50,T=100,Ntrials=5)

plot_step_model(m=50,r=10,x0=0.8,Rh=50,T=100,Ntrials=5)

plot_step_model(m=100,r=10,x0=0.8,Rh=50,T=100,Ntrials=5)
