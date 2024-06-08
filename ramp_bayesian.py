import numpy as np
from inference import hmm_expected_states, poisson_logpdf
from models_HMM import HMM_Ramp_Model
import matplotlib.pyplot as plt

def log_to_prob(grid):
    t = np.exp(grid + np.min( abs(grid)) )
    return t/ np.sum(t)


b = 0.5
s = 0.2

M = 30 #
x0 = 0.2 #
K = 100 #
T = 100 #
Rh = 50 #
#N_trials = 10 #

betas = np.flipud(np.linspace(0.0, 4.0,num=M))
log_sigmas =  np.linspace(np.log(0.04),np.log(4),num=M)
log_p_ns_grid = np.zeros( (M,M) )

#The model that generated the data
ramp_gen = HMM_Ramp_Model(
    beta=0.5,
    sigma=0.2,
    x0=x0,
    Rh=Rh,
    K=K
)

for N_trials in [1]:
    ns = ramp_gen.simulate(N_trials, T)[0]
    for beta in range(betas.size):
        for ls in range(log_sigmas.size):
            ramp = HMM_Ramp_Model(
                beta=betas[beta],
                sigma=np.exp(log_sigmas[ls]),
                x0= x0,
                Rh= Rh,
                K = K
            )
            #GET THE PI and trans matrix by simulating 1 sample
            ramp.simulate(0,100)
            log_p_nt_xt = poisson_logpdf(
                counts = ns,
                lambdas= ramp.Rh * ramp.state_space * ramp.dt
            )
            log_p_ns = 0.0
            for trial in range(N_trials):
                log_p_ns += hmm_expected_states(
                    ramp.pi,
                    ramp.trans_mtx,
                    log_p_nt_xt[trial, :, :]
                )[1]
            log_p_ns_grid[beta,ls] = log_p_ns

    log_prior = -2 * np.log(K)
    p_theta_ns = log_to_prob(log_p_ns_grid + log_prior)

    plt.figure(figsize=(6, 6))
    plt.imshow(log_p_ns_grid + log_prior , cmap='viridis', interpolation = 'nearest')
    plt.colorbar(label='Log posterior of parameters')
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"$\beta$")
    plt.xticks( np.arange(len(np.exp(log_sigmas))), np.round(np.exp(log_sigmas), 2) ,rotation=90)
    plt.yticks( np.arange(len(betas)), np.round(betas, 2) )

    s_index = np.argmin(np.abs(np.exp(log_sigmas) - s))
    b_index = np.argmin(np.abs(betas - b))
    plt.scatter( s_index, b_index, label = "True", c="r", marker="x")

    #p_ns_grid = p_theta_ns / np.sum(p_theta_ns)
    #Find expectation
    p_sigma_ns = np.sum(p_theta_ns, axis=0 )
    p_beta_ns = np.sum(p_theta_ns, axis=1 )

    expected_sigma = np.sum( np.exp(log_sigmas) * p_sigma_ns )
    expected_sigma2 = np.sum(
        (np.exp(log_sigmas)**2) * p_sigma_ns
    )
    expected_beta = np.sum( betas * p_beta_ns )
    expected_beta2 = np.sum(
        (betas**2) * p_beta_ns
    )

    std_sigma = np.sqrt( expected_sigma2 - expected_sigma **2 )
    std_beta = np.sqrt( expected_beta2 - expected_beta **2 )

    mean_s_idx = np.argmin(np.abs(np.exp(log_sigmas) - expected_sigma))
    #std_s_idx = np.argmin(np.abs(np.exp(log_sigmas) - std_sigma))
    mean_b_idx = np.argmin(np.abs(betas - expected_beta))
    #std_b_idx =  np.argmin(np.abs(betas - std_beta))

    plt.scatter( mean_b_idx, mean_s_idx, label = "Expected", c="purple")
    #plt.errorbar( mean_b_idx, mean_s_idx, xerr = std_b_idx, yerr = std_s_idx ,c="r")

    plt.title(f'Heatmap of parameters log-posterior for {N_trials} trials')
    plt.legend()
    plt.show()
    with open(f'heatmap_b={b}_s={s}_n={N_trials}.npy', 'wb') as f:
        np.save(f, log_p_ns_grid)

