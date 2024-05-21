import numpy as np
from approximate_mc import get_interval, get_init_distr, get_transition_mtx, simulate_chain
from models import gamma_isi_point_process

class HMM_Ramp_Model():
        """
        Simulator of the HMM approximation of the Ramping Model
        """

        def __init__(self, beta=0.5, sigma=0.2, x0=.2, K=50, Rh=50, isi_gamma_shape=None, Rl=None, dt=None):
            """
            Simulator of the Ramping Model of Latimer et al. Science 2015.
            :param beta: drift rate of the drift-diffusion process
            :param sigma: diffusion strength of the drift-diffusion process.
            :param x0: average initial value of latent variable x[0]
            :param Rh: the maximal firing rate obtained when x_t reaches 1 (corresponding to the same as the post-step
                       state in most of the project tasks)
            :param isi_gamma_shape: shape parameter of the Gamma distribution of inter-spike intervals.
                                see https://en.wikipedia.org/wiki/Gamma_distribution
            :param Rl: Not implemented. Ignore.
            :param dt: real time duration of time steps in seconds (only used for converting rates to units of inverse time-step)
            """
            self.beta = beta
            self.sigma = sigma
            self.x0 = x0
            self.K = K
            self.Rh = Rh
            if Rl is not None:
                self.Rl = Rl

            self.isi_gamma_shape = isi_gamma_shape
            self.dt = dt

        @property
        def params(self):
            return self.beta, self.sigma, self.x0, self.K

        @property
        def fixed_params(self):
            return self.Rh, self.Rl

        def f_io(self, xs, b=None):
            if b is None:
                return self.Rh * np.maximum(0, xs)
            else:
                return self.Rh * b * np.log(1 + np.exp(xs / b))

        def emit(self, rate):
            """
            emit spikes based on rates
            :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
            :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                     trials and time bins.
            """
            if self.isi_gamma_shape is None:
                # poisson spike emissions
                y = np.random.poisson(rate * self.dt)
            else:
                # sub-poisson/underdispersed spike emissions
                y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

            return y

        def simulate(self, Ntrials=1, T=100, get_rate=True):
            """
            :param Ntrials: (int) number of trials
            :param T: (int) duration of each trial in number of time-steps.
            :param get_rate: whether or not to return the rate time-series
            :return:
            spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                    an array of spike counts in each time-bin (= time step)
            xs:     shape = (Ntrial, T); xs[j] is the latent variable time-series x_t in trial j
            rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
            """
            # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
            state_space, dt = get_interval(self.K)
            self.dt = dt
            pi = get_init_distr(state_space, self.x0, self.sigma, dt)
            trans_mtx = get_transition_mtx(state_space, self.beta, self.sigma, dt)
            xs = simulate_chain(pi, trans_mtx, T, np.round(self.x0 * self.K), state_space)

            rates = self.f_io(xs)  # shape = (Ntrials, T)

            spikes = np.array([self.emit(rate) for rate in rates])  # shape = (Ntrial, T)

            if get_rate:
                return spikes, xs, rates
            else:
                return spikes, xs




class StepHMM_naive():
    def __init__(self, m=50, r=10, x0=0.2, Rh=50, Rl=None, isi_gamma_shape=None, dt=None):
        self.m = m
        self.r = r
        self.x0 = x0

        self.p = r / (m + r)
        self.q = self.p   # tranition matrix is [[1-q, q], [0, 1]]
        self.trans_mtx = np.array([[1-self.q, self.q], [0, 1]])

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl
        
        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt
    
    @property
    def params(self):
        return self.m, self.r, self.x0

    @property
    def fixed_params(self):
        return self.Rh, self.Rl
    
    def get_transition_mtx(self):
        return self.trans_mtx
    
    def simulate_chain(self, T):
        chain = np.ones(T)*self.x0
        for i in range(1,T):
            next_state = np.random.choice([0, 1], p=[1-self.q, self.q])
            if next_state == 1:
                jump_step = i
                chain[i:] = 1
                break
        rate = chain * self.Rh
        return rate, jump_step

    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                    trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = np.random.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y

    def simulate(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        spikes, jumps, rates = [], [], []
        for _ in range(Ntrials):
            # sample jump time
            rate, jump_step = self.simulate_chain(T)
            jumps.append(jump_step)
            
            rates.append(rate)

            spikes.append(self.emit(rate))

        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates)
        else:
            return np.array(spikes), np.array(jumps)
    


    
class StepHMM_better():
    def __init__(self, m=50, r=10, x0=0.2, Rh=50, Rl=None, isi_gamma_shape=None, dt=None):
        self.m = m
        self.r = r
        self.x0 = x0

        self.p = r / (m + r)

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl
        
        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt
    
    @property
    def params(self):
        return self.m, self.r, self.x0

    @property
    def fixed_params(self):
        return self.Rh, self.Rl
    
    def simulate_chain(self, T):
        chain = np.zeros(T+self.r)
        for i in range(1,T+self.r):
            next_state = np.random.choice([chain[i-1], chain[i-1] + 1], p=[1-self.p, self.p])
            if next_state == self.r:
                jump_step = i - self.r
                chain[i:] = self.r
                break
        return chain[self.r:], jump_step
    
    def emit(self, rate):
        """
        emit spikes based on rates
        :param rate: firing rate sequence, r_t, possibly in many trials. Shape: (Ntrials, T)
        :return: spike train, n_t, as an array of shape (Ntrials, T) containing integer spike counts in different
                    trials and time bins.
        """
        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = np.random.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y

    def simulate(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        ts = np.arange(T)

        spikes, jumps, rates = [], [], []
        for _ in range(Ntrials):
            # sample jump time
            chain, jump_step = self.simulate_chain(T)
            jumps.append(jump_step)
            
            rate = np.ones(T) * self.x0 * self.Rh
            rate[ts >= jump_step] = self.Rh
            rates.append(rate)

            spikes.append(self.emit(rate))

        if get_rate:
            return np.array(spikes), np.array(jumps), np.array(rates)
        else:
            return np.array(spikes), np.array(jumps)