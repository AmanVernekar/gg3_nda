import numpy as np
from models import gamma_isi_point_process


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
        