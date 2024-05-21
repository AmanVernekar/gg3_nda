import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from typing import Tuple, Optional

def get_interval(k:int)->Tuple[NDArray,float]:
    state_space,dt=np.linspace(0,1,num=k,retstep=True)
    return state_space, dt

def get_init_distr(
        states:NDArray,
        x0:float,
        sigma:float,
        dt:float
)->NDArray:
    init_distr = np.zeros_like(states)
    #get states in the first and last bins separately
    init_distr[0] = norm.cdf( (states[0] + dt/2 -x0)/ (sigma * np.sqrt(dt) ) )
    init_distr[-1]= 1.0 - norm.cdf( (states[-1]-dt/2-x0)/ (sigma*np.sqrt(dt)) )
    #get the other bins from the inverse cdf
    if states.size>2:
        for i in range(1,states.size-1):
            init_distr[i] = norm.cdf((states[i]+ dt/2- x0)/(sigma* np.sqrt(dt)))- norm.cdf((states[i]-dt/2-x0)/(sigma*np.sqrt(dt)))
    init_distr = init_distr/ np.linalg.norm(init_distr,ord=1)

    return init_distr

def get_transition_mtx(
        states:NDArray,
        beta:float,
        sigma:float,
        dt:float
):
    trans_mtx = np.zeros((states.size,states.size))
    #Handle hitting at x=1
    trans_mtx[:-1, -1] = 1.0 - norm.cdf( (states[-1]-dt/2-states[:-1] - beta * dt) / (sigma * np.sqrt(dt)) )
    trans_mtx[states.size-1,states.size-1] = 1.0
    #Handle floor at x=0
    trans_mtx[:-1,0]=norm.cdf((states[0] + dt/2 - states[:-1] - beta * dt )/(sigma*np.sqrt(dt)))
    if states.size>2: #Rest of the matrix
        for i in np.arange(states.size-1):
            for j in np.arange(1, states.size - 1):
                trans_mtx[i,j] =  norm.cdf((states[j]+dt/2-states[i]-beta* dt)/(sigma* np.sqrt(dt)))- norm.cdf((states[j]-dt/2-states[i]- beta* dt)/(sigma*np.sqrt(dt)))
    #Normalize the matrix
    for i in np.arange(states.size):
        trans_mtx[i,:] = trans_mtx[i,:]/ np.linalg.norm(trans_mtx[i,:],ord=1)
    return trans_mtx

def simulate_chain(
        pi:NDArray,
        trans_mtx:NDArray,
        Ntrials:int,
        T:int,
        s0:int,
        state_space:NDArray,
)->NDArray:
    #state_space = pi.size

    s = np.zeros((Ntrials,T),dtype=float)
    s[:,0]=state_space[s0]
    for i in range(Ntrials):
        p = pi
        for j in range(1,T):
            sample = np.random.choice( pi.size, p=p)
            s[i,j] = state_space[sample] #s.append(sample)
            p = trans_mtx[sample,:]
    return s

if __name__=="__main__":
    np.random.seed(0)
    Ntrials = 20
    T:int = 100
    K:int = 50
    sigma:float = 0.5
    beta:float = 1.0
    x0:float = 0.2
    s0 = round(x0*K)
    state_space, dt = get_interval(K)
    pi = get_init_distr(state_space, x0, sigma, dt)
    trans_mtx = get_transition_mtx(state_space, beta, sigma, dt)
    path = simulate_chain(pi, trans_mtx, Ntrials, T, s0, state_space)
    print(f"Path:\n {path[0,:]}")

