"""
Filename: model_simple.py

Authors: Lorenzo Lagos

Solves the infinite horizon dynamic learning model 
with value funtion iteration.

"""
from textwrap import dedent
from scipy.stats import norm as norm_distribution
from scipy import spatial
import numpy as np

class FireProblem(object):
    
    """

    Parameters
    ----------
    w : np.ndarray
        The monthly wage of the workers (input)
    beta : scalar(float), optional(default=0.95)
        The discount parameter
    y_0 : scalar(float), optional(default=-1)
        The difference between prior expected match quality and wages
    sig2_0 : scalar(float), optional(default=1)
        The prior on match quality variance
    sig2_star : scalar(float), optional(default=2)
        The noise of the match quality signal
    T_star : scalar(int), optional(default=365)
        The number of periods in the grid
    T_k : scalar(int), optional(default=30)
        The end of the probationary period
    b : scalar(float), optional(default=0.358)
        The workers' mandated benefits
    f : scalar(float), optional(default=1/300)
        The firing fines imposed on firms
    sp : scalar(float), optional(default=3)
        The spread on the grid of firm beliefs
    st : scalar(float), optional(default=0.1)
        The step on the grid of firm beliefs


    Attributes
    ----------
    w, beta, y_0, sig2_0, sig2_star, T_star, T_k, b, f : see Parameters
    n : scalar(int)
        The number of workers being simulated (size of w)
    y : np.ndarray
        The grid of firms' belief about match quality
    t : np.ndarray
        The grid of worker tenure
    H_0 : scipy.stats._distn_infrastructure.rv_frozen
        The true CDF of match quality, ndim = n
    h_0 : function
        The true PDF of match quality, ndim = n
        Input=vector; Output=vector
    y_star : np.ndarray
        True match quality, ndim = n
    xi : np.ndarray
        Match quality signals, ndim = n x T_star
    c : np.ndarray
        Firing costs, ndim = n x T_star
    mu : np.ndarray
        The mean for transition probabilities, ndim = n x st x T_star
    sigma : np.ndarray
        The variance for transition probabilities, ndim = n x st x T_star
    h_t : np.ndarray
        The predicted PDF of match quality, ndim = n
        Input=scalar; Output=matrix (st x T_star)

    """

    def __init__(self, w, beta=0.95, y_0=-1, sig2_0=1, 
                 sig2_star=2, T_star=365, T_k=30,
                 b=0.358, f=1/300, sp=3, st=0.1):
        "Initializing match relations with probationary contracts"
        
        self.w = w.reshape((-1, 1)) # Reshape as column vector
        self.beta, self.T_star, self.T_k = beta, T_star, T_k
        self.y_0, self.sig2_0, self.sig2_star = y_0, sig2_0, sig2_star
        self.b, self.f, self.sp, self.st = b, f, sp, st

        self.n = w.size 
        #self.y = np.arange(y_0-(sig2_0)**(0.5)*sp, 
        #                   y_0+(sig2_0)**(0.5)*sp, st) # Belief grid
        self.y = np.arange(y_0-np.ceil((max(sig2_0,sig2_star))**(0.5))*sp, 
                           y_0+np.ceil((max(sig2_0,sig2_star))**(0.5))*sp, st) # Belief grid
        self.t = np.arange(0, T_star+1, 1) # Tenure grid

        self.H_0 = np.array([norm_distribution(y_0+(i/10), sig2_0) for i in w])
        self.h_0 = np.array([i.pdf for i in self.H_0])
        self.y_star = np.array([i.rvs(1) for i in self.H_0])
        self.xi = np.array([norm_distribution.rvs(i, sig2_star, T_star+1) for i in self.y_star])

        self.c = np.hstack((np.array([0.5*(T_k-self.t[self.t<=T_k])*(i/10) for i in w]), 
                            np.array([(1+b)*i+i*(self.t[self.t>T_k]+1)*f for i in w])))
        self.mu = np.array([((sig2_0)/((self.t+1)*sig2_0+sig2_star))*i+
                            ((self.t*sig2_0+sig2_star)/((self.t+1)*sig2_0+sig2_star))*(k+(j/10)) 
                            for i, j in zip(self.y_star, w) for k in self.y]).reshape(self.n,self.y.size,T_star+1)
        self.sigma = np.tile(((sig2_0)/((self.t+1)*sig2_0+sig2_star))**(2)*sig2_star, (self.n,self.y.size,1))
        self.h_t = np.array([norm_distribution(i,j).pdf for i, j in zip(self.mu,self.sigma)])


    def __repr__(self):
        "Meant for the user of an application to see"

        m = "FireProblem(w, beta={b}, y_0={y0}, sig2_0={s20}, sig2_star={s2s}, "
        m += "T_star={ts}, T_k={tk}, b={ben}, f={fin})"
        return m.format(b=self.beta, y0=self.y_0, s20=self.sig2_0,
                        s2s=self.sig2_star, ts=self.T_star, tk=self.T_k,
                        ben=self.b, fin=self.f)

    def __str__(self):
        "Meant for the programmer to see, as in debugging and development"

        m = """\
        Dynamic learning problem
          - beta (discount parameter)          : {b}
          - y_0 (prior on quality minus wage)  : {y0}
          - H_0 (prior quality)                : Norm(y0+w/10,{s20})
          - sig2_star (noise of signal)        : {s2s}
          - T_star (time horizon in grid)      : {ts}
          - T_k (length of prob period)        : {tk}
          - b (worker benfits)                 : {ben}
          - f (firing fines)                   : {fin}
          - n (number of simulations)          : {num}
        """
        hm, hs = self.H_0.args
        return dedent(m.format(b=self.beta, y0=self.y_0, s20=self.sig2_0,
                               s2s=self.sig2_star, ts=self.T_star, tk=self.T_k,
                               ben=self.b, fin=self.f, num=self.n))

    def bellman_operator(self, v):
        """
        The Bellman operator for the dynamic learning model.

        Parameters
        ----------
        v : array_like(float)
            A 3D NumPy array representing the value function
            Interpretation: :math:`v[k, i, j] = v(\n_k, \y_i, \t_j)`

        Returns
        -------
        new_v : array_like(float)
            The updated value function Tv as an array of shape v.shape

            """
        new_v = np.empty(v.shape)
        for k in range(self.n):
        # keep worker
            v0 = np.tile((self.xi[k,:]-np.tile(self.w[k],self.T_star+1)/10),(self.y.size,1)) + \
                 self.beta*np.multiply(v[k, :, :], np.array([self.h_t[k](i)[j[0],:] for i, j in zip(self.y,enumerate(self.y))]))

        # fire worker
            v1 = np.tile(((self.xi[k,:]-np.tile(self.w[k],self.T_star+1)/10) - self.c[k,:]),(self.y.size,1)) +  \
                 self.beta*np.multiply(v[k, :, :], np.transpose(np.tile(self.h_0[k](self.y),(self.t.size,1))))

            new_v[k, :, :] = np.maximum(v0, v1)
        return new_v

    def get_greedy(self, v):
        """
        Compute optimal actions taking v as the value function.

        Parameters
        ----------
        v : array_like(float)
            A 3D NumPy array representing the value function
            Interpretation: :math:`v[k, i, j] = v(n_k, y_i, t_j)`

        Returns
        -------
        policy : array_like(float)
            A 3D NumPy array, where policy[k, i, j] is the optimal action
            at :math:`(n_k, y_i, t_j)`.

            The optimal action is represented as an integer in the set
            [0, 1] where 0 = 'keep' and 1 = 'fire'

        """
        policy = np.empty(v.shape, dtype=int)
        for k in range(self.n):
             
            v0 = np.tile((self.xi[k,:]-np.tile(self.w[k],self.T_star+1)/10),(self.y.size,1)) + \
                 self.beta*np.multiply(v[k, :, :], np.array([self.h_t[k](i)[j[0],:] for i, j in zip(self.y,enumerate(self.y))]))
            v1 = np.tile(((self.xi[k,:]-np.tile(self.w[k],self.T_star+1)/10) - self.c[k,:]),(self.y.size,1)) +  \
                 self.beta*np.multiply(v[k, :, :], np.transpose(np.tile(self.h_0[k](self.y),(self.t.size,1))))

            action = (v1>v0).astype(int)

            policy[k, :, :] = action

        return policy

    def firm_beliefs(self):
        """
        This function creates an array of firm's beliefs about match
        quality by tenure according to the signals produced.

        Returns n x T_star vector in grid values.

        """
        yt = np.empty([self.n, self.T_star+1])
        for t in range(self.T_star+1):
            if t == 0:
                yt[:, t] = self.y_0
            else:
                yt[:, t] = (((t-1)*self.sig2_0+self.sig2_star)/(t*self.sig2_0+self.sig2_star))*(yt[:, t-1]+(self.w/10).T) + \
                            (self.sig2_0/(t*self.sig2_0+self.sig2_star))*self.xi[:, t]
        tree = spatial.KDTree(self.y.reshape(-1,1))
        beliefs = np.array([self.y.reshape(-1,1)[ tree.query([i])[1] ] for i in yt.reshape(-1,1)]).reshape(self.n,self.T_star+1)
        return beliefs

    def firing_bins(self, policy, beliefs):
        """
        This function creates the vector containing the
        frequency of firings by tenure for a given
        optimal policy and beliefs about match quality.

        Returns 1 x T_star vector.

        """
        fire = np.array([ policy[k,np.where(beliefs[k,j]==self.y.T),j] for k in np.arange(0,self.n) for j in self.t ]).reshape(self.n,self.T_star+1)
        bins = np.bincount([np.nonzero(fire[k,:])[0][0] for k in np.arange(0,self.n) if np.sum(fire[k,:],0)!=0], minlength=self.T_star+1)
        return bins