import numpy as np
from scipy import linalg
from scipy.special import digamma
from scipy.optimize import brentq

def log_conditional_tau_article(tau_val, Y_i, mu, tau_i_vec, j, Sigma, nu, mask_i):
    if tau_val <= 0:
        return -np.inf

    
    alpha = nu / 2.0
    log_prior = (alpha - 1) * np.log(tau_val) - alpha * tau_val

   
    if not mask_i[j]:
        return log_prior

  
    tau_local = tau_i_vec.copy()
    tau_local[j] = tau_val
    
    obs = np.where(mask_i)[0]
    
    diff = Y_i[obs] - mu[obs]
    sqrt_tau = np.sqrt(tau_local[obs])
    
    
    Sigma_oo = Sigma[np.ix_(obs, obs)]
    try:
        L = linalg.cholesky(Sigma_oo, lower=True)
    
        z = linalg.solve_triangular(L, sqrt_tau * diff, lower=True)
        quad = np.dot(z, z)
        
        log_det_weighted = 2.0 * np.sum(np.log(np.diag(L))) - np.sum(np.log(tau_local[obs]))
        
        return log_prior - 0.5 * (log_det_weighted + quad)
    except linalg.LinAlgError:
        return -np.inf

class CellWiseSAEM:
    """
    Implémentation exacte de l'algorithme SAEM de l'article[cite: 1, 62].
    """
    def __init__(self, nu=10.0, max_iter=50, mcmc_steps=5, 
                 proposal_scale=0.5, seed=None):
        self.nu = nu
        self.max_iter = max_iter
        self.mcmc_steps = mcmc_steps
        self.proposal_scale = proposal_scale
        self.rng = np.random.default_rng(seed)
        self.n_iter_ = 0

        # Paramètres
        self.mu = None
        self.Sigma = None
        self.Tau = None
        
        
        self.S_Sigma = None   
        self.S_W = None       
        self.S_Wy = None      
        self.S_tau = 0.0      
        self.S_log_tau = 0.0  

    def _compute_conditional_moments(self, Y_i, mu, tau_i, Sigma, mask_i):
        """
        Calcul de m_i (Eq 8) et C_i (Eq 9) par Rao-Blackwellization.
        """
        p = len(mu)
        obs = np.where(mask_i)[0]
        
        if len(obs) == 0:
            return np.zeros(p), Sigma.copy()
            
        Sigma_oo = Sigma[np.ix_(obs, obs)]
        Sigma_all_o = Sigma[:, obs]
        
        try:
            
            K = linalg.solve(Sigma_oo, Sigma_all_o.T, assume_a='pos').T 
            
          
            C_i = Sigma - K @ Sigma_all_o.T
            C_i = 0.5 * (C_i + C_i.T) 
            
            
            res_weighted = np.sqrt(tau_i[obs]) * (Y_i[obs] - mu[obs])
            m_i = K @ res_weighted
            
            return m_i, C_i
        except linalg.LinAlgError:
            return np.zeros(p), Sigma.copy()

    def fit(self, X):
        n, p = X.shape
        mask_obs = ~np.isnan(X)
        N_obs_total = np.sum(mask_obs) 

        
        self.mu = np.nanmedian(X, axis=0)
        self.Sigma = np.diag(np.nanvar(X, axis=0) + 1e-3)
        self.Tau = np.ones((n, p))

        
        self.S_Sigma = np.zeros((p, p))
        self.S_W = np.zeros((p, p))
        self.S_Wy = np.zeros(p)
        
        for k in range(self.max_iter):
            
            gamma = 1.0 if k < 0.5 * self.max_iter else (k - 0.5 * self.max_iter + 1)**(-0.7)

            # --- 1. Simulation Step (MCMC) ---
            for _ in range(self.mcmc_steps):
                for i in range(n):
                    obs_idx = np.where(mask_obs[i])[0]
                    for j in obs_idx:
                        curr = self.Tau[i, j]
                        prop = curr * np.exp(self.rng.normal(0, self.proposal_scale))
                        
                        lp_curr = log_conditional_tau_article(curr, X[i], self.mu, self.Tau[i], j, self.Sigma, self.nu, mask_obs[i])
                        lp_prop = log_conditional_tau_article(prop, X[i], self.mu, self.Tau[i], j, self.Sigma, self.nu, mask_obs[i])
                        
                        
                        if np.log(self.rng.random()) < (lp_prop - lp_curr + np.log(prop) - np.log(curr)):
                            self.Tau[i, j] = prop
            
            
            miss_mask = ~mask_obs
            if np.any(miss_mask):
                self.Tau[miss_mask] = self.rng.gamma(self.nu/2, 2/self.nu, size=np.sum(miss_mask))

            # --- 2. Rao-Blackwellization & Stochastic Approximation ---
            inv_Sigma = linalg.solve(self.Sigma + 1e-6*np.eye(p), np.eye(p))
            
            curr_S_Sigma = np.zeros((p, p))
            curr_S_W = np.zeros((p, p))
            curr_S_Wy = np.zeros(p)
            
            for i in range(n):
                m_i, C_i = self._compute_conditional_moments(X[i], self.mu, self.Tau[i], self.Sigma, mask_obs[i])
                
                
                curr_S_Sigma += (C_i + np.outer(m_i, m_i)) / n
                
                
                sqrt_tau = np.sqrt(self.Tau[i])
                W_i = (sqrt_tau[:, None] * inv_Sigma * sqrt_tau[None, :]) # Eq (5)
                
                
                y_imp = self.mu + (1.0 / sqrt_tau) * m_i
                
                curr_S_W += W_i
                curr_S_Wy += W_i @ y_imp

           
            self.S_Sigma = (1 - gamma) * self.S_Sigma + gamma * curr_S_Sigma
            self.S_W = (1 - gamma) * self.S_W + gamma * curr_S_W
            self.S_Wy = (1 - gamma) * self.S_Wy + gamma * curr_S_Wy
            
             
            obs_tau = self.Tau[mask_obs]
            curr_s_tau = np.sum(obs_tau)
            curr_s_log_tau = np.sum(np.log(obs_tau + 1e-12))
            
            if k == 0:
                self.S_tau, self.S_log_tau = curr_s_tau, curr_s_log_tau
            else:
                self.S_tau = (1 - gamma) * self.S_tau + gamma * curr_s_tau
                self.S_log_tau = (1 - gamma) * self.S_log_tau + gamma * curr_s_log_tau

            
            self.Sigma = 0.5 * (self.S_Sigma + self.S_Sigma.T)
            
            
            self.mu = linalg.solve(self.S_W, self.S_Wy, assume_a='pos')
            
            
            self.nu = self._solve_nu(self.S_tau, self.S_log_tau, N_obs_total)

            self.n_iter_ = k + 1
            
        return self

    def _solve_nu(self, s_tau, s_log_tau, N):
        mean_tau = s_tau / N
        mean_log_tau = s_log_tau / N
        S = 1 + mean_log_tau - mean_tau
        def f(v):
            return np.log(v/2) - digamma(v/2) + S
        try:
            return brentq(f, 0.1, 100.0)
        except:
            return self.nu

    def get_params(self):
        return self.mu, self.Sigma, self.nu