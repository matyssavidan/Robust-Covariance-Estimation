import numpy as np
from scipy.special import digamma
from scipy.optimize import brentq  

class StudentEMAdaptative:
    
    def __init__(self, nu=10.0, max_iter=100, tol=1e-5, reg=1e-6): # nu is just an initial value 
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        
        self.mu = None
        self.Sigma = None
        self.weights_ = None 
        self.n_iter_ = 0
        self.history_ = []

    def _solve_nu(self, mean_tau, mean_log_tau):
        
        S = 1 + mean_log_tau - mean_tau

        def func(v):
            return np.log(v/2) - digamma(v/2) + S
        
        try:
            if func(0.1) * func(200.0) > 0:
                 return 0.1 if abs(func(0.1)) < abs(func(200.0)) else 200.0
            
            new_nu = brentq(func, 0.1, 200.0)
            return new_nu
        except (ValueError, RuntimeError):
            return self.nu 

    def fit(self, X):
        
        n, p = X.shape
        X_init = X.copy()
        self.mu = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X_init[inds] = np.take(self.mu, inds[1])
        mad = np.nanmedian(np.abs(X - self.mu), axis=0)
        sig_rob = np.maximum(1.4826 * mad, 1e-4)
        self.Sigma = np.diag(sig_rob**2) + np.eye(p) * self.reg # robust initial covariance

        for iteration in range(self.max_iter):
            prev_mu = self.mu.copy()
            prev_Sigma = self.Sigma.copy()
            prev_nu = self.nu
            
            W = np.zeros(n)             
            E_log_tau = np.zeros(n)     
            X_hat = np.zeros((n, p))    
            Corrections = np.zeros((p, p)) 
            
            for i in range(n):
                x_i = X[i]
                mask_obs = ~np.isnan(x_i)
                mask_miss = ~mask_obs
                
                idx_obs = np.where(mask_obs)[0]
                idx_miss = np.where(mask_miss)[0]
                
                if len(idx_obs) == 0:
                    W[i] = 1.0
                    E_log_tau[i] = 0.0 
                    X_hat[i] = self.mu
                    Corrections += self.Sigma 
                    continue

                mu_obs = self.mu[idx_obs]
                mu_miss = self.mu[idx_miss]
                
                Sigma_oo = self.Sigma[np.ix_(idx_obs, idx_obs)]
                Sigma_om = self.Sigma[np.ix_(idx_obs, idx_miss)]
                Sigma_mo = self.Sigma[np.ix_(idx_miss, idx_obs)]
                Sigma_mm = self.Sigma[np.ix_(idx_miss, idx_miss)]
                
                try:
                    L = np.linalg.cholesky(Sigma_oo)
                    diff_obs = x_i[idx_obs] - mu_obs
                    z = np.linalg.solve(L, diff_obs)
                    delta_sq = np.dot(z, z)
                    
                    L_inv = np.linalg.solve(L, np.eye(len(idx_obs)))
                    Sigma_oo_inv = L_inv.T @ L_inv
                    
                except np.linalg.LinAlgError:
                    delta_sq = 0.0
                    Sigma_oo_inv = np.eye(len(idx_obs))

                d_obs = len(idx_obs)
                
                alpha_post = (self.nu + d_obs) / 2.0
                beta_post = (self.nu + delta_sq) / 2.0
                
                w_i = alpha_post / beta_post
                W[i] = w_i
                
                e_log_tau_i = digamma(alpha_post) - np.log(beta_post)
                E_log_tau[i] = e_log_tau_i
                
                x_hat_row = np.zeros(p)
                x_hat_row[idx_obs] = x_i[idx_obs]
                
                if len(idx_miss) > 0:
                    cond_mean = mu_miss + Sigma_mo @ Sigma_oo_inv @ diff_obs
                    x_hat_row[idx_miss] = cond_mean
                    cond_cov = Sigma_mm - Sigma_mo @ Sigma_oo_inv @ Sigma_om
                    Corrections[np.ix_(idx_miss, idx_miss)] += cond_cov
                
                X_hat[i] = x_hat_row

            Sw = np.sum(W)
            
            new_mu = np.sum(W[:, None] * X_hat, axis=0) / Sw

            X_centered = X_hat - new_mu
            S_scatter = (X_centered.T * W) @ X_centered
            new_Sigma = (S_scatter + Corrections) / n
            new_Sigma += np.eye(p) * self.reg 
            

            mean_tau = np.mean(W)
            mean_log_tau = np.mean(E_log_tau)

            new_nu = self._solve_nu(mean_tau, mean_log_tau)
            
            diff = (np.linalg.norm(new_mu - prev_mu) + 
                    np.linalg.norm(new_Sigma - prev_Sigma, 'fro') + 
                    abs(new_nu - prev_nu)) 
            
            self.history_.append(diff)
            self.mu = new_mu
            self.Sigma = new_Sigma
            self.nu = new_nu
            self.weights_ = W
            self.n_iter_ = iteration + 1
            
            if diff < self.tol:
                break
                
        return self

    def get_params(self):
        return self.mu, self.Sigma, self.nu