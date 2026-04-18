import numpy as np

class GaussianEM:
    def __init__(self, max_iter=100, tol=1e-5, reg=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg

    def fit(self, Y):
        n, p = Y.shape
        mask = ~np.isnan(Y)

        # initialisation
        mu = np.nanmean(Y, axis=0)
        Sigma = np.nan_to_num(np.cov(np.nan_to_num(Y).T)) + self.reg * np.eye(p)

        ll_old = -np.inf

        for it in range(self.max_iter):
            # === E-step ===
            Ex = np.zeros((n, p))
            Exx = np.zeros((p, p))

            for i in range(n):
                obs = mask[i]
                mis = ~obs

                if np.all(obs):
                    xi = Y[i]
                    Ex[i] = xi
                    Exx += np.outer(xi, xi)
                else:
                    Σoo = Sigma[np.ix_(obs, obs)]
                    Σmo = Sigma[np.ix_(mis, obs)]
                    Σmm = Sigma[np.ix_(mis, mis)]

                    inv_Σoo = np.linalg.inv(Σoo)

                    xo = Y[i, obs]
                    μo = mu[obs]
                    μm = mu[mis]

                    xm_cond = μm + Σmo @ inv_Σoo @ (xo - μo)
                    Σm_cond = Σmm - Σmo @ inv_Σoo @ Σmo.T

                    xi = np.zeros(p)
                    xi[obs] = xo
                    xi[mis] = xm_cond
                    Ex[i] = xi

                    Exx += np.outer(xi, xi)
                    Exx[np.ix_(mis, mis)] += Σm_cond

            # === M-step ===
            mu_new = Ex.mean(axis=0)
            Sigma_new = Exx / n - np.outer(mu_new, mu_new)
            Sigma_new += self.reg * np.eye(p)

            # convergence
            diff = np.linalg.norm(mu_new - mu) + np.linalg.norm(Sigma_new - Sigma)
            mu, Sigma = mu_new, Sigma_new

            if diff < self.tol:
                break

        self.mu_ = mu
        self.Sigma_ = Sigma
        self.n_iter_ = it + 1

    def get_params(self):
        return self.mu_, self.Sigma_
