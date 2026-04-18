import numpy as np
from scipy.linalg import eigvalsh, norm

def computed_fisher_rao_distance(Sigma_true, Sigma_est):
    try:
        evals = eigvalsh(Sigma_est, Sigma_true)

        evals = np.clip(evals, a_min=1e-15, a_max=None)
        
        log_evals = np.log(evals)
        distance = np.sqrt(np.sum(log_evals**2))
        return distance
    except np.linalg.LinAlgError:
        return np.inf

def evaluate_estimation(mu_true, Sigma_true, mu_est, Sigma_est):
    err_mu = norm(mu_true - mu_est)
    
    norm_true = norm(Sigma_true, 'fro')
    norm_diff = norm(Sigma_true - Sigma_est, 'fro')
    err_fro = norm_diff / norm_true if norm_true > 0 else norm_diff

    err_fisher = computed_fisher_rao_distance(Sigma_true, Sigma_est)
    
    return {
        'err_mu_l2': err_mu,
        'err_sigma_fro': err_fro,
        'err_sigma_fisher': err_fisher
    }