from data_gen import generate_gaussian_contaminated
from metrics import evaluate_estimation
import numpy as np

def mean_impute(Y):
    X = Y.copy()
    col_means = np.nanmean(X, axis=0)
    inds = np.isnan(X)
    X[inds] = np.take(col_means, np.where(inds)[1])
    return X

def run_single_trial(data_config, em_config, seed):

    Y, mu_true, Sigma_true, *_ = generate_gaussian_contaminated(
        n=data_config['n'],
        p=data_config['p'],
        seed=seed,
        missing=data_config['missing_rate'],
        mode=data_config['specific_params']['mode'],
        outlier_rate=data_config['specific_params']['outlier_rate'],
        scale=data_config['specific_params']['scale']
    )

    X = mean_impute(Y)

    mu_est = X.mean(axis=0)
    Sigma_est = np.cov(X.T, bias=True)

    results = evaluate_estimation(mu_true, Sigma_true, mu_est, Sigma_est)
    results['n_iter'] = None
    results['converged'] = True

    return results
