from data_gen import generate_gaussian_contaminated
from metrics import evaluate_estimation
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def mice_impute(Y, seed=0):
    imp = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=20,
        random_state=seed,
        sample_posterior=False
    )
    return imp.fit_transform(Y)


def tyler_covariance(X, max_iter=100, tol=1e-6, reg=1e-6):
    n, p = X.shape
    Sigma = np.eye(p)

    for _ in range(max_iter):
        Sigma_inv = np.linalg.inv(Sigma)
        q = np.einsum('ij,jk,ik->i', X, Sigma_inv, X)
        q = np.maximum(q, 1e-12)

        Sigma_new = (p / n) * (X.T / q) @ X
        Sigma_new /= np.trace(Sigma_new) / p
        Sigma_new += reg * np.eye(p)

        if np.linalg.norm(Sigma_new - Sigma) < tol:
            break
        Sigma = Sigma_new

    mu = np.median(X, axis=0)
    return mu, Sigma


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

    X = mice_impute(Y, seed=seed)
    mu_est, Sigma_est = tyler_covariance(X)

    results = evaluate_estimation(mu_true, Sigma_true, mu_est, Sigma_est)
    results['n_iter'] = None
    results['converged'] = True

    return results
