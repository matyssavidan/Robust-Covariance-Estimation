from data_gen import generate_gaussian_contaminated
from gaussian_EM import GaussianEM
from metrics import evaluate_estimation

def run_single_trial(data_config, em_config, seed):

    Y, mu_true, Sigma_true, *_ = generate_gaussian_contaminated(
        n=data_config['n'],
        p=data_config['p'],
        seed=seed,
        missing=data_config.get('missing_rate'),
        mode=data_config['specific_params']['mode'],
        outlier_rate=data_config['specific_params']['outlier_rate'],
        scale=data_config['specific_params']['scale']
    )

    em_model = GaussianEM(
        max_iter=em_config['max_iter'],
        tol=em_config['tol'],
        reg=em_config['reg']
    )

    em_model.fit(Y)
    mu_est, Sigma_est = em_model.get_params()

    results = evaluate_estimation(mu_true, Sigma_true, mu_est, Sigma_est)
    results['n_iter'] = em_model.n_iter_
    results['converged'] = em_model.n_iter_ < em_model.max_iter

    return results
