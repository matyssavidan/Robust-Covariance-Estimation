import numpy as np
from data_gen import generate_t_rowwise, generate_t_cellwise, generate_gaussian_contaminated
from cellwise_SAEM import CellWiseSAEM 
from metrics import evaluate_estimation


def run_single_trial_cellwise(data_config, em_config, seed):

    n = data_config['n']
    p = data_config['p']
    missing_rate = data_config.get('missing_rate')
    dataset_type = data_config.get('type') 
    

    spec_params = data_config.get('specific_params', data_config)


    
    if dataset_type == 't_cellwise':
        nu_gen = spec_params.get('nu')#those are just sanity check to see if row-wise data can 
        #be handled so we use the same nu as initial and generation
        Y, mu_true, Sigma_true, *_ = generate_t_cellwise(
            n=n, p=p, nu=nu_gen, seed=seed, missing=missing_rate
        )

    elif dataset_type == 't_rowwise':
        nu_gen = spec_params.get('nu')  # same remark as above
        Y, mu_true, Sigma_true, *_ = generate_t_rowwise(
            n=n, p=p, nu=nu_gen, seed=seed, missing=missing_rate
        )
        
    elif dataset_type == 'contaminated':
        Y, mu_true, Sigma_true, *_ = generate_gaussian_contaminated(
            n=n, p=p, seed=seed, missing=missing_rate,
            mode=spec_params.get('mode', 'cell_amplified'),
            outlier_rate=spec_params.get('outlier_rate'),
            scale=spec_params.get('scale')
        )
        
    else:
        raise ValueError(f"Type de dataset inconnu : {dataset_type}")

    em_model = CellWiseSAEM(   
        nu=em_config.get('nu'), #   nu is the initial value if estimating
        max_iter=em_config.get('max_iter'),
        mcmc_steps=em_config.get('mcmc_steps'),     
        proposal_scale=em_config.get('proposal_scale'), 
        seed=seed 
    )
    
    em_model.fit(Y)
    
    mu_est, Sigma_est, *_ = em_model.get_params()
    
    results = evaluate_estimation(mu_true, Sigma_true, mu_est, Sigma_est)

    results['n_iter'] = em_model.n_iter_
    results['converged'] = em_model.n_iter_ < em_model.max_iter
    
    return results