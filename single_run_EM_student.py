from data_gen import generate_t_rowwise, generate_gaussian_contaminated
from student_EM import StudentEMAdaptative
from metrics import evaluate_estimation

def run_single_trial(data_config, em_config, seed):

    n = data_config['n']
    p = data_config['p']
    missing_rate = data_config.get('missing_rate')
    dataset_type = data_config.get('type') 
    
    if dataset_type == 't_rowwise':
        nu_gen = data_config.get('specific_params').get('nu') #those are just sanity check to see if row-wise data can 
        #be handled so we use the same nu as initial and generation
        Y, mu_true, Sigma_true, *rest = generate_t_rowwise(
            n=n, p=p, nu=nu_gen, seed=seed, missing=missing_rate
        )
        
    elif dataset_type == 'contaminated':
        Y, mu_true, Sigma_true, *rest = generate_gaussian_contaminated(
            n=n, p=p, seed=seed, missing=missing_rate,
            mode=data_config.get('specific_params').get('mode'),
            outlier_rate=data_config.get('specific_params').get('outlier_rate'),
            scale=data_config.get('specific_params').get('scale')
        )

    else:
        raise ValueError(f"Type de dataset inconnu : {dataset_type}")

    em_model = StudentEMAdaptative( 
        nu=em_config.get('nu'), #nu is the initial value if estimating
        max_iter=em_config.get('max_iter'),
        tol=em_config.get('tol'),
        reg=em_config.get('reg')
    )
    
    em_model.fit(Y)
    
    mu_est, Sigma_est, *_ = em_model.get_params()

    results = evaluate_estimation(mu_true, Sigma_true, mu_est, Sigma_est)
 
    results['n_iter'] = em_model.n_iter_
    results['converged'] = em_model.n_iter_ < em_model.max_iter
    
    return results