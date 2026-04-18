from single_run_SAEM import run_single_trial_cellwise

if __name__ == "__main__":
    
    N = 500
    P = 20
    MISSING = 0.1 
    SEED = 42 #for reproducibility
    OUTLIER_RATE = 0.1
    SCALE = 10.0 # scale of the cell-wise outliers
    NU = 3.0 # degrees of freedom for ideal t-distribution, 
    #here because it is only a sanity check we use the same as initialization and data generation

    config_ideal = {
        'n': N, 'p': P, 'missing_rate': MISSING,
        'type': 't_cellwise',
        'specific_params': {'nu': NU}
    }

    config_compatible = {
        'n': N, 'p': P, 'missing_rate': MISSING,
        'type': 't_rowwise',
        'spec ific_params': {'nu': NU}
    }
    
    config_hard = {
        'n': N, 'p': P, 'missing_rate': MISSING,
        'type': 'contaminated',
        'specific_params': {
            'mode': 'cell_amplified', 
            'outlier_rate': OUTLIER_RATE, 
            'scale': SCALE         
        }
    }

    saem_params = {
        'nu': 10.0, 
        'max_iter': 30,   
        'mcmc_steps': 20, 
        'proposal_scale': 3.5
    }
    
    # res_ideal = run_single_trial_cellwise(config_ideal, saem_params, seed=SEED)
    # print(f"    Fisher Error : {res_ideal['err_sigma_fisher']:.4f}")
    # print(f"    Frobenius    : {res_ideal['err_sigma_fro']:.4f}")
    # print(f"    Iterations   : {res_ideal['n_iter']}")
    
    # res_comp = run_single_trial_cellwise(config_compatible, saem_params, seed=SEED)
    # print(f"    Fisher Error : {res_comp['err_sigma_fisher']:.4f}")
    # print(f"    Frobenius    : {res_comp['err_sigma_fro']:.4f}")

    res_hard = run_single_trial_cellwise(config_hard, saem_params, seed=SEED)
    print(f"    Fisher Error : {res_hard['err_sigma_fisher']:.4f}")
    print(f"    Frobenius    : {res_hard['err_sigma_fro']:.4f}")
    