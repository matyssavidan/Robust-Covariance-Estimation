from single_run_EM_student import run_single_trial
    

if __name__ == "__main__":
    
    N = 500
    P = 5
    MISSING = 0.1 
    SEED = 65
    OUTLIER_RATE = 0.1
    SCALE = 10.0
    NU = 3.0

    config_ideal = {
        'n': N, 'p': P, 'missing_rate': MISSING,
        'type': 't_rowwise',
        'specific_params': {'nu': NU}
    }
    
    config_hard = {
         'n': N, 'p': P, 'missing_rate': MISSING,
         'type': 'contaminated',
         'specific_params': {'mode': 'cell_amplified', 'outlier_rate': OUTLIER_RATE, 'scale': SCALE}
     }
    
    em_params = {'nu': NU, 'max_iter': 150, 'tol': 1e-5, 'reg': 1e-6}
    

    res_ideal = run_single_trial(config_ideal, em_params, seed=SEED)
    print(f"1.1.  Ideal (Row-wise) -> Fisher Error: {res_ideal['err_sigma_fisher']:.4f}")
    print(f"1.2.  Ideal (Row-wise) -> Frobenius: {res_ideal['err_sigma_fro']:.4f}")
    
    res_hard = run_single_trial(config_hard, em_params, seed=SEED)
    print(f"2.1.  Cell-wise (Hard) -> Fisher Error: {res_hard['err_sigma_fisher']:.4f}")
    print(f"2.2.  Cell-wise (Hard) -> Frobenius: {res_hard['err_sigma_fro']:.4f}")
