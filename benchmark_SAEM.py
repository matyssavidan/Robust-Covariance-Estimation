import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ---------------------------------------

import pandas as pd
import time
import itertools
from datetime import datetime
from joblib import Parallel, delayed

try:
    from single_run_SAEM import run_single_trial_cellwise
except ImportError:
    raise ImportError("Can't find the file")


N_RUNS = 20
DATA_N = 500 
DATA_P = 20
SCALE_OUTLIER = 10.0

param_grid = {
    'outlier_rate': [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20],  
    'missing_rate': [0.0, 0.10, 0.20],       
}
print(param_grid)

FIXED_SAEM_PARAMS = {
    'nu': 10.0, # degrees of freedom for the student-t, initialization 
    'max_iter': 30,  
    'mcmc_steps': 20, 
    'proposal_scale': 3.5
}
print(FIXED_SAEM_PARAMS)


def execute_one_run(tid, outlier_r, missing_r, seed):

    data_conf = {
        'n': DATA_N, 
        'p': DATA_P, 
        'missing_rate': missing_r,
        'type': 'contaminated',  
        'specific_params': {
            'mode': 'cell_amplified', 
            'outlier_rate': outlier_r, 
            'scale': SCALE_OUTLIER  
        }
    }
    
    try:
        t0 = time.time()

        res = run_single_trial_cellwise(data_conf, FIXED_SAEM_PARAMS, seed=seed)
        
        duration = time.time() - t0

        return {
            'tid': tid,
            'outlier_rate': outlier_r,
            'missing_rate': missing_r,
            'seed': seed,
            'duration': duration,
            
            'fisher_dist': res.get('err_sigma_fisher', None),
            'frobenius_rel': res.get('err_sigma_fro', None),
            'err_mu': res.get('err_mu_l2', None),
            'n_iter': res.get('n_iter', None)
        }

    except Exception as e:
        return {'error': str(e), 'outlier_rate': outlier_r, 'missing_rate': missing_r}


if __name__ == "__main__":

    keys, values = zip(*param_grid.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    tasks = []
    tid = 0
    for config in configurations:
        for i in range(N_RUNS):
            seed = tid 
            tasks.append((tid, config['outlier_rate'], config['missing_rate'], seed))
            tid += 1
            
    print(f" Benchmark student-t* SAEM| {len(tasks)} ")


    start_time = time.time()
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(execute_one_run)(*t) for t in tasks
    )
    total_time = time.time() - start_time

    print(f" It took {total_time:.1f} s")

    df = pd.DataFrame(results)
    
    if 'error' in df.columns and df['error'].notna().any():
        print("\nErrors founded :")
        print(df[df['error'].notna()]['error'].unique())
        df_clean = df[df['error'].isna()]
    else:
        df_clean = df

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"benchmark_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Sauvegardé : {filename}")

    if not df_clean.empty:
        print(df_clean.groupby(['outlier_rate', 'missing_rate'])[['fisher_dist', 'frobenius_rel']].mean())