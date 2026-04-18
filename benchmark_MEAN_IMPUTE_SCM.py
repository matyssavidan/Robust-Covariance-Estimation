import pandas as pd
import time
import itertools
from datetime import datetime
from joblib import Parallel, delayed

try:
    from single_run_mean_impute_scm import run_single_trial
except ImportError:
    raise ImportError("Can't find single_run_mean_impute_scm.py")

N_RUNS = 20
DATA_N = 500
DATA_P = 20
SCALE_OUTLIER = 10.0

param_grid = {
    'outlier_rate': [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20],
    'missing_rate': [0.0, 0.10, 0.20],
}

FIXED_PARAMS = {}

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
        res = run_single_trial(data_conf, FIXED_PARAMS, seed)
        duration = time.time() - t0

        return {
            'tid': tid,
            'algo': 'MeanImpute + SCM',
            'outlier_rate': outlier_r,
            'missing_rate': missing_r,
            'seed': seed,
            'duration': duration,
            'fisher_dist': res.get('err_sigma_fisher'),
            'frobenius_rel': res.get('err_sigma_fro'),
            'err_mu': res.get('err_mu_l2'),
            'n_iter': None,
            'converged': True
        }

    except Exception as e:
        return {
            'tid': tid,
            'algo': 'MeanImpute + SCM',
            'outlier_rate': outlier_r,
            'missing_rate': missing_r,
            'seed': seed,
            'error': str(e)
        }

if __name__ == "__main__":

    keys, values = zip(*param_grid.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tasks = []
    tid = 0
    for config in configurations:
        for _ in range(N_RUNS):
            tasks.append((tid, config['outlier_rate'], config['missing_rate'], tid))
            tid += 1

    print(f" Benchmark Mean Imputation + SCM | {len(tasks)} runs")

    start = time.time()
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(execute_one_run)(*t) for t in tasks
    )
    print(f" Done in {time.time() - start:.1f}s")

    df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"RESULTS_MEAN_IMPUTE_SCM_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f" Saved: {filename}")
