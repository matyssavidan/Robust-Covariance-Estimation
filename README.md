# Robust Covariance Estimation for Cell-wise Contaminated and Partially Observed Data via Expectation Maximization Algorithm
Code used for the so named article

Benchmarks several algorithms for estimating mean and covariance matrices
on contaminated, partially observed multivariate data.

## Project Structure

| File | Role |
| `data_gen.py` | Data generation (Gaussian, Student-t, contaminated) |
| `metrics.py` | Evaluation metrics (Fisher-Rao, Frobenius, L2) |
| `gaussian_EM.py` | Standard Gaussian EM |
| `cellwise_SAEM.py` | Cellwise robust SAEM |
| `single_run_EM_gaussian.py` | Single trial runner for Gaussian EM |
| `benchmark_EM_gaussian.py` | Benchmark: Gaussian EM |
| `benchmark_EM_student.py` | Benchmark: Student-t EM |
| `benchmark_SAEM.py` | Benchmark: Cellwise SAEM |
| `benchmark_MEAN_IMPUTE_SCM.py` | Benchmark: Mean Imputation + SCM |
| `benchmark_MICE_SCM.py` | Benchmark: MICE + Robust SCM |
| `sanity_check_*.py` | Quick correctness checks |
| `download_intel_berkley.py` | Download Intel Berkeley sensor dataset |

## Algorithms

| Script | Algorithm |
| `benchmark_EM_gaussian.py` | Gaussian EM (standard) 
| `benchmark_EM_student.py` | Student-t EM (row-wise) 
| `benchmark_SAEM.py` | Cellwise SAEM (Student-t*) 
| `benchmark_MEAN_IMPUTE_SCM.py` | Mean Imputation + SCM 
| `benchmark_MICE_SCM.py` | MICE + Robust SCM 

## Data Generation (`data_gen.py`)

Four regimes: clean Gaussian, row-wise Student-t, cellwise Student-t,
and Gaussian with injected cellwise outliers (`cell_amplified` / `cell_shift`).

Default benchmark settings: n=500, p=20, outlier_rate ∈ [0, 20%],
missing_rate ∈ {0%, 10%, 20%}, scale=10×σ, N_RUNS=20 per config.

## Metrics (`metrics.py`)

- **Fisher-Rao distance** — geodesic on the SPD manifold
- **Frobenius relative error** — ‖Σ_true − Σ_est‖_F / ‖Σ_true‖_F
- **L2 error on mean** — ‖μ_true − μ_est‖₂

## Running

```bash
python benchmark_EM_gaussian.py     
python benchmark_SAEM.py             
python benchmark_EM_student.py
python benchmark_MEAN_IMPUTE_SCM.py
python benchmark_MICE_SCM.py

# Sanity checks (fast, single run)
python sanity_check_SAEM.py
python sanity_check_EM_student.py
```

## Notes

- All benchmarks use `joblib.Parallel(n_jobs=-1)`.
- `benchmark_SAEM.py` pins thread counts to 1 to avoid over-subscription.
- `seed = tid` ensures full reproducibility across runs.