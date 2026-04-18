import numpy as np
import matplotlib.pyplot as plt


def random_covariance_realistic(p=5, seed=None, var_range=(1.0, 3.0), corr_strength=0.5):

    rng = np.random.default_rng(seed)
    variances = rng.uniform(*var_range, size=p)

    A = rng.normal(size=(p, p))
    Corr = A @ A.T
    Corr /= np.outer(np.sqrt(np.diag(Corr)), np.sqrt(np.diag(Corr)))

    Corr = (1 - corr_strength) * np.eye(p) + corr_strength * Corr

    Psi = np.outer(np.sqrt(variances), np.sqrt(variances)) * Corr
    return Psi

def generate_gaussian(n=500, p=2, seed=None, missing=0.0):

    rng = np.random.default_rng(seed)
    mu = np.zeros(p)
    Psi = random_covariance_realistic(p, seed=seed)

    X = rng.multivariate_normal(mu, Psi, size=n)
    Y = X.copy()

    missing_mask = np.zeros_like(Y, dtype=bool)
    if missing > 0:
        missing_mask = rng.random(Y.shape) < missing
        Y[missing_mask] = np.nan

    mask_obs = ~missing_mask

    return Y, mu, Psi, X, mask_obs

def generate_t_rowwise(n=500, p=2, nu=3, seed=None, missing=0.0):

    rng = np.random.default_rng(seed)
    mu = np.zeros(p)
    Psi = random_covariance_realistic(p, seed=seed)

    X = rng.multivariate_normal(mu, Psi, size=n)
    tau = rng.gamma(shape=nu/2, scale=2/nu, size=n)[:, np.newaxis]
    Y = X / np.sqrt(tau)

    missing_mask = np.zeros_like(Y, dtype=bool)
    if missing > 0:
        missing_mask = rng.random(Y.shape) < missing
        Y[missing_mask] = np.nan
    mask_obs = ~missing_mask

    return Y, mu, Psi, X, mask_obs

def generate_t_cellwise(n=500, p=2, nu=3, seed=None, missing=0.0):

    rng = np.random.default_rng(seed)
    mu = np.zeros(p)
    Psi = random_covariance_realistic(p, seed=seed)

    X = rng.multivariate_normal(mu, Psi, size=n)
    tau = rng.gamma(shape=nu/2, scale=2/nu, size=(n, p))
    Y = X / np.sqrt(tau)

    missing_mask = np.zeros_like(Y, dtype=bool)
    if missing > 0:
        missing_mask = rng.random(Y.shape) < missing
        Y[missing_mask] = np.nan
    mask_obs = ~missing_mask

    return Y, mu, Psi, X, mask_obs

def generate_gaussian_contaminated(
    n=500, p=5, seed=None, missing=0.0,
    mode="cell_amplified",             
    outlier_rate=0.0,               
    scale=10.0,                      
    shift=10.0,                       
):

    rng = np.random.default_rng(seed)
    mu = np.zeros(p)
    Psi = random_covariance_realistic(p, seed=seed)


    X = rng.multivariate_normal(mu, Psi, size=n)
    Y = X.copy()
    sigma = np.sqrt(np.diag(Psi))
    mask_outliers = np.zeros((n, p), dtype=bool)


    if outlier_rate > 0:
        if mode == "cell_amplified":
            m = int(outlier_rate * n * p)
            idx_flat = rng.choice(n * p, size=m, replace=False)
            for idxf in idx_flat:
                i, j = divmod(idxf, p)
                Y[i, j] = rng.normal(0.0, scale * sigma[j])
                mask_outliers[i, j] = True

        elif mode == "cell_shift":
            m = int(outlier_rate * n * p)
            sgn = rng.choice([-1.0, 1.0], size=m)
            idx_flat = rng.choice(n * p, size=m, replace=False)
            for k, idxf in enumerate(idx_flat):
                i, j = divmod(idxf, p)
                Y[i, j] = sgn[k] * shift * sigma[j] + rng.normal(0.0, 0.5 * sigma[j])
                mask_outliers[i, j] = True
        else:
            raise ValueError("mode inconnu")

    missing_mask = np.zeros_like(Y, dtype=bool)
    if missing > 0:
        missing_mask = rng.random(Y.shape) < missing
        Y[missing_mask] = np.nan
    mask_obs = ~missing_mask

    return Y, mu, Psi, X, mask_obs, mask_outliers




