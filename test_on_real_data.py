import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import computed_fisher_rao_distance
from student_EM import StudentEMAdaptative
from cellwise_SAEM import CellWiseSAEM


def load_and_prep_intel_data(path, sample_size=500, p=10):

    df = pd.read_csv(path, sep=r'\s+', header=None, 
                    names=['date', 'time', 'epoch', 'moteid', 'temp', 'humid', 'light', 'volt'],
                    on_bad_lines='skip') 


    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df = df.dropna(subset=['temp', 'epoch', 'moteid'])
    
    df = df[(df['temp'] > 5) & (df['temp'] < 40)]


    df_pivot = df.pivot_table(index='epoch', columns='moteid', values='temp')
    

    counts = df_pivot.count().sort_values(ascending=False)
    if len(counts) < p:
        raise ValueError(f"Pas assez de capteurs valides (trouvé {len(counts)}, demandé {p})")
    
    top_sensors = counts.head(p).index 
    df_clean = df_pivot[top_sensors].copy()
    
    df_clean = df_clean.interpolate(method='linear', limit_direction='both').dropna()
    
    if len(df_clean) > sample_size:
        start = len(df_clean) // 2 - (sample_size // 2)
        start = max(0, start)
        X_ground_truth = df_clean.iloc[start : start + sample_size].values
    else:
        X_ground_truth = df_clean.values
        
    return X_ground_truth

def generate_contaminated_data(X_clean, outlier_rate, missing_rate=0.0, scale=10, seed=23 ):

    X_dirty = X_clean.copy()
    n, p = X_clean.shape
    rng = np.random.default_rng(seed)
    
    if outlier_rate > 0:
        true_std = np.nanmedian(np.std(X_clean, axis=0))
        
        magnitude = scale * true_std
        

        mask_outliers = rng.random((n, p)) < outlier_rate
        signs = rng.choice([-1, 1], size=(n, p))
        
        noise = signs * magnitude
        X_dirty[mask_outliers] += noise[mask_outliers]
    
    if missing_rate > 0:
        mask_missing = rng.random((n, p)) < missing_rate
        X_dirty[mask_missing] = np.nan
        
    return X_dirty

def evaluate_algorithms(X_clean, X_dirty, max_iter=30):

    mu_true = np.mean(X_clean, axis=0)
    cov_true = np.cov(X_clean, rowvar=False)
    norm_cov_true = np.linalg.norm(cov_true, 'fro')
    

    em = StudentEMAdaptative(max_iter=max_iter) 
    em.fit(X_dirty)
    

    saem = CellWiseSAEM(max_iter=max_iter)
    saem.fit(X_dirty)
    
    # 3. Calcul Métriques
    results = {}
    
    # Mu Error (L2)
    results['mu_em'] = np.linalg.norm(em.mu - mu_true)
    results['mu_saem'] = np.linalg.norm(saem.mu - mu_true)
    
    # Sigma Error (Frobenius Relative %)
    results['sig_em'] = (np.linalg.norm(em.Sigma - cov_true, 'fro') / norm_cov_true) * 100
    results['sig_saem'] = (np.linalg.norm(saem.Sigma - cov_true, 'fro') / norm_cov_true) * 100
    
    # Sigma Error (Fisher-Rao)
    results['fish_em'] = computed_fisher_rao_distance(cov_true, em.Sigma)
    results['fish_saem'] = computed_fisher_rao_distance(cov_true, saem.Sigma)


    return results

def run_single_experiment(path, p=5, rate=0.0, scale=10, missing_rate=0.0):
    X_clean = load_and_prep_intel_data(path, p=p)
    X_dirty = generate_contaminated_data(X_clean, rate, scale=scale, missing_rate=missing_rate)
    
    print(f"\n--- Running Single Experiment (Rate={rate*100}%, Scale={scale}) ---")
    res = evaluate_algorithms(X_clean, X_dirty)
    
    print("\n" + "="*55)
    print(f"RÉSULTATS FINAUX (Intel Lab, p={p})")
    print("="*55)
    print(f"{'Metric':<25} | {'Standard EM':<12} | {'Proposed SAEM':<12}")
    print("-" * 55)
    print(f"{'Error Mu (L2)':<25} | {res['mu_em']:.4f}       | {res['mu_saem']:.4f}")
    print(f"{'Error Sigma (Frob %)':<25} | {res['sig_em']:.2f}%       | {res['sig_saem']:.2f}%")
    print(f"{'Error Sigma (Fisher-Rao)':<25} | {res['fish_em']:.4f}       | {res['fish_saem']:.4f}")
    print("="*55 + "\n")

def run_sweep_plot_experiment(path, p=5, scales=[10], missing_rate=0.0):
    X_clean = load_and_prep_intel_data(path, p=p)
    rates = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    # Stockage
    history = {'rates': rates, 'em_mu': [], 'saem_mu': [], 'em_sig': [], 'saem_sig': [], 'em_fish': [], 'saem_fish': []}
    
    print(f"\n--- Running Sweep Experiment (0% to 20%) ---")
    for r in rates:
        print(f"  > Processing contamination {r*100}%...")
        X_dirty = generate_contaminated_data(X_clean, r, scale=scales[0], missing_rate=missing_rate)
        res = evaluate_algorithms(X_clean, X_dirty)
        
        history['em_mu'].append(res['mu_em'])
        history['saem_mu'].append(res['mu_saem'])
        history['em_sig'].append(res['sig_em'])
        history['saem_sig'].append(res['sig_saem'])
        history['em_fish'].append(res['fish_em'])
        history['saem_fish'].append(res['fish_saem'])
    
    print(history)
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # 1. Frobenius
    plt.subplot(1, 3, 1)
    plt.plot(rates, history['em_sig'], 'g--o', label='Standard EM', lw=2)
    plt.plot(rates, history['saem_sig'], 'r-s', label='Proposed SAEM', lw=2)
    plt.xlabel('Contamination Rate')
    plt.ylabel('Frobenius Error (%)')
    plt.title(f'Robustness (Frobenius)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Mu
    plt.subplot(1, 3, 2)
    plt.plot(rates, history['em_mu'], 'g--o', label='Standard EM', lw=2)
    plt.plot(rates, history['saem_mu'], 'r-s', label='Proposed SAEM', lw=2)
    plt.xlabel('Contamination Rate')
    plt.ylabel('Mean Error (L2)')
    plt.title('Center Estimation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Fisher
    plt.subplot(1, 3, 3)
    plt.plot(rates, history['em_fish'], 'g--o', label='Standard EM', lw=2)
    plt.plot(rates, history['saem_fish'], 'r-s', label='Proposed SAEM', lw=2)
    plt.xlabel('Contamination Rate')
    plt.ylabel('Fisher-Rao Distance')
    plt.title('Covariance Geometry Preservation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"sweep_plot_p{p}_scale{scales[0]}_miss{int(missing_rate*100)}.png"
    plt.savefig(filename, dpi=300)
    plt.show()

if __name__ == "__main__":

    data_path = r" " ###complete the path 
    
    mode = 'plot' 
    
    if mode == 'single':
        run_single_experiment(data_path, p=5, rate=0.1, missing_rate=0.1, scale=5)
    elif mode == 'plot':
        run_sweep_plot_experiment(data_path, p=10, scales=[10], missing_rate=0.1)