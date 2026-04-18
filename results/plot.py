import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


files = {
    "EM Student-t": "n500-p20-s10-20runs-EMstu.csv",
    "Proposed Cell-wise SAEM": "n500-p20-s10-20runs-SAEM.csv",
    "EM Gaussian": "n500-p20-s10-20runs-EMgau.csv",
    "MICE + robust SCM": "n500-p20-s10-20runs-MICE.csv",
    "Mean imputation + SCM": "n500-p20-s10-20runs-MEAN.csv",
}


dfs = []

for algo, file in files.items():
    df = pd.read_csv(file)
    df["Algorithm"] = algo
    df["outlier_rate"] = pd.to_numeric(df["outlier_rate"])
    df["missing_rate"] = pd.to_numeric(df["missing_rate"])
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

group_cols = ["Algorithm", "missing_rate", "outlier_rate"]

df_mean = (
    df_all
    .groupby(group_cols, as_index=False)
    .mean(numeric_only=True)
)

df_mean["Missing Rate"] = df_mean["missing_rate"].apply(
    lambda x: f"{int(100*x)}%"
)

df_mean = df_mean[df_mean["missing_rate"] > 0]


sns.set_theme(style="whitegrid", context="talk")

palette = {
    "EM Student-t": "#1b1b1b",
    "Proposed Cell-wise SAEM": "#004488",
    "EM Gaussian": "#BB5566",
    "MICE + robust SCM": "#228833",
    "Mean imputation + SCM": "#CCBB44",
}

metrics = [
    ("err_mu", "Estimation of μ", r"$\|\hat{\mu}-\mu\|_2$"),
    ("frobenius_rel", "Estimation of Σ (Frobenius)", "Relative Frobenius error"),
    ("fisher_dist", "Geometry of Σ (Fisher)", "Fisher distance"),
]


for col, title, ylabel in metrics:

    fig, ax = plt.subplots(figsize=(8.5, 6))

    sns.lineplot(
        data=df_mean,
        x="outlier_rate",
        y=col,
        hue="Algorithm",
        style="Missing Rate",
        palette=palette,
        markers=True,
        dashes=False,
        linewidth=1.4,
        markersize=7,
        alpha=0.85,
        errorbar=None,
        ax=ax
    )

  
    ax.set_xlim(left=0.025) 
    ax.xaxis.set_major_locator(MultipleLocator(0.025))   
    ax.xaxis.set_minor_locator(MultipleLocator(0.0125))  
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


    ax.set_yscale("log")

    ax.set_xlabel("Outlier rate", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.set_title(title, fontweight="bold", pad=12)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig.tight_layout()
    fig.savefig(f"plot_{col}.png", dpi=300, bbox_inches="tight")
    plt.show()
