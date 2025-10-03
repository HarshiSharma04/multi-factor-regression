"""
Matplotlib plotting functions with a pastel aesthetic theme.
Each plot gets its own figure (no subplots).
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

PASTEL_PALETTE = [
    "#ffd6e0", "#cfe7ff", "#f3e6ff", "#d6ffe7", "#fff3cf",
    "#e7f0ff", "#ffdeda", "#fbe6ff", "#e6fff4", "#ffe7d6"
]

def _pastel_colors(n):
    # repeat palette if fewer colors requested
    palette = PASTEL_PALETTE
    out = []
    for i in range(n):
        out.append(palette[i % len(palette)])
    return out

def plot_stacked_contributions(contrib_df, out_path):
    """
    Stacked area plot of factor contributions over time.
    contrib_df: DataFrame of factors (columns) indexed by date
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    cols = [c for c in contrib_df.columns]
    colors = _pastel_colors(len(cols))
    # stacked area
    x = contrib_df.index
    ys = [contrib_df[c].fillna(0).values for c in cols]
    plt.stackplot(x, ys, labels=cols, colors=colors, alpha=0.9)
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.title("Factor Contributions Over Time (Stacked)")
    plt.xlabel("Date")
    plt.ylabel("Contribution to Return")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_coefficients_over_time(coeffs_df, out_path):
    """
    Line plot of coefficients across time (one figure).
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    cols = coeffs_df.columns
    colors = _pastel_colors(len(cols))
    for i, c in enumerate(cols):
        plt.plot(coeffs_df.index, coeffs_df[c], label=c, linewidth=1.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.title("Rolling Coefficients Over Time")
    plt.xlabel("Date")
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_actual_vs_predicted(dates, actual, predicted, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(dates, actual, label="Actual", linewidth=1.2)
    plt.plot(dates, predicted, label="Predicted", linewidth=1.2, linestyle="--")
    plt.legend()
    plt.title("Actual vs Predicted Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
