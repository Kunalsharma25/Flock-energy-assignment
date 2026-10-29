"""Plotting styling and helper utilities for REFIT NILM project."""

import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """Set publication-quality plotting style."""
    sns.set_theme(style="whitegrid", palette="tab10")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
