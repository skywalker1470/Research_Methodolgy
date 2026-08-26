"""Generates the two figures referenced in final_report.tex:
    outputs/figures/06_mean_degradation.png
    outputs/figures/05_robustness_drop.png

Usage:
    python -m src.plots
(run after src/experiment.py has written outputs/results.csv)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

MODEL_ORDER = ["LR", "MNB", "SVM", "Ens."]
NOISE_ORDER = ["swap", "deletion", "keyboard"]
NOISE_TITLES = {"swap": "Character Swap", "deletion": "Character Deletion", "keyboard": "Keyboard Neighbour"}


def load_results() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "results.csv")
    baseline = df[df["noise_type"] == "none"].copy()
    curves = []
    for noise_type in NOISE_ORDER:
        b = baseline.copy()
        b["noise_type"] = noise_type
        b["intensity"] = 0.0
        curves.append(b)
        curves.append(df[df["noise_type"] == noise_type])
    return pd.concat(curves, ignore_index=True)


def plot_mean_degradation(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, noise_type in zip(axes, NOISE_ORDER):
        sub = df[df["noise_type"] == noise_type]
        stats = sub.groupby(["intensity", "model"])["accuracy"].agg(["mean", "std"]).reset_index()
        for model in MODEL_ORDER:
            m = stats[stats["model"] == model].sort_values("intensity")
            ax.plot(m["intensity"] * 100, m["mean"], marker="o", label=model)
            ax.fill_between(m["intensity"] * 100, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.15)
        ax.set_title(NOISE_TITLES[noise_type])
        ax.set_xlabel("Noise intensity (%)")
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_mean_degradation.png", dpi=300)
    plt.close(fig)


def plot_robustness_drop(df: pd.DataFrame):
    sub = df[df["intensity"] == 0.25]
    stats = sub.groupby(["noise_type", "model"])["delta_acc"].agg(["mean", "std"]).reset_index()

    x = np.arange(len(NOISE_ORDER))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, model in enumerate(MODEL_ORDER):
        m = stats[stats["model"] == model].set_index("noise_type").reindex(NOISE_ORDER)
        ax.bar(x + (i - 1.5) * width, m["mean"], width, yerr=m["std"], capsize=3, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_TITLES[n] for n in NOISE_ORDER])
    ax.set_ylabel(r"Mean $\Delta$Acc (0% $\to$ 25%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_robustness_drop.png", dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    plot_mean_degradation(df)
    plot_robustness_drop(pd.read_csv(OUT_DIR / "results.csv"))
    print("Figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
