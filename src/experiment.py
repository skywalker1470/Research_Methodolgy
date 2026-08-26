"""Main experiment: classifier robustness to character-level noise on SST-2.

Reproduces the study in final_report.tex:
  - Train on all non-neutral phrases (subtrees) from the SST train split.
  - Evaluate on the non-neutral root sentences of the SST test split.
  - TF-IDF (unigrams+bigrams, sublinear TF, max_df=0.95), fit once on clean
    train text and reused unchanged for every noisy evaluation.
  - Four models: Logistic Regression, Multinomial Naive Bayes, Linear SVM,
    and a hard-voting ensemble of the three.
  - Three noise types (swap, deletion, keyboard-neighbour) x five intensities
    (5-25%), each evaluated across five random seeds, without retraining.

Usage:
    python -m src.experiment
Outputs:
    outputs/results.csv                 raw per-seed/condition/model rows
    outputs/tables/table1_robustness.csv
    outputs/tables/table2_swap.csv
    outputs/tables/table3_deletion.csv
    outputs/tables/table4_keyboard.csv
    outputs/stat_tests.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.noise import INTENSITIES, NOISE_FUNCS, apply_noise
from src.sst_data import load_split, phrase_level_dataset, sentence_level_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "sentiment-treebank-master" / "binary"
OUT_DIR = ROOT / "outputs"
SEEDS = [42, 7, 123, 2024, 99]
MODEL_NAMES = ["LR", "MNB", "SVM", "Ens."]


def build_models(seed: int):
    return {
        "LR": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed),
        "MNB": MultinomialNB(alpha=0.1),
        "SVM": LinearSVC(C=1.0, random_state=seed),
    }


def majority_vote(preds: dict) -> np.ndarray:
    stacked = np.vstack([preds["LR"], preds["MNB"], preds["SVM"]])
    return (stacked.sum(axis=0) >= 2).astype(int)


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def run() -> pd.DataFrame:
    print("Loading SST-2 splits...")
    train_trees = load_split(DATA_DIR, "train")
    test_trees = load_split(DATA_DIR, "test")

    X_train_text, y_train = phrase_level_dataset(train_trees)
    X_test_text, y_test = sentence_level_dataset(test_trees)
    y_test = np.array(y_test)
    print(f"  train phrases: {len(X_train_text)}, test sentences: {len(X_test_text)}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.95)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test_clean = vectorizer.transform(X_test_text)

    rows = []
    for seed in SEEDS:
        print(f"seed={seed}")
        models = build_models(seed)
        for model in models.values():
            model.fit(X_train, y_train)

        # 0% noise baseline
        preds = {name: m.predict(X_test_clean) for name, m in models.items()}
        preds["Ens."] = majority_vote(preds)
        for name in MODEL_NAMES:
            metrics = evaluate(y_test, preds[name])
            rows.append({"seed": seed, "noise_type": "none", "intensity": 0.0,
                         "model": name, **metrics})

        for noise_type in NOISE_FUNCS:
            for p in INTENSITIES:
                noisy_text = apply_noise(X_test_text, noise_type, p, seed)
                X_noisy = vectorizer.transform(noisy_text)
                preds = {name: m.predict(X_noisy) for name, m in models.items()}
                preds["Ens."] = majority_vote(preds)
                for name in MODEL_NAMES:
                    metrics = evaluate(y_test, preds[name])
                    rows.append({"seed": seed, "noise_type": noise_type, "intensity": p,
                                 "model": name, **metrics})

    return pd.DataFrame(rows)


def add_delta_acc(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df[df["noise_type"] == "none"]
        .set_index(["seed", "model"])["accuracy"]
        .rename("baseline_accuracy")
    )
    df = df.join(baseline, on=["seed", "model"])
    df["delta_acc"] = df["baseline_accuracy"] - df["accuracy"]
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["noise_type", "intensity", "model"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
        delta_acc_mean=("delta_acc", "mean"), delta_acc_sd=("delta_acc", "std"),
    ).reset_index()
    return grouped


def make_noise_table(agg: pd.DataFrame, noise_type: str) -> pd.DataFrame:
    sub = agg[agg["noise_type"].isin(["none", noise_type])]
    sub = sub.sort_values("intensity")
    return sub.pivot(index="intensity", columns="model",
                      values=["accuracy_mean", "accuracy_sd", "f1_mean", "f1_sd"])


def make_robustness_table(agg: pd.DataFrame) -> pd.DataFrame:
    sub = agg[(agg["intensity"] == 0.25)]
    return sub.pivot(index="noise_type", columns="model",
                      values=["delta_acc_mean", "delta_acc_sd"])


def paired_ttest(df: pd.DataFrame, model_a: str, model_b: str, metric: str = "accuracy"):
    """Paired t-test over the 15 non-zero noise conditions (mean across seeds
    per condition), comparing model_a vs model_b."""
    noisy = df[df["noise_type"] != "none"]
    per_condition = (
        noisy.groupby(["noise_type", "intensity", "model"])[metric]
        .mean()
        .unstack("model")
    )
    a, b = per_condition[model_a], per_condition[model_b]
    t_stat, p_value = stats.ttest_rel(a, b)
    return {"model_a": model_a, "model_b": model_b, "metric": metric,
            "n": len(a), "mean_diff": (a - b).mean(), "t_stat": t_stat, "p_value": p_value}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)

    df = run()
    df = add_delta_acc(df)
    df.to_csv(OUT_DIR / "results.csv", index=False)

    agg = aggregate(df)
    make_robustness_table(agg).to_csv(OUT_DIR / "tables" / "table1_robustness.csv")
    make_noise_table(agg, "swap").to_csv(OUT_DIR / "tables" / "table2_swap.csv")
    make_noise_table(agg, "deletion").to_csv(OUT_DIR / "tables" / "table3_deletion.csv")
    make_noise_table(agg, "keyboard").to_csv(OUT_DIR / "tables" / "table4_keyboard.csv")

    comparisons = []
    for a, b in [("LR", "Ens."), ("MNB", "Ens."), ("SVM", "Ens."), ("LR", "MNB"), ("LR", "SVM")]:
        comparisons.append(paired_ttest(df, a, b, metric="accuracy"))
    pd.DataFrame(comparisons).to_csv(OUT_DIR / "stat_tests.csv", index=False)

    print("\nTable 1 (mean delta-acc @ 25% noise):")
    print(make_robustness_table(agg).round(3))
    print("\nDone. Outputs written to", OUT_DIR)


if __name__ == "__main__":
    main()
