import os
import pandas as pd
from sklearn.model_selection import train_test_split


def process_and_save(filepath, target_col, prefix, sep=",", drop_cols=None):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return

    df = pd.read_csv(filepath, sep=sep)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratify if categorical target
    stratify_param = y if len(y.unique()) < 15 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    # Export to processed directory
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    X_train.to_csv(f"{out_dir}/X_train_{prefix}.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test_{prefix}.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train_{prefix}.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test_{prefix}.csv", index=False)

    print(
        f"[INFO] {prefix.upper()} processed -> Train: {X_train.shape}, Test: {X_test.shape}"
    )


if __name__ == "__main__":
    print("--- 1. DATA PREPROCESSING ---")
    process_and_save(
        filepath="data/wisconsin_breast_cancer.csv",
        target_col="diagnosis",
        prefix="bc",
        drop_cols=["id", "Unnamed: 32"],
    )
    process_and_save(
        filepath="data/winequality-red.csv",
        target_col="quality",
        prefix="wine",
        sep=";",
    )
