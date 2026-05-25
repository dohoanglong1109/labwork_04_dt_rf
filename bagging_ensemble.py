import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_flowchart():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    # Simple architecture diagram
    ax.text(
        0.5,
        0.8,
        "Original Training Subset (N)",
        ha="center",
        bbox=dict(boxstyle="round", fc="#e6f2ff"),
    )
    ax.text(
        0.5,
        0.5,
        "Bootstrap Sampling (K=100)\n~63.2% Unique, ~36.8% OOB",
        ha="center",
        style="italic",
    )
    ax.text(
        0.2, 0.2, "Bootstrap 1", ha="center", bbox=dict(boxstyle="round", fc="#f8f9fa")
    )
    ax.text(0.5, 0.2, ". . .", ha="center", fontweight="bold")
    ax.text(
        0.8,
        0.2,
        "Bootstrap 100",
        ha="center",
        bbox=dict(boxstyle="round", fc="#f8f9fa"),
    )

    plt.savefig("figure2.png", dpi=300, bbox_inches="tight")
    print("[INFO] Bagging flowchart exported to 'figure2.png'")


def calc_empirical_oob(prefix, dataset_name):
    try:
        X_train = pd.read_csv(f"data/processed/X_train_{prefix}.csv")
    except FileNotFoundError:
        return

    N = len(X_train)
    oob_rates = []
    np.random.seed(42)

    for _ in range(100):
        indices = np.random.choice(N, size=N, replace=True)
        unique_indices = set(indices)
        oob_size = N - len(unique_indices)
        oob_rates.append(oob_size / N * 100)

    print(
        f"[STAT] {dataset_name} -> Avg Out-of-Bag (OOB) Rate: {np.mean(oob_rates):.2f}%"
    )


if __name__ == "__main__":
    print("\n--- 3. BAGGING MECHANISM ANALYSIS ---")
    # generate_flowchart()
    calc_empirical_oob("bc", "Breast Cancer")
    calc_empirical_oob("wine", "Wine Quality")
