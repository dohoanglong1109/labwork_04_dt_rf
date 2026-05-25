import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def evaluate_baseline(prefix, dataset_name):
    try:
        X_train = pd.read_csv(f"data/processed/X_train_{prefix}.csv")
        X_test = pd.read_csv(f"data/processed/X_test_{prefix}.csv")
        y_train = pd.read_csv(f"data/processed/y_train_{prefix}.csv").squeeze()
        y_test = pd.read_csv(f"data/processed/y_test_{prefix}.csv").squeeze()
    except FileNotFoundError:
        print(f"[ERROR] Processed data for {dataset_name} not found.")
        return

    # Train unpruned Decision Tree
    dt = DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=42)
    dt.fit(X_train, y_train)

    acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"[RESULT] {dataset_name} Baseline DT Accuracy: {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    print("\n--- 2. BASELINE DECISION TREE EVALUATION ---")
    evaluate_baseline("bc", "Breast Cancer")
    evaluate_baseline("wine", "Wine Quality")
