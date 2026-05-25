import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from models import CustomRandomForest


def run_evaluation(prefix, dataset_name):
    try:
        X_train = pd.read_csv(f"data/processed/X_train_{prefix}.csv")
        X_test = pd.read_csv(f"data/processed/X_test_{prefix}.csv")
        y_train = pd.read_csv(f"data/processed/y_train_{prefix}.csv").squeeze()
        y_test = pd.read_csv(f"data/processed/y_test_{prefix}.csv").squeeze()
    except FileNotFoundError:
        print(f"[ERROR] Processed data for {dataset_name} not found.")
        return

    # 1. Baseline Decision Tree
    dt = DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=42)
    dt.fit(X_train, y_train)
    dt_train_err = (1 - accuracy_score(y_train, dt.predict(X_train))) * 100
    dt_test_err = (1 - accuracy_score(y_test, dt.predict(X_test))) * 100

    # 2. Custom Random Forest
    rf = CustomRandomForest(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_test_err = (1 - accuracy_score(y_test, rf.predict(X_test))) * 100

    print(f"\n[{dataset_name.upper()} RESULTS]")
    print(f"DT Train Error : {dt_train_err:.2f}%")
    print(f"DT Test Error  : {dt_test_err:.2f}%")
    print(f"RF Test Error  : {rf_test_err:.2f}%")
    print(f"Error Reduction: {dt_test_err - rf_test_err:.2f}%")


if __name__ == "__main__":
    print("\n--- 4. FINAL MODEL EVALUATION (TABLE 3) ---")
    run_evaluation("bc", "Breast Cancer")
    run_evaluation("wine", "Wine Quality")
