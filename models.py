import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter


class CustomRandomForest:
    """A custom implementation of Random Forest using Bagging and Majority Voting."""

    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_arr, y_arr = np.array(X), np.array(y)
        n_samples = X_arr.shape[0]

        for i in range(self.n_estimators):
            # Bootstrap sampling with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)

            tree = DecisionTreeClassifier(
                criterion="gini", max_depth=None, random_state=self.random_state + i
            )
            tree.fit(X_arr[indices], y_arr[indices])
            self.trees.append(tree)

    def predict(self, X):
        X_arr = np.array(X)
        # Gather predictions from all trees
        preds = np.array([tree.predict(X_arr) for tree in self.trees])

        # Majority voting
        return np.array(
            [Counter(preds[:, i]).most_common(1)[0][0] for i in range(X_arr.shape[0])]
        )
