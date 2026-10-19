"""OOP class for the supplementary binary classification experiment."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticClassificationModel:
    """Train and predict a binary Logistic Regression classifier."""

    def __init__(self, C):
        self.C = C
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(
                C=C,
                max_iter=3000,
                random_state=42,
            )),
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
