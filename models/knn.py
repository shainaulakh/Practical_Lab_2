"""OOP class for scaled k-nearest-neighbors regression."""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KNNModel:
    """Train and predict with scaled kNN regression."""

    def __init__(self, k):
        self.k = k
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=k, weights="uniform"
            )),
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
