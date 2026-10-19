"""OOP class for decision tree regression."""

from sklearn.tree import DecisionTreeRegressor


class DecisionTreeModel:
    """Train and predict with a decision tree regressor."""

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
