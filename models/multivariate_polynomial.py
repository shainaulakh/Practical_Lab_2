"""OOP class for polynomial regression using all ten input features."""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class MultivariatePolynomialModel:
    """Fit a polynomial linear regression model on multiple features."""

    def __init__(self, degree):
        self.degree = degree
        self.model = Pipeline([
            ("polynomial", PolynomialFeatures(
                degree=degree, include_bias=False
            )),
            ("regression", LinearRegression()),
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
