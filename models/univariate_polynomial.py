"""OOP class for BMI-only polynomial regression (degrees 0 to 5)."""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class UnivariatePolynomialModel:
    """Regress progression on polynomial terms of one feature (BMI)."""

    def __init__(self, degree):
        self.degree = degree
        self.model = Pipeline([
            ("polynomial", PolynomialFeatures(
                degree=degree, include_bias=True
            )),
            ("regression", LinearRegression(fit_intercept=False)),
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
