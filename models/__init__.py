"""Custom OOP wrappers for Practical Lab 2 machine-learning experiments."""

from .univariate_polynomial import UnivariatePolynomialModel
from .multivariate_polynomial import MultivariatePolynomialModel
from .decision_tree import DecisionTreeModel
from .knn import KNNModel
from .logistic_classification import LogisticClassificationModel

__all__ = [
    "UnivariatePolynomialModel",
    "MultivariatePolynomialModel",
    "DecisionTreeModel",
    "KNNModel",
    "LogisticClassificationModel",
]
