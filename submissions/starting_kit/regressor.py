from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Regressor(BaseEstimator):
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LinearRegression())

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)