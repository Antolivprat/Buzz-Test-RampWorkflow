from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class Regressor(BaseEstimator):
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), RandomForestRegressor())

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
