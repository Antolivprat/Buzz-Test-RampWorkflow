from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


class Regressor(BaseEstimator):
    def __init__(self):
        None

    def fit(self, X, y):
        data = lgb.Dataset(X, label=y)
        param = {'num_leaves': 31, 'objective':'regression', "verbose": -1}
        param['metric']= 'auc'
        self.model = lgb.train(param, data)

    def predict(self, X):
        return self.model.predict(X)
x²²