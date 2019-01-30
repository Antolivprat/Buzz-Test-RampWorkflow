from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        #X_df_new = compute_derivative(X_df_new, 'NCD', 0, 2)
        #X_df_new = compute_ratio(X_df_new, 'NCD', 'NAD', [0,1,2])
        return X_df_new.drop(columns = 'site')


def compute_derivative(data, feature, int1, int2):
    """
    For a given dataframe, compute the normalized difference over the
    defined period of time time (int1, int2) of a feature (feature) 

    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the difference
    int1 : int
        1rst column to take
    int2 : int
        last column to take
    
    Return
    ----------
    The same data frame with int2 - int1 - 1 colums more

    """
    features = [str(feature) + '_' + str(i) for i in range(int1, int2 + 1)]
    for i in range(len(features) - 1):
        data[str(feature) + '_Diff_' + str(i)] = data[features[i+1]] - data[features[i]]
    return data

def compute_ratio(data, feature1, feature2, ints):
    """
        For a given dataframe, compute the normalized difference over the
    defined period of time time (int1, int2) of a feature (feature) 

    Parameters
    ----------
    data : dataframe
    feature1 : str
        feature in the dataframe which will be in the numerator
    feature2 : str
        feature in the dataframe which will be in the denominator
    ints : list of int
        for which number you want to compute the ratio
    
    Return
    ----------
    The dataframe with len(ints) more columns    
    
    """
    new_feature = str(feature1) + "/" + str(feature2)
    for i in ints:
        temp = data[feature2 + '_' + str(i)] + 1 #to avoid 0
        data[new_feature + '_' + str(i)] = data[feature1 + '_' + str(i)] / temp
    
    return data
