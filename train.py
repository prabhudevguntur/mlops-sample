# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn
import joblib
assert sklearn.__version__ >= "0.20"
# Common imports
import numpy as np
# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
import os
import tarfile
import urllib.request
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def get_data(housing_url, housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_data(housing_url,housing_path):
    get_data(housing_url,housing_path)
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_data(housing):
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    data = {"train": {"X": strat_train_set.drop("median_house_value", axis=1),
                      "y": strat_train_set["median_house_value"].copy()},
            "test": {"X": strat_test_set.drop("median_house_value", axis=1),
                     "y": strat_test_set["median_house_value"].copy()}}
    return data


def feature_eng(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            return self  # nothing else to do

        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household,
                             bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()), ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs), ])
    #housing_prepared = full_pipeline.fit_transform(housing)
    #return housing_prepared
    return full_pipeline


def train_model(housing_prepared, housing_labels):
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [3, 10, 100], 'max_features': [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [True], 'n_estimators': [10, 100], 'max_features': [2, 3, 4]}, ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)
    final_model = grid_search.best_estimator_
    return final_model


def get_model_metrics(forest_reg, X, y):
    preds = forest_reg.predict(X)
    mse = mean_squared_error(preds, y)
    metrics = {"mse": mse}
    return metrics


def main():
    np.random.seed(42)
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    # Load Data
    housing = load_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)  # ---->main

    # Split Data into Training and Validation Sets
    data = split_data(housing)

    # Train Model on Training Set
    full_pipeline = feature_eng(data["train"]["X"])
    housing_prepared = full_pipeline.fit_transform(data["train"]["X"])
    reg = train_model(housing_prepared, data["train"]["y"])

    # Validate Model on Validation Set
    housing_prepared_test = full_pipeline.transform(data["test"]["X"])
    test_y = data["test"]["y"]
    metrics = get_model_metrics(reg, housing_prepared_test, test_y)
    print(metrics)

    # Save Model
    model_name = "rf_tuned_model.pkl"

    joblib.dump(value=reg, filename=model_name)
    joblib.dump(value=full_pipeline, filename= "full_pipeline.pkl")

if __name__ == '__main__':
    main()
