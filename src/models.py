"""Leakage-safe preprocessing, model factories and modest hyperparameter search spaces."""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

SEED = 42
MODELS = ['Decision Tree', 'Bagging', 'Random Forest', 'XGBoost', 'LightGBM']


def make_preprocessor(x):
    numeric = x.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    transformer = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), numeric),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ], remainder='drop', verbose_feature_names_out=True)
    transformer.set_output(transform='pandas')
    return transformer


def make_estimator(name, task, params=None):
    params = dict(params or {})
    is_cls = task == 'risk'
    if task not in ('grade', 'risk'):
        raise ValueError('task is grade or risk')
    if name == 'Decision Tree':
        cls = DecisionTreeClassifier if is_cls else DecisionTreeRegressor
        base = dict(max_depth=5, min_samples_leaf=3, random_state=SEED)
    elif name == 'Bagging':
        cls = BaggingClassifier if is_cls else BaggingRegressor
        tree = (DecisionTreeClassifier if is_cls else DecisionTreeRegressor)(max_depth=7, min_samples_leaf=2,
                                                                             random_state=SEED)
        base = dict(estimator=tree, n_estimators=75, bootstrap=True, n_jobs=1, random_state=SEED)
    elif name == 'Random Forest':
        cls = RandomForestClassifier if is_cls else RandomForestRegressor
        base = dict(n_estimators=120, max_depth=8, min_samples_leaf=2, bootstrap=True, n_jobs=1,
                    random_state=SEED)
    elif name == 'XGBoost':
        cls = XGBClassifier if is_cls else XGBRegressor
        base = dict(n_estimators=140, max_depth=3, learning_rate=.065, subsample=.85,
                    colsample_bytree=.9, reg_alpha=.2, reg_lambda=3.0, n_jobs=1,
                    tree_method='hist', random_state=SEED)
        base['objective'] = 'binary:logistic' if is_cls else 'reg:squarederror'
        if is_cls:
            base['eval_metric'] = 'logloss'
    elif name == 'LightGBM':
        cls = LGBMClassifier if is_cls else LGBMRegressor
        base = dict(n_estimators=130, max_depth=4, num_leaves=12, learning_rate=.06,
                    min_child_samples=12, reg_alpha=.2, reg_lambda=3.0, subsample=.9,
                    colsample_bytree=.9, verbose=-1, n_jobs=1, random_state=SEED)
    else:
        raise ValueError(f'Unknown model: {name}')
    base.update(params)
    return cls(**base)


def make_pipeline(name, task, x, params=None):
    return Pipeline([('prep', make_preprocessor(x)), ('model', make_estimator(name, task, params))])


def configurations(name, fast=False):
    """Simple candidate choices compared using training-only CV; never optimized on held-out test."""
    if fast:
        return [{}]
    options = {
        'Decision Tree': [{}, {'max_depth': 2, 'min_samples_leaf': 5, 'ccp_alpha': .01},
                          {'max_depth': 8, 'min_samples_leaf': 5, 'ccp_alpha': .02}],
        'Bagging': [{}, {'n_estimators': 40, 'max_samples': .8},
                    {'n_estimators': 100, 'max_features': .9}],
        'Random Forest': [{}, {'n_estimators': 80, 'max_depth': 4, 'min_samples_leaf': 4},
                          {'n_estimators': 160, 'max_depth': None, 'min_samples_leaf': 3}],
        'XGBoost': [{}, {'n_estimators': 70, 'max_depth': 2, 'learning_rate': .08,
                         'reg_alpha': 1., 'reg_lambda': 8.},
                    {'n_estimators': 170, 'max_depth': 4, 'learning_rate': .04,
                     'reg_alpha': 0., 'reg_lambda': 1.}],
        'LightGBM': [{}, {'n_estimators': 90, 'num_leaves': 7, 'max_depth': 3,
                          'min_child_samples': 20, 'reg_lambda': 8.},
                     {'n_estimators': 160, 'num_leaves': 20, 'max_depth': 5,
                      'min_child_samples': 9, 'reg_lambda': 1.}],
    }
    return options[name]
