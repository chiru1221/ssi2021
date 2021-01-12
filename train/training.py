import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler
import feature_extract as fe


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def mixup_logloss(preds, data):
    y_true = data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - y_true
    hess = preds * (1.0 - preds)
    return grad, hess

def mixup_f1_score(y_hat, data):
    y_true = data.get_label()
    y_true = np.where(y_true < 0.5, 0, 1).astype(np.int32)
    y_hat = np.where(y_hat < 0.5, 0, 1).astype(np.int32) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


def train(lgb_train, lgb_valid, params=None):
    if params is None:
        params = dict()
    params['objective'] = 'binary'
    params['random_state'] = '0'
    params['metric'] = 'None'
    params['verbose'] = -1
    params['feature_pre_filter'] = False

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        feval=lgb_f1_score,
        verbose_eval=100,
        num_boost_round=10000,
        early_stopping_rounds=500
    )
    return model

def train_mixup(lgb_train, lgb_valid, params=None):
    if params is None:
        params = dict()
    params['objective'] = mixup_logloss
    params['random_state'] = '0'
    params['metric'] = 'None'
    params['verbose'] = -1
    params['feature_pre_filter'] = False

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        fobj=mixup_logloss,
        feval=mixup_f1_score,
        verbose_eval=100,
        num_boost_round=10000,
        early_stopping_rounds=500
    )
    return model

def evaluation_mixup(model, x, y):
    y_pred = np.where(model.predict(x) < 0.5, 0, 1).astype(np.int32)
    return f1_score(y, y_pred)

def evaluation(model, x, y):
    y_pred = np.round(model.predict(x))
    return f1_score(y, y_pred)

def tuning(lgb_train, lgb_valid, n_trials=100):
    def create_model(trial):
        num_leaves = trial.suggest_int('num_leaves', 7, 256)
        # 500 or 1000, -> (450, 550) or (950, 1050)
        n_estimators = trial.suggest_int('n_estimators', 950, 1050)
        max_depth = trial.suggest_int('max_depth', 3, 8)
        # 0.0001, 0.1
        learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.1)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 50, 90)
        bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.1, 1.0)
        feature_fraction = trial.suggest_uniform('feature_fraction', 0.1, 1.0)
        model = lgb.train(
            {
                'objective': 'binary',
                'metric': 'None',
                'num_leaves': num_leaves,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'min_data_in_leaf': min_data_in_leaf,
                'bagging_fraction': bagging_fraction,
                'feature_fraction': feature_fraction,
                'feature_pre_filter': False,
                'verbose': -1,
                'random_state': 0
            }, 
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            feval=lgb_f1_score,
            verbose_eval=-1,
            num_boost_round=10000,
            early_stopping_rounds=1000
        )
        return model

    def objective(trial):
        model = create_model(trial)
        score = evaluation(model, lgb_valid.get_data(), lgb_valid.get_label())
        return score
    optuna.logging.disable_default_handler()
    sampler = TPESampler(seed=0)
    optim = optuna.create_study(direction="maximize", sampler=sampler)
    optim.optimize(objective, n_trials=n_trials)
    params = optim.best_params
    return params

def tuning_mixup(lgb_train, lgb_valid, n_trials=10):
    def create_model(trial):
        num_leaves = trial.suggest_int('num_leaves', 63, 127)
        # 500 or 1000, -> (450, 550) or (950, 1050)
        n_estimators = trial.suggest_int('n_estimators', 450, 550)
        max_depth = trial.suggest_int('max_depth', 6, 9)
        # 0.0001, 0.1
        learning_rate = trial.suggest_uniform('learning_rate', 0.001, 0.1)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 50, 90)
        bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.1, 1.0)
        feature_fraction = trial.suggest_uniform('feature_fraction', 0.1, 1.0)
        model = lgb.train(
            {
                'objective': mixup_logloss,
                'metric': 'None',
                'num_leaves': num_leaves,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'min_data_in_leaf': min_data_in_leaf,
                'bagging_fraction': bagging_fraction,
                'feature_fraction': feature_fraction,
                'feature_pre_filter': False,
                'verbose': -1,
                'random_state': 0
            }, 
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            fobj=mixup_logloss,
            feval=mixup_f1_score,
            verbose_eval=-1,
            num_boost_round=10000,
            early_stopping_rounds=500
        )
        return model

    def objective(trial):
        model = create_model(trial)
        score = evaluation_mixup(model, lgb_valid.get_data(), lgb_valid.get_label())
        return score
    optuna.logging.disable_default_handler()
    sampler = TPESampler(seed=0)
    optim = optuna.create_study(direction="maximize", sampler=sampler)
    optim.optimize(objective, n_trials=n_trials)
    params = optim.best_params
    return params


def cv(train_df, n_splits, val_size=0.1, seed=0):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    all_df = train_df.copy()

    train_dfs, valid_dfs, test_dfs = list(), list(), list()
    for train_idx, test_idx in kf.split(np.zeros(len(all_df)), all_df.state.values):
        np.random.seed(seed)
        valid_idx = np.random.choice(train_idx, size=int(len(train_idx)*val_size), replace=False)
        train_idx = np.setdiff1d(train_idx, valid_idx)
        train_df, valid_df, test_df = all_df.iloc[train_idx, :], all_df.iloc[valid_idx, :], all_df.iloc[test_idx, :]
        train_dfs.append(train_df)
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)
    
    return train_dfs, valid_dfs, test_dfs
