import pandas as pd
import numpy as np
import lightgbm as lgb
import itertools
import os
import feature_extract as fe
import training as training

def preprocess(df):
    df = fe.goal_split(df)
    df = fe.country_encoding(df)
    df = fe.category1_encoding(df)
    df = fe.category2_encoding(df)
    
    return df


def add_group_feature(x_train, x_valid, x_test, x_sub):
    others = [x_valid, x_test, x_sub]
    x_train, others = fe.goal_min_group(x_train, others)
    x_train, others = fe.goal_max_group(x_train, others)
    x_train, others = fe.duration_group(x_train, others)
    return x_train, others[0], others[1], others[2]

def mixup(x_train, y_train, alpha=0.2, seed=0):
    x_mixes, y_mixes = list(), list()
    index = np.arange(len(x_train))
    num_sample = 100
    for i in range(num_sample):
        np.random.seed(seed * num_sample + i)
        lam = np.random.beta(alpha, alpha, size=len(x_train))
        np.random.seed(seed * num_sample + i)
        np.random.shuffle(index)
        x_mix = lam.reshape(-1, 1) * x_train.values + (1 - lam).reshape(-1, 1) * x_train.values[index]
        y_mix = lam * y_train.values + (1 - lam) * y_train.values[index]
        x_mixes.append(x_mix)
        y_mixes.append(y_mix)

    x_mixes = np.concatenate(x_mixes)
    y_mixes = np.concatenate(y_mixes)

    return x_mix, y_mix

if __name__ == '__main__':
    train_df, test_df = fe.read_df()
    sub_df = pd.read_csv('data/sample_submit.csv', header=None)
    sub_df.iloc[:, 1] = np.zeros(len(sub_df))

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    train_df['country_original'] = train_df['country_encoding']
    train_df['category1_original'] = train_df['category1_encoding']
    test_df['country_original'] = test_df['country_encoding']
    test_df['category1_original'] = test_df['category1_encoding']
    
    train_df = pd.get_dummies(train_df, drop_first=True, columns=['country_encoding', 'category1_encoding'])
    test_df = pd.get_dummies(test_df, drop_first=True, columns=['country_encoding', 'category1_encoding'])

    train_df = train_df.rename(columns={'country_original': 'country_encoding', 'category1_original': 'category1_encoding'})
    test_df = test_df.rename(columns={'country_original': 'country_encoding', 'category1_original': 'category1_encoding'})

    features = list(train_df.columns)[8:]
    features.append('duration')
    target = 'state'

    # print(features)
    cv = 10
    train_dfs, valid_dfs, test_dfs = training.cv(train_df, cv)
    scores = list()
    params = None
    tune = True
    name = 'mixup_group_features'
    for cv_idx in range(cv):
        #'prepare'
        x_train, y_train = train_dfs[cv_idx][features], train_dfs[cv_idx][target]
        x_valid, y_valid = valid_dfs[cv_idx][features], valid_dfs[cv_idx][target]
        x_test, y_test = test_dfs[cv_idx][features], test_dfs[cv_idx][target]

        x_train, x_valid, x_test, x_sub = add_group_feature(x_train, x_valid, x_test, test_df[features])
        
        x_train, y_train = mixup(x_train, y_train, seed=cv_idx)
        lgb_train, lgb_valid = lgb.Dataset(x_train, y_train), lgb.Dataset(x_valid, y_valid, free_raw_data=False)

        #'train'
        if tune:
            params = training.tuning_mixup(lgb_train, lgb_valid, 100)
            pd.to_pickle(params, 'params/{0}_cv{1}.pkl'.format(name, cv_idx))
        model = training.train_mixup(lgb_train, lgb_valid, params)
        score = training.evaluation_mixup(model, x_test, y_test)
        scores.append(score)
        model.save_model('model/{0}_cv{1}.txt'.format(name, cv_idx), num_iteration=model.best_iteration)

        #'predict'
        pred = model.predict(x_sub)
        pred = np.where(pred < 0.5, 0, 1)
        sub_df.iloc[:, 1] += pred
    sub_df.iloc[:, 1] /= cv
    sub_df.iloc[:, 1] = np.round(sub_df.iloc[:, 1]).astype(np.int8)


    print(scores)
    print(np.mean(scores))

    sub_df.to_csv('result/{0}.csv'.format(name), index=None, header=None)
