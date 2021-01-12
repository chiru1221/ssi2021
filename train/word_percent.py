import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import feature_extract as fe
import training as training

def preprocess(df):
    df = fe.goal_split(df)
    df = fe.country_encoding(df)
    df = fe.category1_encoding(df)
    df = fe.category2_encoding(df)
    
    return df


if __name__ == '__main__':
    train_df, test_df = fe.read_df()
    sub_df = pd.read_csv('data/sample_submit.csv', header=None)
    sub_df.iloc[:, 1] = np.zeros(len(sub_df))

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    features = [
        'duration', 'goal_min', 'goal_max', 
        'country_encoding', 'category1_encoding', 'category2_encoding',
        'html_content'
    ]
    target = 'state'


    cv = 10
    train_dfs, valid_dfs, test_dfs = training.cv(train_df, cv)
    scores = list()
    params = None
    tune = True
    name = 'word_percent'
    for cv_idx in range(cv):
        'prepare'
        x_train, y_train = train_dfs[cv_idx][features], train_dfs[cv_idx][target]
        x_valid, y_valid = valid_dfs[cv_idx][features], valid_dfs[cv_idx][target]
        x_test, y_test = test_dfs[cv_idx][features], test_dfs[cv_idx][target]
        
        x_train, others = fe.text_to_word_percent(x_train, [x_valid, x_test, test_df[features]])
        x_valid, x_test, x_sub = others[0], others[1], others[2]
        
        lgb_train, lgb_valid = lgb.Dataset(x_train, y_train, categorical_feature=[3, 4, 5], free_raw_data=False), lgb.Dataset(x_valid, y_valid, categorical_feature=[3, 4, 5], free_raw_data=False)

        'train'
        if tune:
            params = training.tuning(lgb_train, lgb_valid, 100)
            pd.to_pickle(params, 'params/{0}_cv{1}.pkl'.format(name, cv_idx))
        model = training.train(lgb_train, lgb_valid, params)
        score = training.evaluation(model, x_test, y_test)
        scores.append(score)
        model.save_model('model/{0}_cv{1}.txt'.format(name, cv_idx), num_iteration=model.best_iteration)

        'predict'
        pred = model.predict(x_sub)
        pred = np.round(pred)
        sub_df.iloc[:, 1] += pred
    sub_df.iloc[:, 1] /= cv
    sub_df.iloc[:, 1] = np.round(sub_df.iloc[:, 1]).astype(np.int8)


    print(scores)
    print(np.mean(scores))
    # 0l7820360070930705
    # 

    sub_df.to_csv('result/{0}.csv'.format(name), index=None, header=None)
