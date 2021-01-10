import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'preprocess'))
import feature_extract as fe
import training as training

if __name__ == '__main__':
    train_df, test_df = fe.read_df()
    sub_df = pd.read_csv('data/sample_submit.csv', header=None)
    sub_df.iloc[:, 1] = np.zeros(len(sub_df))

    train_df = training.preprocess(train_df)
    test_df = training.preprocess(test_df)

    cv = 10
    train_dfs, valid_dfs, test_dfs = training.cv(train_df, cv)
    scores = list()
    for cv_idx in range(cv):
        'prepare'
        x_train, y_train = train_dfs[cv_idx].iloc[:, 8:], train_dfs[cv_idx].iloc[:, 7]
        x_valid, y_valid = valid_dfs[cv_idx].iloc[:, 8:], valid_dfs[cv_idx].iloc[:, 7]
        x_test, y_test = test_dfs[cv_idx].iloc[:, 8:], test_dfs[cv_idx].iloc[:, 7]
        lgb_train, lgb_valid = lgb.Dataset(x_train, y_train), lgb.Dataset(x_valid, y_valid, free_raw_data=False)

        'train'
        params = training.tuning(lgb_train, lgb_valid)
        model = training.train(lgb_train, lgb_valid, params)
        score = training.evaluation(model, x_test, y_test)
        scores.append(score)
        model.save_model('model/baseline_cv{0}.txt'.format(cv_idx), num_iteration=model.best_iteration)

        'predict'
        pred = model.predict(test_df.iloc[:, 7:])
        pred = np.round(pred)
        sub_df.iloc[:, 1] += pred
    sub_df.iloc[:, 1] /= cv
    sub_df.iloc[:, 1] = np.round(sub_df.iloc[:, 1]).astype(np.int8)


    print(scores)
    print(test_df.columns)

    sub_df.to_csv('result/baseline.csv', index=None, header=None)
