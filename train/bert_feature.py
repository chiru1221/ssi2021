import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import feature_extract as fe
import training as training
os.environ['PYTHONHASHSEED'] = '0'

def preprocess(df):
    # df = fe.goal_split(df)
    # df = fe.country_encoding(df)
    # df = fe.category1_encoding(df)
    # df = fe.category2_encoding(df)

    df = fe.bert_feature(df)
    return df


if __name__ == '__main__':
    train_df, test_df = fe.read_df()
    sub_df = pd.read_csv('data/sample_submit.csv', header=None)
    sub_df.iloc[:, 1] = np.zeros(len(sub_df))

    # train_df = preprocess(train_df)
    train_feature = fe.bert_feature(train_df)
    train_feature['state'] = train_df.state
    test_feature = fe.bert_feature(test_df)
    # test_df = preprocess(test_df)
    # features = [
    #     'html_content'
    # ]
    features = list(test_feature.columns)
    target = 'state'

    print(train_feature.shape, test_feature.shape)
    
    cv = 10
    train_dfs, valid_dfs, test_dfs = training.cv(train_feature, cv)
    scores = list()
    params = None
    tune = True
    name = 'bert_feature'
    for cv_idx in range(cv):
        'prepare'
        x_train, y_train = train_dfs[cv_idx][features], train_dfs[cv_idx][target]
        x_valid, y_valid = valid_dfs[cv_idx][features], valid_dfs[cv_idx][target]
        x_test, y_test = test_dfs[cv_idx][features], test_dfs[cv_idx][target]
        
        lgb_train, lgb_valid = lgb.Dataset(x_train, y_train, free_raw_data=False), lgb.Dataset(x_valid, y_valid, free_raw_data=False)

        'train'
        if tune:
            params = training.tuning(lgb_train, lgb_valid, 100)
            pd.to_pickle(params, 'params/{0}_cv{1}.pkl'.format(name, cv_idx))
        model = training.train(lgb_train, lgb_valid, params)
        score = training.evaluation(model, x_test, y_test)
        scores.append(score)
        model.save_model('model/{0}_cv{1}.txt'.format(name, cv_idx), num_iteration=model.best_iteration)

        'predict'
        pred = model.predict(test_feature[features])
        pred = np.round(pred)
        sub_df.iloc[:, 1] += pred
    sub_df.iloc[:, 1] /= cv
    sub_df.iloc[:, 1] = np.round(sub_df.iloc[:, 1]).astype(np.int8)


    print(scores)
    print(np.mean(scores))
    # 

    sub_df.to_csv('result/{0}.csv'.format(name), index=None, header=None)
    