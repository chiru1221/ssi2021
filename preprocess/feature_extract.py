import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import itertools

def read_df():
    dtype = {
        'id': np.int8,
        'goal': str,
        'country': str,
        'duration': np.int32,
        'category1': str,
        'category2': str,
        'html_content': str,
        'state': np.int8
    }
    train_df = pd.read_csv('data/train.csv', dtype=dtype)
    test_df = pd.read_csv('data/test.csv', dtype=dtype)

    
    # np.save('data/country.npy', train_df.country.unique())
    # np.save('data/category1.npy', train_df.category1.unique())
    # np.save('data/category2.npy', np.union1d(train_df.category2.unique(), test_df.category2.unique()))


    return train_df, test_df

def goal_split(df):
    def split_min_max(x):
        if len(x) != 2:
            return [-1, x[0].replace('+', '')]
        return x
    df.goal = df.goal.str.split('-').apply(split_min_max)
    df['goal_min'] = df.goal.apply(lambda x: x[0]).astype(np.int64)
    df['goal_max'] = df.goal.apply(lambda x: x[1]).astype(np.int64)
    return df

def country_encoding(df):
    le = LabelEncoder()
    country = np.load('data/country.npy', allow_pickle=True)
    # countries = [
    #     'CH', 'NL', 'US', 'GB', 'CA', 'ES', 'FR', 'DE', 'AU', 'IT', 'MX', 
    #     'NZ', 'NO', 'DK', 'HK', 'SG', 'BE', 'SE', 'IE', 'JP', 'AT', 'LU'
    # ]
    le.fit(country)
    df['country_encoding'] = le.transform(df.country.values)
    return df

def category1_encoding(df):
    le = LabelEncoder()
    category = np.load('data/category1.npy', allow_pickle=True)
    # countries = [
    #     'publishing', 'fashion', 'food', 'technology', 'photography', 
    #     'art', 'film & video', 'dance', 'design', 'games', 'journalism', 
    #     'music', 'comics', 'theater', 'crafts'
    # ]
    le.fit(category)
    df['category1_encoding'] = le.transform(df.category1.values)
    return df

def category2_encoding(df):
    le = LabelEncoder()
    category = np.load('data/category2.npy', allow_pickle=True)
    le.fit(category)
    df['category2_encoding'] = le.transform(df.category2.values)
    return df

def goal_min_group(df):
    col_name = 'goal_min'
    col_names = ['country_encoding', 'category1_encoding', 'category2_encoding']
    for col in itertools.combinations(col_names, 2):
        col = list(col)
        group = df.groupby(col)[col_name].agg([np.mean, np.std]).add_prefix('{0}_{1}_{2}_'.format(col_name, col[0], col[1]))
        df = pd.merge(df, group, left_on=col, right_index=True)
    return df

def goal_max_group(df):
    col_name = 'goal_max'
    col_names = ['country_encoding', 'category1_encoding', 'category2_encoding']
    for col in itertools.combinations(col_names, 2):
        col = list(col)
        group = df.groupby(col)[col_name].agg([np.mean, np.std]).add_prefix('{0}_{1}_{2}_'.format(col_name, col[0], col[1]))
        df = pd.merge(df, group, left_on=col, right_index=True)
    return df

def duration_group(df):
    col_name = 'duration'
    col_names = ['country_encoding', 'category1_encoding', 'category2_encoding']
    for col in itertools.combinations(col_names, 2):
        col = list(col)
        group = df.groupby(col)[col_name].agg([np.mean, np.std]).add_prefix('{0}_{1}_{2}_'.format(col_name, col[0], col[1]))
        df = pd.merge(df, group, left_on=col, right_index=True)
    return df

