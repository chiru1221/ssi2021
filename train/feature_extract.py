import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
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

def goal_min_group(df, others):
    col_name = 'goal_min'
    col_names = ['country_encoding', 'category1_encoding', 'category2_encoding']
    # col_names = ['country_encoding', 'category1_encoding']
    for col in itertools.combinations(col_names, 2):
        col = list(col)
        group = df.groupby(col)[col_name].agg([np.mean, np.std]).add_prefix('{0}_{1}_{2}_'.format(col_name, col[0], col[1]))
        df = pd.merge(df, group, left_on=col, how='left', right_index=True)
        for i, other in enumerate(others):
            others[i] = pd.merge(other, group, left_on=col, how='left', right_index=True)
    return df, others

def goal_max_group(df, others):
    col_name = 'goal_max'
    col_names = ['country_encoding', 'category1_encoding', 'category2_encoding']
    # col_names = ['country_encoding', 'category1_encoding']
    for col in itertools.combinations(col_names, 2):
        col = list(col)
        group = df.groupby(col)[col_name].agg([np.mean, np.std]).add_prefix('{0}_{1}_{2}_'.format(col_name, col[0], col[1]))
        df = pd.merge(df, group, left_on=col, how='left', right_index=True)
        for i, other in enumerate(others):
            others[i] = pd.merge(other, group, left_on=col, how='left', right_index=True)
    return df, others

def duration_group(df, others):
    col_name = 'duration'
    col_names = ['country_encoding', 'category1_encoding', 'category2_encoding']
    # col_names = ['country_encoding', 'category1_encoding']
    for col in itertools.combinations(col_names, 2):
        col = list(col)
        group = df.groupby(col)[col_name].agg([np.mean, np.std]).add_prefix('{0}_{1}_{2}_'.format(col_name, col[0], col[1]))
        df = pd.merge(df, group, left_on=col, how='left', right_index=True)
        for i, other in enumerate(others):
            others[i] = pd.merge(other, group, left_on=col, how='left', right_index=True)
    return df, others

def text_to_word_count(df, others, thre=0.25, del_html_content=True):
    to_text = lambda x: BeautifulSoup(x, 'lxml').text
    df['html_content_text'] = df.html_content.apply(to_text).copy()
    for i, other in enumerate(others):
        others[i]['html_content_text'] = other.html_content.apply(to_text).copy()
    
    vectorizer = CountVectorizer()
    corpus = df.html_content_text
    vectorizer.fit(corpus)
    
    x = vectorizer.transform(corpus).toarray()
    col_mean = np.mean(x, axis=0)
    col_idx = (col_mean >= thre)
    x_reduce = x[:, col_idx]

    if del_html_content:
        drop_col = ['html_content', 'html_content_text']
    else:
        drop_col = ['html_content_text']

    df = df.drop(columns=drop_col)
    add_df = pd.DataFrame(x_reduce).add_prefix('word_count_')
    df, add_df = df.reset_index(drop=True), add_df.reset_index(drop=True)
    df = pd.concat([df, add_df], axis=1)

    for i, other in enumerate(others):
        x = vectorizer.transform(other.html_content_text).toarray()
        x_reduce = x[:, col_idx]
        other = other.drop(columns=drop_col)
        add_df = pd.DataFrame(x_reduce).add_prefix('word_count_')
        other, add_df = other.reset_index(drop=True), add_df.reset_index(drop=True)
        others[i] = pd.concat([other, add_df], axis=1)
    return df, others

def text_length(df):
    to_text = lambda x: BeautifulSoup(x, 'lxml').text
    df['html_content_text_len'] = df.html_content.apply(to_text).str.len()
    return df


def text_to_h1_word_count(df, others, thre=0.001, del_html_content=True):
    to_text = lambda x: BeautifulSoup(x, 'lxml').find('h1').text if BeautifulSoup(x, 'lxml').find('h1') is not None else ''
    df['html_content_text'] = df.html_content.apply(to_text).copy()
    for i, other in enumerate(others):
        others[i]['html_content_text'] = other.html_content.apply(to_text).copy()
    
    vectorizer = CountVectorizer()
    corpus = df.html_content_text
    vectorizer.fit(corpus)
    
    x = vectorizer.transform(corpus).toarray()
    col_mean = np.mean(x, axis=0)
    col_idx = (col_mean >= thre)
    x_reduce = x[:, col_idx]

    if del_html_content:
        drop_col = ['html_content', 'html_content_text']
    else:
        drop_col = ['html_content_text']

    df = df.drop(columns=drop_col)
    add_df = pd.DataFrame(x_reduce).add_prefix('h1_word_count_')
    df, add_df = df.reset_index(drop=True), add_df.reset_index(drop=True)
    df = pd.concat([df, add_df], axis=1)

    for i, other in enumerate(others):
        x = vectorizer.transform(other.html_content_text).toarray()
        x_reduce = x[:, col_idx]
        other = other.drop(columns=drop_col)
        add_df = pd.DataFrame(x_reduce).add_prefix('h1_word_count_')
        other, add_df = other.reset_index(drop=True), add_df.reset_index(drop=True)
        others[i] = pd.concat([other, add_df], axis=1)
    return df, others


def text_to_word_percent(df, others, thre=0.25):
    to_text = lambda x: BeautifulSoup(x, 'lxml').text
    df['html_content_text'] = df.html_content.apply(to_text).copy()
    for i, other in enumerate(others):
        others[i]['html_content_text'] = other.html_content.apply(to_text).copy()
    
    vectorizer = CountVectorizer()
    corpus = df.html_content_text
    vectorizer.fit(corpus)
    
    x = vectorizer.transform(corpus).toarray()
    col_mean = np.mean(x, axis=0)
    col_idx = (col_mean >= thre)
    x_reduce = x[:, col_idx]
    x_reduce = x_reduce / np.sum(x_reduce, axis=1).reshape(-1, 1)
    x_reduce = np.nan_to_num(x_reduce)

    df = df.drop(columns=['html_content', 'html_content_text'])
    add_df = pd.DataFrame(x_reduce).add_prefix('word_count_')
    df, add_df = df.reset_index(drop=True), add_df.reset_index(drop=True)
    df = pd.concat([df, add_df], axis=1)

    for i, other in enumerate(others):
        x = vectorizer.transform(other.html_content_text).toarray()
        x_reduce = x[:, col_idx]
        x_reduce = x_reduce / np.sum(x_reduce, axis=1).reshape(-1, 1)
        x_reduce = np.nan_to_num(x_reduce)
        other = other.drop(columns=['html_content', 'html_content_text'])
        add_df = pd.DataFrame(x_reduce).add_prefix('word_percent_')
        other, add_df = other.reset_index(drop=True), add_df.reset_index(drop=True)
        others[i] = pd.concat([other, add_df], axis=1)
    return df, others

def html_content_figure_count(df):
    figure_count = lambda x: len(BeautifulSoup(x, 'lxml').find_all('figure'))
    df['html_content_figure_count'] = df.html_content.apply(figure_count)
    return df

def target_encoding(df, label, others, column):
    df_copy = df.copy()
    df_copy['label'] = label

    group = df_copy.groupby(column)['label'].agg([np.mean]).add_prefix('{0}_{1}_'.format(column, 'target'))
    df = pd.merge(df, group, left_on=column, how='left', right_index=True)
    for i, other in enumerate(others):
        others[i] = pd.merge(other, group, left_on=column, how='left', right_index=True)
    return df, others

def multi_target_encoding(df, label, others, columns):
    df_copy = df.copy()
    df_copy['label'] = label

    for col in itertools.combinations(columns, 2):
        col = list(col)
        group = df_copy.groupby(col)['label'].agg([np.mean, np.std]).add_prefix('{0}_{1}_{2}_'.format('target', col[0], col[1]))
        df = pd.merge(df, group, left_on=col, how='left', right_index=True)
        for i, other in enumerate(others):
            others[i] = pd.merge(other, group, left_on=col, how='left', right_index=True)
    return df, others
