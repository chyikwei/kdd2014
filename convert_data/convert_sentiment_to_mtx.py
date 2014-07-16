import pandas as pd
import scipy.io as sio
import numpy as np
import scipy.sparse as sp


project_file = 'raw_data/projects.csv'
sentiment_file = 'sentiment_essays_no_lem.csv'

def convert():
    projects = pd.read_csv('raw_data/projects.csv')
    projects = projects.sort('projectid')

    sentiments = pd.read_csv(sentiment_file)
    sentiments = sentiments.sort('projectid')

    proj_sen = pd.merge(projects, sentiments, on='projectid')

    train = proj_sen[proj_sen['date_posted'] < '2014-01-01']
    test = proj_sen[proj_sen['date_posted'] >= '2014-01-01']

    train_2013 = train[train['date_posted'] >= '2013-01-01']

    fields = ['pos_cnt', 'neg_cnt', 'token_cnts']
    train_data = []
    for field in fields:
        train_data.append(train_2013[field])

    train_mtx = np.vstack(train_data).T
    train_mtx = sp.csr_matrix(train_mtx)
    sio.mmwrite('train_2013_sentiment_mtx', train_mtx)

    test_data = []
    for field in fields:
        test_data.append(test[field])

    test_mtx = np.vstack(test_data).T
    test_mtx = sp.csr_matrix(test_mtx)
    sio.mmwrite('test_sentiment_mtx', test_mtx)
