import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from datetime import datetime, timedelta

year_file_name = 'all_project_year.csv'

def get_range(date, days=15):
    d = datetime.strptime(date, '%Y-%m-%d')
    start_date = d - timedelta(days=days)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = d + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')

def get_date_count(dates, date):
    return dates[dates == date].count()

def test():
    df = pd.read_csv(year_file_name)
    df = df.sort('projectid')

    min_date = '2002-09-13'
    max_date = '2014-05-12'
    dates = df['date_posted']

    start_date = datetime.strptime(min_date, '%Y-%m-%d')
    end_date = datetime.strptime(max_date, '%Y-%m-%d')

    date_list = []
    cnt_list = []
    current_date = start_date
    while current_date <= end_date:
        str_date = current_date.strftime('%Y-%m-%d')
        cnt = get_date_count(dates, str_date)
        print '%s: %d' % (str_date, cnt)
        date_list.append(str_date)
        cnt_list.append(cnt)
        current_date += timedelta(days=1)

    pd.DataFrame({'date': date_list, 'cnt': cnt_list}).to_csv('daily_proj_cnt.csv', index=False)

    date_count_30 = {}
    max_idx = len(cnt_list) - 1
    date_range = 7
    for idx in xrange(len(cnt_list)):
        date = start_date + timedelta(days=idx)
        str_date = date.strftime('%Y-%m-%d')

        begin_idx = max(0, idx - date_range)
        end_idx = min(max_idx, idx + date_range)

        count = sum(cnt_list[begin_idx:end_idx])
        duration = end_idx - begin_idx
        if duration !=  2 * date_range:
            count  = 2. * count * (float(date_range) / duration)

        date_count_30[str_date] = float(count)

    date_count_60 = {}
    date_range = 15
    for idx in xrange(len(cnt_list)):
        date = start_date + timedelta(days=idx)
        str_date = date.strftime('%Y-%m-%d')

        begin_idx = max(0, idx - date_range)
        end_idx = min(max_idx, idx + date_range)

        count = sum(cnt_list[begin_idx:end_idx])
        duration = end_idx - begin_idx
        if duration !=  2 * date_range:
            count  = 2. * count * (float(date_range) / duration)

        date_count_60[str_date] = float(count)    


    df['proj_norm_date'] = df['date_posted'].apply(lambda x: date_count_30[x] / (1. + date_count_60[x]))
    df['log_proj_cnt'] = df['proj_norm_date'].apply(lambda x: np.log(x))

    train = df[df['date_posted'] < '2014-01-01']
    train_2013 = train[train['date_posted'] >= '2013-01-01']
    test = df[df['date_posted'] >= '2014-01-01']

    train_2013 = train_2013.sort('projectid')
    test = test.sort('projectid')

    train_proj_cnt_arr = train_2013['proj_norm_date'].values
    train_proj_log_cnt_arr = train_2013['log_proj_cnt'].values
    train_mtx = np.vstack([train_proj_cnt_arr, train_proj_log_cnt_arr]).T
    train_mtx = sp.csr_matrix(train_mtx)

    test_proj_cnt_arr = test['proj_norm_date'].values
    test_proj_log_cnt_arr = test['log_proj_cnt'].values
    test_mtx = np.vstack([test_proj_cnt_arr, test_proj_log_cnt_arr]).T
    test_mtx = sp.csr_matrix(test_mtx)

    sio.mmwrite('train_2013_proj_cnt_norm_mtx_2', train_mtx)
    sio.mmwrite('test_proj_cnt_norm_mtx_2', test_mtx)