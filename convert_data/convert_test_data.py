import pandas as pd
from itertools import groupby
import scipy.sparse as sp
import numpy as np

# load data paste
projects = pd.read_csv('projects.csv')
projects = projects.sort('projectid')
essay_tokens = pd.read_csv('essay_tokens.csv')
essay_tokens = essay_tokens.sort('projectid')

essay_proj = pd.merge(essay_tokens, projects, on='projectid')

test_idx = essay_proj['date_posted'] >= '2014-01-01'
test_data = essay_proj[test_idx]
test_data.to_csv('test_tokens.csv', index=False)

test_tokens = test_data['tokens']


def build_word_dict(file_name):
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        lines = filter(lambda x: len(x) > 3, lines)
        words = [l.split(",")[0] for l in lines]

        ret = {}
        for (idx, word) in enumerate(words):
            ret[word] = idx

        return ret

def load_dict(file_name):
    """
    file format: word:idx
    """
    w_dict = {}
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = [line.strip('\n') for line in lines]
        for line in lines:
            tokens = line.split(":")
            if len(tokens) != 2:
                print 'invalid line %s' % line
                continue
            word = tokens[0]
            idx = int(tokens[1])
            w_dict[word] = idx
        return w_dict


def build_df(kw, file_name):
    dfs = {}
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = [line.strip('()\n') for line in lines]
        for line in lines:
            tokens = line.split(",")
            if len(tokens) != 2:
                print 'invalid line %s' % line
                continue
            word = tokens[1]
            df = int(tokens[0])
            if word in kw:
                dfs[word] = df
        print "df count: %d" % len(dfs)
        return dfs

w_dict = load_dict('dict_1k_2k.csv')
df = build_df(w_dict, 'df_tokens/token_dfs.txt')


def data_value(tf, df, N):
    return tf * np.log((N *1.0)/ df)


def build_test_matrix(w_dict, dfs, N, essay_list):
    data = []
    indices = []
    idxptr = [0]
    acc_idxptr = 0
    dict_size = len(w_dict)
    for essay in essay_list:
        tokens = essay.split('|')
        tokens = sorted(tokens)
        tokens = [(k, len(list(g))) for k, g in groupby(tokens)]
        
        local_data = []
        local_col_idx = []
        for word, tf in tokens:
            idx = w_dict.get(word)
            if idx != None:
                df = dfs[word]
                val = data_value(tf, df , N)
                local_data.append(val)
                local_col_idx.append(idx)
        if local_data:
            data += local_data
            indices += local_col_idx
        acc_idxptr += len(local_col_idx)
        idxptr.append(acc_idxptr)

    shape = (len(essay_list), len(w_dict))
    matrix = sp.csr_matrix((data,indices,idxptr), dtype=float, shape=shape)
    return matrix


def convert_test_data():
    file_name = 'test_tokens.csv'
    test_data = pd.read_csv(file_name)
    test_data = test_data.sort('projectid')
    test_tokens = test_data['tokens']

