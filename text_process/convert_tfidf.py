import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import numpy as np
import cPickle as pickle
from itertools import groupby


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


def dump_word_dict(d, file_name):
    with open(file_name, 'wb') as f:
        for k,v in d.iteritems():
            f.write('%s,%d\n' % (k, v))
        print 'done'


def data_value(tf, df, N, dict_size):
    smooth_df = df + 1.0
    smooth_N = N + dict_size

    idf = np.log(float(smooth_N) / smooth_df) + 1.0
    return (tf * idf)


def build_sp_matrix(texts, kw_dict, shape):
    data = []
    indices = []
    idxptr = [0]
    acc_idxptr = 0
    for idx, essay in enumerate(texts):
        local_data = []
        local_col_idx = []
        tokens = essay.split('|')
        for t in tokens:
            word, tf, df = t.split(':')
            tf = int(tf)
            df = int(df)
            col_idx  = kw_dict.get(word)
            if col_idx:
                val = data_value(tf, df, shape[0], len(kw_dict))
                #print '%s, %d, %d, %d, %.3f' % (word, tf, df, shape[0], val)
                local_data.append(val)
                local_col_idx.append(col_idx)
        if local_data:
            data += local_data
            indices += local_col_idx
        acc_idxptr += len(local_col_idx)
        idxptr.append(acc_idxptr)
        if idx % 1000 == 0:
            print "%d : %d" % (idx, acc_idxptr)

    print len(indices)
    print len(data)
    print len(idxptr)
    matrix = sp.csr_matrix((data,indices,idxptr), dtype=float, shape=shape)
    return matrix


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


def convert_with_dict(texts, kw_dict, shape, verbose=True):
    ret = []
    for idx, essay in enumerate(texts):
        local_data = []
        local_col_idx = []
        tokens = essay.split('|')
        for t in tokens:
            word, tf, df = t.split(':')
            tf = int(tf)
            df = int(df)
            col_idx  = kw_dict.get(word)
            if col_idx:
                val = data_value(tf, df, shape[0])
                print '%s, %d, %d, %d, %.3f' % (word, tf, df, shape[0], val)
                local_data.append(val)
                local_col_idx.append(col_idx)
        local_data_str = ",".join(['%.4f' % d for d in local_data])
        local_col_idx_str = ",".join(['%d' % d for d in local_col_idx])
        if idx % 1000 == 0:
            print "%d : %s" % (idx, local_col_idx_str)
        ret.append("%s|%s" % (local_col_idx_str, local_data_str))

    return ret


def build_test_sp_matrix(w_dict, dfs, N, essay_list):
    data = []
    indices = []
    idxptr = [0]
    acc_idxptr = 0
    dict_size = len(w_dict)
    for essay_idx, essay in enumerate(essay_list):
        tokens = essay.split('|')
        tokens = sorted(tokens)
        tokens = [(k, len(list(g))) for k, g in groupby(tokens)]
        
        local_data = []
        local_col_idx = []
        for word, tf in tokens:
            idx = w_dict.get(word)
            if idx != None:
                df = dfs[word]
                val = data_value(tf, df , N, dict_size)
                local_data.append(val)
                local_col_idx.append(idx)
        if local_data:
            data += local_data
            indices += local_col_idx
        acc_idxptr += len(local_col_idx)
        idxptr.append(acc_idxptr)
        if essay_idx % 1000 == 0:
            print "%d : %s" % (essay_idx, acc_idxptr)

    shape = (len(essay_list), len(w_dict))
    matrix = sp.csr_matrix((data,indices,idxptr), dtype=float, shape=shape)
    return matrix


def build_train_matrix():
    projects = pd.read_csv('projects.csv')
    essays = pd.read_csv('tfidf/tfidf_tokens.csv')
    total_essays = essays.shape[0]

    #projects = projects.sort('projectid')
    essays = essays.sort('projectid')
    #ess_proj = pd.merge(essays, projects, on='projectid')

    #train_idx = ess_proj['date_posted'] < '2014-01-01'
    #train_essay_data = ess_proj[train_idx]['tokens']
    train_essay_data = essays['tokens']
    keyword_dict = build_word_dict('word_5k/top_5k_tokens.txt')
    dump_word_dict(keyword_dict, 'word_5k/dict_top_5k_tokens.txt')

    num_row = train_essay_data.shape[0]
    num_col = len(keyword_dict)
    shape = (num_row, num_col)
    print "row=%d, col=%d" % (num_row, num_col)

    ret = build_sp_matrix(train_essay_data, keyword_dict, shape)
    #ret_1 = convert_with_dict(train_essay_data[:100000], keyword_dict, shape)
    print ret.shape
    sio.mmwrite('word_5k/train_essay_mtx', ret)


def build_test_matrix():
    file_name = 'test_tokens.csv'
    test_data = pd.read_csv(file_name)
    test_data = test_data.sort('projectid')
    test_tokens = test_data['tokens']
    keyword_dict = build_word_dict('word_5k/top_5k_tokens.txt')
    dfs = build_df(keyword_dict, 'general_stats/token_dfs.txt')
    print "keyword/dfs size %d/%d" % (len(keyword_dict), len(dfs))
    train_size = 619326

    ret = build_test_sp_matrix(keyword_dict, dfs, train_size, test_tokens)
    print ret.shape
    sio.mmwrite('word_5k/test_essay_mtx', ret)
    print 'done'


if __name__ == '__main__':
    build_test_matrix()
