import scipy.io as sio
import numpy as np
import scipy.sparse as sp
import pandas as pd

from sklearn import metrics
from sklearn import cross_validation
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

tfidf_train_file = 'train_tfidf_2013_mtx.mtx'
tfidf_test_file = 'test_tfidf_mtx.mtx'

def main():
    train_tfidf = sio.mmread(tfidf_train_file)
    test_tfidf = sio.mmread(tfidf_test_file)

    svd = TruncatedSVD(400)
    svd_X_train = svd.fit_transform(train_tfidf)
    svd_X_test = svd.transform(test_tfidf)

    sio.mmwrite('train_tfidf_2013_svd_400_mtx', svd_X_train)
    sio.mmwrite('test_tfidf_svd_400_mtx', svd_X_test)

