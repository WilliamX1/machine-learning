import time
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB


def load_data():
    __filepath__ = ['./data/X_train.csv', './data/Y_train.csv', './data/X_test.csv', './data/Y_test.csv']
    __X_train__ = pd.read_csv(__filepath__[0]).values
    # __X_train__ = preprocessing.scale(__X_train__)  # 均值-标准差缩放
    # __X_train__ = preprocessing.MinMaxScaler().fit_transform(__X_train__)  # min-max 标准化
    # __X_train__ = preprocessing.MaxAbsScaler().fit_transform(__X_train__)  # 最大值标准化
    __X_train__ = preprocessing.normalize(__X_train__, norm='l2')  # 规范化 有效！！
    # __X_train__ = preprocessing.Binarizer().fit_transform(__X_train__)  # 二值化

    __Y_train__ = pd.read_csv(__filepath__[1])['label'].values

    __X_test__ = pd.read_csv(__filepath__[2]).values
    # __X_test__ = preprocessing.scale(__X_test__)
    # __X_test__ = preprocessing.MinMaxScaler().fit_transform(__X_test__)
    # __X_test__ = preprocessing.MaxAbsScaler().fit_transform(__X_test__)
    __X_test__ = preprocessing.normalize(__X_test__, norm='l2')
    # __X_test__ = preprocessing.Binarizer().fit_transform(__X_test__)

    __Y_test__ = pd.read_csv(__filepath__[3])['label'].values
    return __X_train__, __Y_train__, __X_test__, __Y_test__


def bayes_classify():
    # start the timer
    start_time = time.time()

    # load data
    X_train, Y_train, X_test, Y_test = load_data()

    # train model : Naive Bayes
    gnb = GaussianNB()
    # mnb = MultinomialNB()
    # cnb = ComplementNB()
    # bnb = BernoulliNB()
    # ctnb = CategoricalNB()
    nb = gnb

    nb.fit(X_train, Y_train)

    # make predictions
    Y_pred = nb.predict(X_test)

    # announce result
    test_size = len(X_test)
    mispred_size = (Y_test != Y_pred).sum()
    correct_rate = 1 - mispred_size / test_size
    print("INFO: Number of mislabeled points out of a total %d points : %d, with correct rate %f"
          % (test_size, mispred_size, correct_rate))

    # end the timer
    end_time = time.time()
    tot_time = end_time - start_time

    print("INFO: Bayes algorithm - time cost : %f s" % tot_time)

    return correct_rate, tot_time


if __name__ == '__main__':
    N = 10
    correct_rate_tot = 0.0
    time_tot = 0.0

    for i in range(N):
        cr, tt = bayes_classify()
        correct_rate_tot = correct_rate_tot + cr
        time_tot = time_tot + tt

    avg_cr_tot = correct_rate_tot / N
    avg_tt = time_tot / N

    print("INFO: Result: %d times, %f average correct rate, %f average time consume" % (N, avg_cr_tot, avg_tt))
