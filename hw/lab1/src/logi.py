import time
import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_data():
    __filepath__ = ['./data/X_train.csv', './data/Y_train.csv', './data/X_test.csv', './data/Y_test.csv']
    __X_train__ = pd.read_csv(__filepath__[0]).values
    __X_train__ = preprocessing.scale(__X_train__)  # 均值-标准差缩放
    # __X_train__ = preprocessing.MinMaxScaler().fit_transform(__X_train__)  # min-max 标准化
    # __X_train__ = preprocessing.MaxAbsScaler().fit_transform(__X_train__)  # 最大值标准化
    # __X_train__ = preprocessing.normalize(__X_train__, norm='l2')
    # __X_train__ = preprocessing.Binarizer().fit_transform(__X_train__)  # 二值化

    __Y_train__ = pd.read_csv(__filepath__[1])['label']
    __Y_train__ = __Y_train__.values

    __X_test__ = pd.read_csv(__filepath__[2]).values
    __X_test__ = preprocessing.scale(__X_test__)  # 有效！！
    # __X_test__ = preprocessing.MinMaxScaler().fit_transform(__X_test__)
    # __X_test__ = preprocessing.MaxAbsScaler().fit_transform(__X_test__)
    # __X_test__ = preprocessing.normalize(__X_test__, norm='l2')
    # __X_test__ = preprocessing.Binarizer().fit_transform(__X_test__)

    __Y_test__ = pd.read_csv(__filepath__[3])['label'].values
    return __X_train__, __Y_train__, __X_test__, __Y_test__


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    # limit = np.sqrt(1 / dim)
    # w = np.random.uniform(-limit, limit, (dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    """
    传参:
    w -- 权重, shape： (num_px * num_px * 3, 1)
    b -- 偏置项, 一个标量
    X -- 数据集，shape： (num_px * num_px * 3, m),m为样本数
    Y -- 真实标签，shape： (1,m)

    返回值:
    cost， dw ，db，后两者放在一个字典grads里
    """
    # 获取样本数 m：
    m = X.shape[1]

    # 前向传播 ：
    A = sigmoid(np.dot(w.T, X) + b)  # 调用前面写的sigmoid函数
    cost = - (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m

    # 反向传播：
    dZ = A - Y
    dw = (np.dot(X, dZ.T)) / m
    db = (np.sum(dZ)) / m

    # 返回值：
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False, show_iteration_accuracy=False):
    # 定义一个 costs 数组，存放每若干次迭代后的 cost ，从而可以画图看看 cost 的变化趋势：
    costs = []
    # 进行迭代：
    for i in range(num_iterations):
        # 用 propagate 计算出每次迭代后的cost和梯度：
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        # 用上面得到的梯度来更新参数：
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 每 100 次迭代，保存一个 cost 看看：
        if i % 100 == 0:
            costs.append(cost)

        # 这个可以不在意，我们可以每 100 次把 cost 打印出来看看，从而随时掌握模型的进展：
        if print_cost and i % 100 == 0 and show_iteration_accuracy is False:
            print("Cost after iteration %i: %f" % (i, cost))

    # 迭代完毕，将最终的各个参数放进字典，并返回：
    params = {"w": w,
              "b": b}
    return params, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(m):
        if A[0, i] > 0.48:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction


def logistic_model(__X_train__, __Y_train__, __X_test__, __Y_test__, learning_rate=0.1, num_iterations=2000,
                   print_cost=False, show_iteration_accuracy=True):
    # 获特征维度，初始化参数：
    global prediction_test, costs, prediction_train, accuracy_train, accuracy_test
    dim = __X_train__.shape[0]
    W, b = initialize_with_zeros(dim)

    # 梯度下降，迭代求出模型参数：
    if show_iteration_accuracy:
        for i in range(num_iterations + 1):

            if i % 100 != 0: continue

            params, costs = optimize(W, b, __X_train__, __Y_train__, num_iterations, learning_rate, print_cost,
                                     show_iteration_accuracy)
            W = params['w']
            b = params['b']

            # 用学得的参数进行预测：
            prediction_train = predict(W, b, __X_train__)
            prediction_test = predict(W, b, __X_test__)

            # 计算准确率，分别在训练集和测试集上：
            accuracy_train = 1 - np.mean(np.abs(prediction_train - __Y_train__))
            accuracy_test = 1 - np.mean(np.abs(prediction_test - __Y_test__))
            print("After iteration %d Accuracy on train set: %f" % (i, accuracy_train))
            print("After iteration %d Accuracy on test set: %f" % (i, accuracy_test))
    else:
        params, costs = optimize(W, b, __X_train__, __Y_train__, num_iterations, learning_rate, print_cost,
                                 show_iteration_accuracy)
        W = params['w']
        b = params['b']

        # 用学得的参数进行预测：
        prediction_train = predict(W, b, __X_train__)
        prediction_test = predict(W, b, __X_test__)

        # 计算准确率，分别在训练集和测试集上：
        accuracy_train = 1 - np.mean(np.abs(prediction_train - __Y_train__))
        accuracy_test = 1 - np.mean(np.abs(prediction_test - __Y_test__))
        print("Accuracy on train set:", accuracy_train)
        print("Accuracy on test set:", accuracy_test)

    # 为了便于分析和检查，我们把得到的所有参数、超参数都存进一个字典返回出来：
    d = {"costs": costs,
         "Y_prediction_test": prediction_test,
         "Y_prediction_train": prediction_train,
         "w": W,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "train_acy": accuracy_train,
         "test_acy": accuracy_test
         }
    return d


if __name__ == '__main__':
    start_time = time.time()

    X_train, Y_train, X_test, Y_test = load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).T
    Y_train = Y_train.reshape(Y_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T
    Y_test = Y_test.reshape(Y_test.shape[0], -1).T

    d = logistic_model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.12,
                       print_cost=True, show_iteration_accuracy=False)

    end_time = time.time()
    tot_time = end_time - start_time
    print("INFO: %f average correct rate, %f average time consume" % (d['train_acy'], tot_time))
