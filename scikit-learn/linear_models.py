# reference https://scikit-learn.org/stable/modules/linear_model.html
# y(w, x) = w0 + w1 * x1 + ... + xp * xp
# vector w = (w1,...,wp) -> coef_   w0 -> intercept_

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 1.1.1 Ordinary Least Squares
def ordinary_least_squares(X: list, y: list, weight: list = None):
    assert (len(X) == len(y))
    if weight:
        assert (len(X) == len(weight))
    reg = linear_model.LinearRegression()
    reg.fit(X, y, weight)
    return reg.intercept_, reg.coef_


# 1.1.1 examples
def linear_regression_example():
    # load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # split the targets into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # create linear regression object
    regr = linear_model.LinearRegression()

    # train the model using the training set
    regr.fit(diabetes_X_train, diabetes_y_train)

    # make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # the coefficients
    print('Coefficients: \n', regr.coef_)

    # the mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))

    # the coefficient of determination : 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    # plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# 1.1.1.1 Non-Negative Least Squares
def non_negative_least_squares():
    np.random.seed(42)

    n_samples, n_features = 200, 50
    X = np.random.randn(n_samples, n_features)
    true_coef = 3 * np.random.randn(n_features)
    # threshold coefficients to render them non-negative
    true_coef[true_coef < 0] = 0
    y = np.dot(X, true_coef)

    # add some noise
    y += 5 * np.random.normal(size=(n_samples,))

    # split the data in train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # fit the none-negative least squares
    reg_nnls = LinearRegression(positive=True)
    y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)
    r2_score_nnls = r2_score(y_test, y_pred_nnls)
    print("NNLS R2 score", r2_score_nnls)

    # fit an OLS
    reg_ols = LinearRegression()
    y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)
    r2_score_ols = r2_score(y_test, y_pred_ols)
    print("OLS R2 socre", r2_score_ols)

    # plot
    fig, ax = plt.subplots()
    ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth=0, marker=".")

    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    low = max(low_x, low_y)
    high = min(high_x, high_y)
    ax.plot([low, high], [low, high], ls="--", c=".3", alpha=.5)
    ax.set_xlabel("OLS regression coefficients", fontweight="bold")
    ax.set_ylabel("NNLS regression coefficients", fontweight="bold")
    plt.show()


# 1.1.2 Ridge regression and classification
# address ordinary least squares by imposing a penalty on the size of the coefficients
# 1.1.2.1 Ridge Regression
def ridge_regression(X: list, y: list, weight: list = None, alpha: float = 1.0):
    assert (len(X) == len(y))
    if weight:
        assert (len(X) == len(weight))
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X, y, weight)
    return reg.intercept_, reg.coef_


# 1.1.2.2 Ridge Classification
# 1.1.2.4 setting the regularization parameter: leave-one-out cross-validation
def ridge_classification(X: list, y: list):
    assert (len(X) == len(y))
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13)) # 等比数列，以 10 为底
    reg.fit(X, y)
    return reg.alpha_


if __name__ == '__main__':
    X_list = [[0, 0], [1, 1], [2, 2]]
    y_list = [0, 1, 2]
    weight_list = [0.2, 0.4, 0.1]
    _alpha = 0.5

    print(ordinary_least_squares(X_list, y_list))
    linear_regression_example()
    non_negative_least_squares()
    print(ridge_regression(X_list, y_list, weight_list, _alpha))
    print(ridge_classification(X_list, y_list))
