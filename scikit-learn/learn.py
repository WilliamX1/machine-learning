from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint


# implementation of https://scikit-learn.org/stable/getting_started.html


def fitting_and_predicting_estimator_basics():
    clf = RandomForestClassifier(random_state=0)
    X = [[1, 2, 3],  # 2 samples, 3 features
         [11, 12, 13]]
    y = [0, 1]  # classes of each sample
    clf.fit(X, y)
    print(clf.predict(X))
    print(clf.predict([[4, 5, 6], [14, 15, 16]]))


def transformers_and_pre_processors():
    X = [[0, 15], [1, -10]]
    print(StandardScaler().fit(X).transform(X))


def pipelines_chaining_pre_processors_and_estimators():
    # create a pipeline object
    pipe = make_pipeline(StandardScaler(), LogisticRegression())

    # load the iris dataset and split it into train and test sets
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # fit the whole pipeline
    pipe.fit(X_train, y_train)

    # we can now use it like any other estimator
    print(accuracy_score(pipe.predict(X_test), y_test))


def model_evaluation():
    X, y = make_regression(n_samples=1000, random_state=0)
    lr = LinearRegression()

    result = cross_validate(lr, X, y)  # defaults to 5-fold CV
    print(result['test_score'])


def automatic_parameter_searches():
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # define the parameter space that will be searched over
    param_distributions = {'n_estimators': randint(1, 5),
                           'max_depth': randint(5, 10)}

    # now create a searchCV object and fit it to the data
    search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                                n_iter=5,
                                param_distributions=param_distributions,
                                random_state=0)
    search.fit(X_train, y_train)

    print(search.best_params_)

    # the search object now acts like a normal random forest estimator
    # with max_depth=9 and n_estimators=4
    search.score(X_test, y_test)


if __name__ == '__main__':
    fitting_and_predicting_estimator_basics()
    transformers_and_pre_processors()
    pipelines_chaining_pre_processors_and_estimators()
    model_evaluation()
    automatic_parameter_searches()
