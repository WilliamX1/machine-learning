# https://scikit-learn.org/stable/modules/neighbors.html
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


# 1.6.1.1
def finding_the_nearest_neighbors():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)  # 'auto', 'ball_tree', 'kd_tree', 'brute'
    distances, indices = nbrs.kneighbors(X)
    print("indices: \n", indices)
    print("distances: \n", distances)
    print("nbrs.kneighbors_graph(X).toarray(): \n", nbrs.kneighbors_graph(X).toarray())


def nearest_neighbors_classification_example():
    n_neighbors = 15

    # import some data to play with
    iris = datasets.load_iris()

    # only take the first two features.
    X = iris.data[:, :2]
    y = iris.target

    # step size in the mesh
    h = .02

    # create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_hold = ['darkorange', 'c', 'darkblue']

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], palette=cmap_hold, alpha=1.0, edgecolor="black")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s'" % (n_neighbors, weights))
        plt.xlabel(iris.feature_names[0])
        plt.ylabel(iris.feature_names[1])
    plt.show()


def face_completion_with_a_multi_output_estimators():
    # load the faces datasets
    data, targets = fetch_olivetti_faces(return_X_y=True)

    train = data[targets < 30]
    test = data[targets >= 30] # test on independent people

    # Test on a subset of people
    n_faces = 5
    rng = check_random_state(4)
    face_ids = rng.randint(test.shape[0], size=(n_faces, ))
    test = test[face_ids, :]

    n_pixels = data.shape[1]
    # Upper half of the faces
    X_train = train[:, :(n_pixels + 1) // 2]
    # Lower half of the faces
    y_train = train[:, n_pixels // 2:]
    X_test = test[:, :(n_pixels + 1) // 2]
    y_test = test[:, n_pixels // 2:]

    # Fit estimators
    ESTIMATORS = {
        "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
        "K-nn": KNeighborsRegressor(),
        "Linear regression": LinearRegression(),
        "Ridge": RidgeCV(),
    }

    y_test_predict = dict()
    for name, estimator in ESTIMATORS.items():
        estimator.fit(X_train, y_train)
        y_test_predict[name] = estimator.predict(X_test)

    # Plot the completed faces
    image_shape = (64, 64)

    n_cols = 1 + len(ESTIMATORS)
    plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
    plt.suptitle("Face completion with multi-output estimators", size=16)

    for i in range(n_faces):
        true_face = np.hstack((X_test[i], y_test[i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")

        sub.axis("off")
        sub.imshow(true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")

        for j, est in enumerate(sorted(ESTIMATORS)):
            completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

            if i:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
            else:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)

            sub.axis("off")
            sub.imshow(completed_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")
    plt.show()


if __name__ == '__main__':
    finding_the_nearest_neighbors()
    nearest_neighbors_classification_example()
    face_completion_with_a_multi_output_estimators()
