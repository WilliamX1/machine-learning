# https://scikit-learn.org/stable/modules/tree.html
# from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz


# 1.10.1 only support labels are [-1, 1] or [0, ..., k - 1]
def decision_tree_classification(X: list, y: list, pred_list: list):
    assert (len(X) == len(y))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    print("The prediction: ", clf.predict(pred_list))
    print("The corresponding possibility: ", clf.predict_proba(pred_list))


# 1.10.1
def decision_tree_classification_example():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    tree.plot_tree(clf)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data, format='fig')
    graph.render("iris")
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    print(graph)


# 1.10.1
def decision_tree_classification_example_list():
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import export_text
    iris = load_iris()
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    decision_tree = decision_tree.fit(iris.data, iris.target)
    r = export_text(decision_tree, feature_names=iris['feature_names'])
    print(r)


# 1.10.2 support labels are float
def decision_tree_regression(X: list, y: list, pred_list: list):
    assert (len(X) == len(y))
    clf = tree.DecisionTreeClassifier().fit(X, y)
    print("The predict: ", clf.predict(pred_list))


# 1.10.3 multi-output problems


if __name__ == "__main__":
    X_list = [[0, 0], [1, 1]]
    y_list = [0, 1]
    predict_list = [[2, 2]]
    decision_tree_classification(X_list, y_list, predict_list)
    decision_tree_classification_example()
    decision_tree_classification_example_list()
    decision_tree_regression(X_list, y_list, predict_list)
