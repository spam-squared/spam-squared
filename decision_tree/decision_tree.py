from sklearn.datasets import load_iris
from sklearn import tree

clf = None


def train_tree(train_data):
    train_data = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data.data, train_data.target)


def predict(features):
    clf.predict_proba(iris.data[:1, :])
