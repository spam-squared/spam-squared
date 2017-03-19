import mail_parser
from sklearn import tree
from random import shuffle
from math import math
from functools import reduce

clf = None
train_data = None
test_data = None
TRAIN_DATA_PERCENTAGE = 0.7


def train_tree(mails):
    """Train the decision tree"""
    shuffle(mails)

    global train_data, test_data, TRAIN_DATA_PERCENTAGE
    train_data = mails[:math.floor(len(mails) * TRAIN_DATA_PERCENTAGE)]
    test_data = mails[math.floor(len(mails) * TRAIN_DATA_PERCENTAGE):]

    train_data = {
        "data": [mail_parser.parse_mail(mail.data) for mail in train_data],
        "target": [mail.target for mail in train_data]
    }

    global clf
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data['data'], train_data['target'])


def predict(mail_features):
    """Predict response for mail"""
    return clf.predict_proba(mail_features)


def test():
    """Calculates test accuracy of current tree"""
    global train_data, test_data, TRAIN_DATA_PERCENTAGE
    accuracy = reduce((lambda acc, mail: acc + predict(mail['data'])), test_data, 0)
    accuracy /= len(test_data)
    print("Test Accuracy: ", accuracy)
    return accuracy

