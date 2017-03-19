import mail_parser
from sklearn import tree
from random import shuffle
from math import math
from functools import reduce


class DecisionTree(object):
    clf = None
    train_data = None
    test_data = None
    TRAIN_DATA_PERCENTAGE = 0.7

    def train_tree(self, mails):
        """Train the decision tree"""
        shuffle(mails)

        self.train_data = mails[:math.floor(len(mails) * self.TRAIN_DATA_PERCENTAGE)]
        self.test_data = mails[math.floor(len(mails) * self.TRAIN_DATA_PERCENTAGE):]

        self.train_data = {
            "data": [mail_parser.parse_mail(mail.data) for mail in self.train_data],
            "target": [mail.target for mail in self.train_data]
        }

        self.clf = tree.DecisionTreeClassifier()
        self.clf = self.clf.fit(self.train_data['data'], self.train_data['target'])

    def predict(self, mail_features):
        """Predict response for mail"""
        return self.clf.predict_proba(mail_features)

    def test(self):
        """Calculates test accuracy of current tree"""
        accuracy = reduce((lambda acc, mail: acc + self.predict(mail['data'])), self.test_data, 0)
        accuracy /= len(self.test_data)
        print("Test Accuracy: ", accuracy)
        return accuracy
