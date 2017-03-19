from decision_tree.mail_parser import MailParser
from sklearn import tree
from random import shuffle
import math
from random import randint
from functools import reduce


class DecisionTree(object):
    empty = True
    clf = None
    train_data = None
    test_data = None
    TRAIN_DATA_PERCENTAGE = 0.7

    def train_tree(self, mails):
        if len(mails) == 0:
            self.empty = True
            return randint(0, 7)

        self.empty = False

        """Train the decision tree"""
        shuffle(mails)

        mails = list(map(lambda row: [row[3], row[1]], mails))

        self.train_data = mails[:math.floor(len(mails) * self.TRAIN_DATA_PERCENTAGE)]
        self.test_data = mails[math.floor(len(mails) * self.TRAIN_DATA_PERCENTAGE):]

        mail_parser = MailParser()
        self.train_data = {
            "data": [mail_parser.parse_mail(mail.data) for mail in self.train_data],
            "target": [mail.target for mail in self.train_data]
        }

        self.clf = tree.DecisionTreeClassifier()
        self.clf = self.clf.fit(self.train_data['data'], self.train_data['target'])

    def predict(self, mail_features):
        if self.empty:
            return 2

        """Predict response for mail"""
        return self.clf.predict_proba(mail_features)

    def test(self):
        if self.empty:
            print("Test Accuracy: No data")
            return -1

        """Calculates test accuracy of current tree"""
        accuracy = reduce((lambda acc, mail: acc + self.predict(mail['data'])), self.test_data, 0)
        accuracy /= len(self.test_data)
        print("Test Accuracy: ", accuracy)
        return accuracy
