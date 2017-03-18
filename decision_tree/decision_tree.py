import mail_parser
from sklearn import tree

clf = None


def train_tree(mails):
    train_data = {
        "data": [mail_parser.parse_mail(mail.data) for mail in mails],
        "target": [mail.target for mail in mails]
    }

    global clf
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data['data'], train_data['target'])


def predict(mail_features):
    return clf.predict_proba(mail_features)
