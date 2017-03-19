from mail_service.mail_service import MailService
from training_data.training_data_handler import TrainingDataHandler
from decision_tree.decision_tree import DecisionTree
#from tensorflow.model_v2 import Tensorflow
from responses.response_handler import ResponseHandler
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--tensorflow', dest='tensorflow', action='store_true')
parser.add_argument('--no-tensorflow', dest='tensorflow', action='store_false')
parser.set_defaults(tensorflow=False)

args = parser.parse_args()

# Set up database
training_data_handler = TrainingDataHandler()
training_data_handler.init_db()
training_data_handler.load_db()

# Set up decision tree
decision_tree = DecisionTree()
decision_tree.train_tree(training_data_handler.get_training_data())
decision_tree.test()

# Set up tensor flow
# tensorflow = Tensorflow()
# restored = tensorflow.restore()

# Set up mail service
mail_service = MailService()
mail_service.add_receiver(lambda data: update_model(data))


# if (not args.tensorflow):
#     # Set up decision tree
#     decision_tree = DecisionTree()
#     decision_tree.train_tree(training_data_handler.get_training_data())
#     decision_tree.test()
# else:
#     tensorflow = Tensorflow()
#     restored = tensorflow.restore()


# ----------------------------------------------------------------------
# Data Model
# ----------------------------------------------------------------------
def update_model(mail):
    training_data_handler.insert_sample(mail['body'], -1, mail['mail_from'])
    train_new_decison_tree(mail)
    predict_and_reply(mail)


def predict_and_reply(mail):
    global decision_tree
    decision_tree_prediction = decision_tree.predict(mail['body'])

    # Choose a response
    response_id = 0

    # Load responses
    response_handler = ResponseHandler()
    response_handler.load_responses()
    responses = response_handler.get_responses()

    # Send chosen response
    global mail_service
    mail_service.send_mail(mail['mail_from'], mail['subject'], responses[response_id])
    training_data_handler.set_target(mail['mail_from'], response_id)


# ----------------------------------------------------------------------
# Decision tree
# ----------------------------------------------------------------------
def train_new_decison_tree(mail):
    """
    :param mail: Mail: mail_from, subject, body
    :return:
    """
    new_decision_tree = DecisionTree()
    new_decision_tree.train_tree(training_data_handler.get_training_data())
    new_decision_tree.test()

    global decision_tree
    decision_tree = new_decision_tree
