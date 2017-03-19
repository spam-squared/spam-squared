from mail_service.mail_service import MailService
from training_data.training_data_handler import TrainingDataHandler
from decision_tree.decision_tree import DecisionTree
from tensorflow.model_v2 import Tensorflow
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

if (not args.tensorflow):
    # Set up decision tree
    decision_tree = DecisionTree()
    decision_tree.train_tree(training_data_handler.get_training_data())
    decision_tree.test()
else:
    tensorflow = Tensorflow()
    restored = tensorflow.restore()


# Set up email service
mailService = MailService()