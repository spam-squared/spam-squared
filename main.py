from mail_service.mail_service import MailService
from training_data.training_data_handler import TrainingDataHandler
from decision_tree.decision_tree import DecisionTree

# Set up database
training_data_handler = TrainingDataHandler()
training_data_handler.init_db()
training_data_handler.load_db()

# Set up decision tree
decision_tree = DecisionTree()
decision_tree.train_tree(training_data_handler.get_training_data())
decision_tree.test()

# Set up email service
mailService = MailService()