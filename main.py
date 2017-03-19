from sendgrid import SendGrid
from training_data_handler import TrainingDataHandler
from decision_tree import DecisionTree

# Set up database
training_data_handler = TrainingDataHandler()
training_data_handler.init_db(training_data_handler)
training_data_handler.load_db(training_data_handler)

# Set up decision tree
decision_tree = DecisionTree()
decision_tree.train_tree(training_data_handler.get_training_data(training_data_handler))
decision_tree.test(decision_tree)

# Set up email service
sendgrid = SendGrid()
sendgrid.add_reciever(sendgrid, lambda sample: training_data_handler.insert_sample(training_data_handler, sample, -1))
sendgrid.add_reciever(sendgrid, lambda sample: decision_tree.predict(training_data_handler, sample))