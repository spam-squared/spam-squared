import functools as fp
import numpy as np
import tensorflow as tf
import math

batch_size = 512

# Exponential decay of learning rate
learning_rate = 0.4
decay_steps = 5
decay_rate = 0.1

# Dropout regularization
keep_prob = 0.5

# L2 regularization
beta_l2 = 0.001


def accuracy(predictions, labels):
    """Calculate accuracy"""
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def multilayer_network(x, weights, biases, train=False):
    """Apply hidden layers to network"""
    layer_h = x

    for i in range(0, len(weights) - 1):
        layer_h = tf.add(tf.matmul(x, weights['h' + str(i)]), biases['b' + str(i)])
        layer_h = tf.nn.relu(layer_h)
        if train:
            layer_h = tf.nn.dropout(layer_h, keep_prob)

    return tf.matmul(layer_h, weights['out']) + biases['out']


# Apply L2 regularization
def regulize_l2(weights, beta):
    return tf.multiply(beta, fp.reduce(lambda x, y: tf.add(x, y), map(lambda x: tf.nn.l2_loss(x), weights)))


n_input = image_size * image_size
n_classes = 10

graph = tf.Graph()

with graph.as_default():
    """Define tensor flow model"""

    weights = {
        'h0': tf.Variable(tf.random_normal([n_input, 1024])),
        'h1': tf.Variable(tf.random_normal([1024, 512])),
        'out': tf.Variable(tf.random_normal([512, n_classes]))
    }
    biases = {
        'b0': tf.Variable(tf.zeros([1024])),
        'b1': tf.Variable(tf.zeros([512])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    logits = multilayer_network(tf_train_dataset, weights, biases, train=True)
    loss = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)),
                  regulize_l2([v for v in weights.values()], beta_l2))

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(multilayer_network(tf_test_dataset, weights, biases))
    valid_prediction = tf.nn.softmax(multilayer_network(tf_valid_dataset, weights, biases))


acc_batch, acc_test, acc_valid = (np.array([]) for _ in range(3))

num_steps = 10001
save_intervall = 100;
k = 0

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % save_intervall == 0):
            acc_batch = np.append(acc_batch, accuracy(predictions, batch_labels))
            acc_test = np.append(acc_test, accuracy(test_prediction.eval(), test_labels))
            acc_valid = np.append(acc_valid, accuracy(valid_prediction.eval(), valid_labels))

        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % (acc_batch[-1]))
            # print("Minibatch accuracy: %.1f%%" % temp_acc)
            print("Validation accuracy: %.1f%%" % (acc_valid[-1]))
            print("Test accuracy: %.1f%%" % (acc_test[-1]))