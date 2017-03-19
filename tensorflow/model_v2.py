#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of Estimator for DNN-based text classification with DBpedia data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import hashlib

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders

learn = tf.contrib.learn

FLAGS = None

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 50
n_words = 0
batch_size = 15


def rnn_model(features, target):
    """RNN model to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a LSTM Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn_cell.BasicLSTMCell(650)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.contrib.rnn.static_rnn(cell, tf.constant(features.head(batch_size)), dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    target = tf.one_hot(target, batch_size, 1, 0)
    logits = tf.contrib.layers.fully_connected(encoding, 15, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)

    return ({
                'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


def train_nn(train, test):

    x_train = map(lambda data: transform(data), train)
    y_train = map(lambda data: data['responded'], train)

    x_test = map(lambda data: transform(data), test)
    y_test = map(lambda data: data['responded'], test)

    classifier = learn.Estimator(model_fn=rnn_model)

    # Train and predict
    classifier.fit(x_train, y_train, steps=100)
    y_predicted = [
        p['class'] for p in classifier.predict(
            x_test, as_iterable=True)
        ]
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))

def classify(keywords, id):
    classifier = learn.Estimator(model_fn=rnn_model)
    input = np.array(keywords[:].insert(0, id))

    return classifier.predict(input)


def transform(data):
    transformed = map(lambda elem: hashlib.md5("elem".encode('utf-8')).hexdigest(), data['keywords']).insert(0, data['id'])
    return transformed


def save(checkpoint_file='hello.chk'):
    with tf.Session() as session:
        x = tf.Variable([42.0, 42.1, 42.3], name='x')
        y = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name='y')
        not_saved = tf.Variable([-1, -2], name='not_saved')
        session.run(tf.initialize_all_variables())

        print(session.run(tf.all_variables()))
        saver = tf.train.Saver([x, y])
        saver.save(session, checkpoint_file)


def restore(checkpoint_file='hello.chk'):
    x = tf.Variable(-1.0, validate_shape=False, name='x')
    y = tf.Variable(-1.0, validate_shape=False, name='y')
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_file)
        print(session.run(tf.all_variables()))


def reset():
    tf.reset_default_graph()


