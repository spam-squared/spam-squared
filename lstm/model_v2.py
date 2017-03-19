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
import math
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders

from random import shuffle



class Tensorflow(object):
    learn = tf.contrib.learn

    FLAGS = None

    MAX_DOCUMENT_LENGTH = 10
    EMBEDDING_SIZE = 50
    n_words = 0
    batch_size = 15

    def rnn_model(self, features, target):
        # Create a LSTM Unit cell with hidden size of EMBEDDING_SIZE.
        cell = tf.nn_cell.BasicLSTMCell(650)

        # Create an unrolled Recurrent Neural Networks to length of
        # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
        _, encoding = tf.contrib.rnn.static_rnn(cell, tf.constant(features.head(self.batch_size), name='not_saved'),
                                                dtype=tf.float32)

        # Given encoding of RNN, take encoding of last step (e.g hidden size of the
        # neural network of last step) and pass it as features for logistic
        # regression over output classes.
        target = tf.one_hot(target, self.batch_size, 1, 0)
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

    def transformDB(self, row):
        body = row[3]
        keywords = self.keyword.getKeywords(body)
        return {'keywords': keywords, 'id': row[1], 'responded': row[2]}

    def train(self, data):
        data = map(self.transformDB, filter(lambda row: row[1] != -1, data))
        shuffle(data)
        lenTrain = math.floor(len(data) * 0.7)
        train = data[:lenTrain]
        test = data[lenTrain+1, len(data)]
        self.train_nn(train, test)

    # train, test = [{'keywords':[],‘id':1,'responded':True}]
    def train_nn(self, train, test):
        x_train = map(lambda data: self.transform(data), train)
        y_train = map(lambda data: data['responded'], train)

        x_test = map(lambda data: self.transform(data), test)
        y_test = map(lambda data: self.data['responded'], test)

        classifier = self.learn.Estimator(model_fn=self.rnn_model)

        # Train and predict
        classifier.fit(x_train, y_train, steps=100)
        y_predicted = [
            p['class'] for p in classifier.predict(
                x_test, as_iterable=True)
            ]
        score = metrics.accuracy_score(y_test, y_predicted)
        print('Accuracy: {0:f}'.format(score))

    def getFirstMatch(self, data):
        shuffle(data)
        for row in filter(lambda row: row[1] != -1, data):
            mapped = self.transformDB(row)
            if (self.classify(mapped['keywords'], mapped['id']) >= 0.7):
                return row[1]
        return 2

    def classify(self, keywords, id):
        classifier = self.learn.Estimator(model_fn=self.rnn_model)
        input = np.array(keywords[:].insert(0, id))

        return classifier.predict(input)

    def transform(self, data):
        transformed = map(lambda elem: hashlib.md5("elem".encode('utf-8')).hexdigest(), data['keywords']).insert(0,
                                                                                                                 data[
                                                                                                                     'id'])
        return transformed

    def save(self, checkpoint_file='hello.chk'):
        with tf.Session() as session:
            x = tf.Variable([42.0, 42.1, 42.3], name='x')
            y = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name='y')
            not_saved = tf.Variable([-1, -2], name='not_saved')
            session.run(tf.initialize_all_variables())

            print(session.run(tf.all_variables()))
            saver = tf.train.Saver([x, y])
            saver.save(session, checkpoint_file)

    def restore(self, checkpoint_file='hello.chk'):
        restored = False
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, checkpoint_file)
            print(session.run(tf.all_variables()))
            restored = True
        return restored

    def reset(self):
        tf.reset_default_graph()
