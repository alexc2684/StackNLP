from __future__ import print_function
import collections
import random
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

n_input = 3
n_hidden = 512
e = .001
steps = 50000

def lstm(x, weights, biases, n_input, n_hidden):
    x = tf.reshape(x, [1, n_input])
    x = tf.split(x, n_input, 1)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
    output, states =  rnn.static_rnn(rnn_cell, x, dtype='float32')
    return tf.matmul(output[-1], weights) + biases

def weight_variable(h, v):
    initial = tf.random_normal([h, v])
    return tf.Variable(initial)

def bias_variable(v):
    initial = tf.random_normal([v])
    return tf.Variable(initial)

def calc_accuracy(total, step):
    if total == 0:
        return 0
    return total*100/step

with open("data.pickle", "rb") as f:
    data = pickle.load(f)

with open("wti.pickle", "rb") as f:
    wti = pickle.load(f)

with open("itw.pickle", "rb") as f:
    itw = pickle.load(f)

vocab_size = len(wti)
x = tf.placeholder(tf.float32, shape=[None, n_input, 1])
y = tf.placeholder(tf.float32, shape=[None, vocab_size])

weights = weight_variable(n_hidden, vocab_size)
biases = bias_variable(vocab_size)

out = lstm(x, weights, biases, n_input, n_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
train_step = tf.train.RMSPropOptimizer(learning_rate=e).minimize(cost)
correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    offset = random.randint(0, n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    # writer.add_graph(sess.graph)
    while step < steps:
        # gen mini batch with randomness
        if offset > (len(data)-end_offset):
            offset = random.randint(0, n_input+1)

        #indices of train points
        train_indices = [[wti[str(data[i])]] for i in range(offset, offset + n_input)]
        train_indices = np.reshape(np.array(train_indices), [-1, n_input, 1])

        #one hot of actual label
        out_onehot = np.zeros([vocab_size], dtype='float32')
        out_onehot[wti[str(data[offset+n_input])]] = 1.0
        out_onehot = np.reshape(out_onehot, [1,-1])

        _, acc, loss, prediction = sess.run([train_step, accuracy, cost, out],
                                            feed_dict={x: train_indices, y: out_onehot})
        loss_total += loss
        acc_total += acc
        if step % 1000 == 0 and step > 0:
            print(step)
            print("Loss: ", loss_total/step)
            print("Accuracy: ", acc_total*100/step, "%")
        step += 1
        offset += n_input + 1

    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        words = sentence.strip().split(' ')

        if len(words) != n_input:
            continue
        try:
            train_indices = [wti[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(train_indices), [-1, n_input, 1])
                onehot_pred = sess.run(out, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,itw[onehot_pred_index])
                train_indices = train_indices[1:]
                train_indices.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")
