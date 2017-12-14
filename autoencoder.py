""" 

Copyright (c) 2015, Aymeric Damien.
Copyright (c) 2017, Olav Markussen
Copyright (c) 2017, Aleksander Wasaznik

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import decay

# Training Parameters
initial_learning_rate = learning_rate = 0.9

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 4 # 2nd layer num features (the latent dim)
num_input = 80*80 # Pong data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 8))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
sess = tf.Session()

iteration = 0
def train_on(batch_x):
    """ Run optimization op (backprop) and cost op (to get loss value)
    """
    global learning_rate
    global iteration
    learning_rate = decay(initial_learning_rate, iteration)
    iteration += 1

    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
    print(f'Minibatch Loss: {l} lr {learning_rate}')

def latent_of(observation):
    """Return the latent representation according to the autoencoder
    """
    return sess.run(encoder_op, feed_dict={X: observation})

def make_data_dirs():
    try:
        os.mkdir('images')
        os.mkdir('images/reconstructed')
        os.mkdir('images/original')
    except Exception as e:
        print(e) 
        # Directory is already there
        pass

# Run the initializer
sess.run(init)
make_data_dirs()

def visualize_reconstruction(batch_x, learning_iteration, encoder_iteration):
    """Encode and decode observations from a game. Visualizes reconstructed and
    original for comparison
    """
    n = 1
    canvas_orig = np.empty((80 * n, 80 * n))
    canvas_recon = np.empty((80 * n, 80 * n))
    for i in range(n):
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 80:(i + 1) * 80, j * 80:(j + 1) * 80] = \
                batch_x[j].reshape([80, 80])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 80:(i + 1) * 80, j * 80:(j + 1) * 80] = \
                g[j].reshape([80, 80])

    plt.figure(figsize=(n, n))
    image_path = f"images/original/original_{learning_iteration:04d}_{encoder_iteration:02d}.png"
    plt.imsave(image_path, canvas_orig, origin="upper", cmap="Greys")

    plt.figure(figsize=(n, n))
    image_path = f"images/reconstructed/reconstructed_{learning_iteration:04d}_{encoder_iteration:07d}.png"
    plt.imsave(image_path, canvas_recon, origin="upper", cmap="Greys")
