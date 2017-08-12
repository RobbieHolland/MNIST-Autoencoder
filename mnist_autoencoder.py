import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Hyperparameters
mnist_width = 28
corruption_level = 0.2
n_visible = mnist_width * mnist_width
code_layer_size = 30
layers = [n_visible, 400, code_layer_size]
alpha = 0.01

# Input placeholders for original image and its corrupted version
original = tf.placeholder("float", [None, n_visible])
corrupted = tf.placeholder("float", [None, n_visible])

# Initial weights need to be built and saved so that their transpose may be
# used for the reflected part of the network
def build_weights(layers):
    weights = []
    for i in range(len(layers) - 1):
        weights_shape = [layers[i], layers[i+1]]
        W_init = tf.random_uniform(shape=weights_shape,
                                   minval=-0.5,
                                   maxval=0.5)
        weights += [W_init]
    return weights

def build_layers(weights, Y, name):
    for i in range(len(weights)):
        W_init = weights[i]

        W = tf.Variable(W_init)
        b = tf.Variable(tf.zeros([weights[i].shape[1].value]))
        Y = tf.nn.sigmoid(tf.matmul(Y, W) + b)  # hidden state
    return Y

# Build the encoder
weights = build_weights(layers)
encoding = build_layers(weights, corrupted, 'Encoding Layer')

# Used the transpose weights to initialise decoder
# Technique from: Geoffery Hinton
decoding_weights = [tf.transpose(x) for x in weights[::-1]]
decoding = build_layers(decoding_weights, encoding, 'Decoding Layer')

# Cost is defined as error between original and reproduced
cost = tf.reduce_sum(tf.pow(decoding - original, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(alpha).minimize(cost)  # construct an optimizer

init = tf.global_variables_initializer()

# Load MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

saver = tf.train.Saver()

# Train on training data, every epoch evaluate with same evaluation data
def train_eval(sess):
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            input_ = trX[start:end]
            noise_mask = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(train_op, feed_dict={original: input_, corrupted: noise_mask * input_})

        noise_mask = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print(i, sess.run(cost, feed_dict={original: teX, corrupted: noise_mask * teX}))

        saver.save(sess, './autoencoder_model')

# Method to show results visually
def show_encoding(sess, i):
    digit = teX[i]
    images = np.array([digit])
    plt.imshow(digit.reshape((28, 28)), cmap='gray')
    plt.show()

    noise_mask = np.random.binomial(1, 1 - corruption_level, images.shape)

    plt.imshow((noise_mask * images).reshape((28, 28)), cmap='gray')
    plt.show()

    # print(img.shape)
    # print(mask_np.shape)
    # print((mask_np * img).shape)
    reconstructed = sess.run(decoding, feed_dict={original: images, corrupted: mask_np * images})
    # plt.imshow(digit.reshape((28, 28)), cmap='gray')

    plt.imshow(reconstructed.reshape((28, 28)), cmap='gray')
    plt.show()

# Run training / viewing
with tf.Session() as sess:
    # tf.initialize_all_variables().run()
    saver.restore(sess, './autoencoder_model')
    train_eval(sess)
    # for x in range(20):
    #     show_encoding(sess, x)
