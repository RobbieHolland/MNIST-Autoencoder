import tensorflow as tf
import tensorflow_autoencoder as model
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

# Paths
model_save_path = './trained_models/autoencoder_model'
mnist_data_path = './MNIST_data/'

# Parameters
corruption_level = 0.3

# Method to show results visually
def show_encoding(sess, i):
    digit = teX[i]
    images = np.array([digit])
    plt.imshow(digit.reshape((28, 28)), cmap='gray')
    plt.show()

    noise_mask = np.random.binomial(1, 1 - corruption_level, images.shape)

    plt.imshow((noise_mask * images).reshape((28, 28)), cmap='gray')
    plt.show()

    reconstructed = sess.run(model.decoding, feed_dict={model.original: images,
                                                        model.corrupted: noise_mask * images})

    plt.imshow(reconstructed.reshape((28, 28)), cmap='gray')
    plt.show()

# Evaluate model
with tf.Session() as sess:
    saver = tf.train.Saver()

    # tf.initialize_all_variables().run()
    saver.restore(sess, model_save_path)

    # Load MNIST data
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    # Evaluate model on noisy data
    noise_mask = np.random.binomial(1, 1 - corruption_level, teX.shape)
    print('Cost: ', sess.run(model.cost, feed_dict={model.original: teX, model.corrupted: noise_mask * teX}))

    # Test images used for examples in README
    indices = [6, 15]
    for x in indices:
        show_encoding(sess, x)
