import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_autoencoder as model

# Paths
model_save_path = './trained_models/autoencoder_model'
mnist_data_path = './MNIST_data/'

# Parameters
mnist_width = 28
corruption_level = 0.3
n_visible = mnist_width * mnist_width
code_layer_size = 20
layers = [n_visible, 500, code_layer_size]

# Hyperparameters
alpha = 0.01

# Load MNIST data
mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

saver = tf.train.Saver()

# Train on training data, every epoch evaluate with same evaluation data
def train_model(sess):
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            input_ = trX[start:end]
            noise_mask = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(model.train_op, feed_dict={model.original: input_, model.corrupted: noise_mask * input_})

        noise_mask = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print('Cost: ', sess.run(model.cost, feed_dict={model.original: teX, model.corrupted: noise_mask * teX}))

        saver.save(sess, model_save_path)

# Run training / viewing
with tf.Session() as sess:
    # tf.initialize_all_variables().run()
    saver.restore(sess, model_save_path)
    train_model(sess)
