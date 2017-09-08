import tensorflow as tf

# Parameters
mnist_width = 28
n_visible = mnist_width * mnist_width
code_layer_size = 20
layers = [n_visible, 500, code_layer_size]

# Hyperparameters
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
        Y = tf.nn.sigmoid(tf.matmul(Y, W) + b)
    return Y

# Build the encoder
weights = build_weights(layers)
encoding = build_layers(weights, corrupted, 'Encoding Layer')

# Used the transpose weights to initialise decoder
decoding_weights = [tf.transpose(x) for x in weights[::-1]]
decoding = build_layers(decoding_weights, encoding, 'Decoding Layer')

# Cost is defined as error between original and reproduced
cost = tf.reduce_sum(tf.pow(decoding - original, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(alpha).minimize(cost)  # construct an optimizer

init = tf.global_variables_initializer()
