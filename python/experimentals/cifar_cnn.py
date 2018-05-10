import time

CIFAR_DIR = 'cifar-10-batches-py/'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

all_data = [0, 1, 2, 3, 4, 5, 6]

for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR + direc)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

import matplotlib.pyplot as plt
import numpy as np

X = data_batch1[b'data']

all_images = X.reshape(10000, 3, 32, 32)


def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class CifarHelper():

    def __init__(self):
        self.i = 0

        # Grabs a list of all the data batches for training
        self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        # Grabs a list of all the test batches (really just one batch)
        self.test_batch = [test_batch]

        # Intialize some empty variables for later on
        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        print("Setting Up Training Images and Labels")

        # Vertically stacks the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        # Reshapes and normalizes training images
        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("Setting Up Test Images and Labels")

        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        # Reshapes and normalizes test images
        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size):
        # Note that the 100 dimension in the reshape call is set by an assumed batch size of 100
        x = self.training_images[self.i:self.i + batch_size].reshape(batch_size, 32, 32, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


# Before Your tf.Session run these two lines
ch = CifarHelper()
ch.set_up_images()

# During your session to grab the next batch use this line
# (Just like we did for mnist.train.next_batch)
# batch = ch.next_batch(100)

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
y_true = tf.placeholder(tf.float32, shape=[None,10])

hold_prob = tf.placeholder(tf.float32)

#HELPERS

def init_weights(shape):
    rnd = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(rnd)

def init_bias(shape):
    rnd = tf.constant(0.1,shape=shape)
    return tf.Variable(rnd)

def conv2d(x,W):
    #x --> [batch, H, W, nChannels]
    #W --> [filterH, filterW, ChannelsIN, ChannelsOUT]
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    #x --> [batch, H, W, nChannels]
    return tf.nn.max_pool(x, ksize=[1,2,2,1] , strides=[1,2,2,1] , padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

convo_1 = convolutional_layer(x,shape=[4,4,3,32])
convo_1_pooling = max_pool_2x2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling,shape=[4,4,32,64])
convo_2_pooling = max_pool_2x2(convo_2)

conv_2_flat = tf.reshape(convo_2_pooling, [-1,8*8*64])

full_layer_one = tf.nn.relu(normal_full_layer(conv_2_flat,1024))
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)
y_pred = normal_full_layer(full_layer_one,10)

cross_entorpy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entorpy)

init = tf.global_variables_initializer()

gpu = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu)) as sess:
# with tf.Session() as sess:
    sess.run(init)

    for i in range(5000):
        batch = ch.next_batch(50)

        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

        if i % 100 == 0:
            print('STEP: {}'.format(i))

            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={x: ch.test_images, y_true: ch.test_labels, hold_prob: 1.0}))
            print('\n')
