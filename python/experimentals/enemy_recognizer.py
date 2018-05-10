import cv2 as cv
import glob
import time
import numpy as np
import random
import tensorflow as tf


class MyClassifierHelper():

    def __init__(self):
        filenames0 = glob.glob('../saved_res/enemy/*.jpg')
        filenames1 = glob.glob('../saved_res/nothing/*.jpg')
        self.inputs = [(name, 1) for name in filenames0] + [(name, 0) for name in filenames1]
        random.shuffle(self.inputs)

        self.data = []
        self.labels = []
        self.test = []
        self.test_label = []
        self.test_file = []
        self.index = 0
        pass

    def load_data(self):
        index = 0
        images = []
        img_lbl = []
        for files, label in self.inputs:
            image = cv.imread(files)
            image = cv.resize(image, (24, 48), interpolation=cv.INTER_CUBIC)
            images.append((image,files))
            if label == 0:
                img_lbl.append([1, 0])
            else:
                img_lbl.append([0, 1])
            index += 1
            if index % 500 == 0:
                print('Loading {} out of {} images'.format(index, len(self.inputs)))

        from sklearn.model_selection import train_test_split
        # self.data, self.test, self.labels, self.test_label = train_test_split(images, img_lbl, test_size=0.25)
        data, test, self.labels, self.test_label = train_test_split(images, img_lbl, test_size=0.25)
        for d in data:
            self.data.append(d[0])
        # self.data = self.data[0]
        for t in test:
            self.test.append(t[0])
            self.test_file.append(t[1])
        # self.test_file = self.test[1]
        # self.test = self.test[0]

        # print(np.shape(self.test))
        # print(np.shape(self.test_label))
        # print(np.shape(self.test_file))
        # print(self.test[0])
        # print(self.test_label[0])
        # print(self.test_file[0])

    def next_batch(self, batch_size):
        x = self.data[self.index:self.index + batch_size]
        y = self.labels[self.index:self.index + batch_size]
        self.index = (self.index + batch_size) % len(self.data)
        return x, y


helper = MyClassifierHelper()
helper.load_data()

X = tf.placeholder(tf.float32, shape=[None, 48, 24, 3])
y = tf.placeholder(tf.float32, shape=[None, 2])

hold_prob = tf.placeholder(tf.float32)


def init_weights(shape):
    rnd = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(rnd)


def init_bias(shape):
    rnd = tf.constant(0.1, shape=shape)
    return tf.Variable(rnd)


def conv2d(x, W):
    # x --> [batch, H, W, nChannels]
    # W --> [filterH, filterW, ChannelsIN, ChannelsOUT]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # x --> [batch, H, W, nChannels]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


conv1 = convolutional_layer(X, shape=[4, 4, 3, 24])
conv1pool = max_pool_2x2(conv1)

conv2 = convolutional_layer(conv1pool, shape=[4, 4, 24, 48])
conv2pool = max_pool_2x2(conv2)

conv2shape = tf.shape(conv2pool)
conv2flat = tf.reshape(conv2pool, [-1, 12*6 * 48])

fully_layer_one = tf.nn.relu(normal_full_layer(conv2flat, 768))
fully_one_dropout = tf.nn.dropout(fully_layer_one, keep_prob=hold_prob)
output = normal_full_layer(fully_one_dropout, 2)

cross_entorpy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entorpy)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # batch = helper.next_batch(30)
    # print(sess.run(conv2shape, feed_dict={X: batch[0]}))

    for i in range(10000):
        break #to not run this training again
        batch = helper.next_batch(30)
        # print(np.shape(batch[0]))
        # print(np.shape(batch[1]))

        sess.run(train, feed_dict={X: batch[0], y: batch[1], hold_prob: 0.5})

        if i % 100 == 0:
            print('STEP: {}'.format(i))

            matches = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={X: helper.test, y: helper.test_label, hold_prob: 1.0}))
            print('\n')
        if i%1000 == 0:
            save_path = saver.save(sess, "./enemy_or_not/model.ckpt")
            print("Model saved in path: %s" % save_path)

with tf.Session() as sess:
    saver.restore(sess, "../enemy_or_not/model.ckpt")

    matches = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    print(np.shape(helper.test))
    print(np.shape(helper.test_label))
    match = sess.run(matches, feed_dict={X: helper.test, y: helper.test_label, hold_prob: 1.0})
    print(match)
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    accuracy = sess.run(acc, feed_dict={X: helper.test, y: helper.test_label, hold_prob: 1.0})
    print(accuracy)
    print('\n')

for m,index in zip(match, range(len(match))):

    if not m:
        print('___misclassified___')
        print(helper.test_file[index])
        if helper.test_label[index][0] == 1:
            print('true = Not Enemy')
            print('predict = Enemy!!')
        else:
            print('true = Enemy')
            print('predict = Not Enemy!!')
        print(index, helper.test_label[index])
        print('\n')
        cv.imshow("MyImage", helper.test[index])
        cv.waitKey(5000)
        # print(helper.data[i], helper.labels[i])


# print(helper.data[0])
# print(helper.labels[0])
# cv.imshow("MyImage", helper.data[0])
# cv.waitKey(5000)
cv.destroyAllWindows()
