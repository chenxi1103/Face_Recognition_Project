# -- Author: Jiayue Bao --
# -- Created Date: 2018/12/12
# -- VGG16 model for emotion detection
import tensorflow as tf
import numpy as np
import facial_data

CNN_PATH = "model/vgg16model"

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=5e-2)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


Weights = {
    # convolution layers
    'conv1_1': weight_variable([3, 3, 1, 64]),
    'conv1_2': weight_variable([3, 3, 64, 64]),
    'conv2_1': weight_variable([3, 3, 64, 128]),
    'conv2_2': weight_variable([3, 3, 128, 128]),
    'conv3_1': weight_variable([3, 3, 128, 256]),
    'conv3_2': weight_variable([3, 3, 256, 256]),
    'conv3_3': weight_variable([3, 3, 256, 256]),
    'conv4_1': weight_variable([3, 3, 256, 512]),
    'conv4_2': weight_variable([3, 3, 512, 512]),
    'conv4_3': weight_variable([3, 3, 512, 512]),
    'conv5_1': weight_variable([3, 3, 512, 512]),
    'conv5_2': weight_variable([3, 3, 512, 512]),
    'conv5_3': weight_variable([3, 3, 512, 512]),

    # fully connected layers
    'fc2': weight_variable([4096, 4096]),
    'fc3': weight_variable([4096, 1000]),
    'out': weight_variable([1000, 7])
}

Bias = {
    # convolution layers
    'conv1_1': bias_variable([64]),
    'conv1_2': bias_variable([64]),
    'conv2_1': bias_variable([128]),
    'conv2_2': bias_variable([128]),
    'conv3_1': bias_variable([256]),
    'conv3_2': bias_variable([256]),
    'conv3_3': bias_variable([256]),
    'conv4_1': bias_variable([512]),
    'conv4_2': bias_variable([512]),
    'conv4_3': bias_variable([512]),
    'conv5_1': bias_variable([512]),
    'conv5_2': bias_variable([512]),
    'conv5_3': bias_variable([512]),

    # fully connected layers
    'fc1': bias_variable([4096]),
    'fc2': bias_variable([4096]),
    'fc3': bias_variable([1000]),
    'out': bias_variable([7])
}


# cnn network
def cnn_net(x, weights, bias, keep_prob):
    x = tf.reshape(x, [-1, 48, 48, 1])

    # layer-1
    h_conv1_1 = tf.nn.relu(conv2d(x, weights['conv1_1']) + bias['conv1_1'])
    h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, weights['conv1_2']) + bias['conv1_2'])
    h_pool1 = max_pool(h_conv1_2, 2)

    # layer-2
    h_conv2_1 = tf.nn.relu(conv2d(h_pool1, weights['conv2_1']) + bias['conv2_1'])
    h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, weights['conv2_2']) + bias['conv2_2'])
    h_pool2 = max_pool(h_conv2_2, 2)

    # layer-3
    h_conv3_1 = tf.nn.relu(conv2d(h_pool2, weights['conv3_1']) + bias['conv3_1'])
    h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, weights['conv3_2']) + bias['conv3_2'])
    h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2, weights['conv3_3']) + bias['conv3_3'])
    h_pool3 = max_pool(h_conv3_3, 2)

    # layer-4
    h_conv4_1 = tf.nn.relu(conv2d(h_pool3, weights['conv4_1']) + bias['conv4_1'])
    h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, weights['conv4_2']) + bias['conv4_2'])
    h_conv4_3 = tf.nn.relu(conv2d(h_conv4_2, weights['conv4_3']) + bias['conv4_3'])
    h_pool4 = max_pool(h_conv4_3, 2)

    # layer-5
    h_conv5_1 = tf.nn.relu(conv2d(h_pool4, weights['conv5_1']) + bias['conv5_1'])
    h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, weights['conv5_2']) + bias['conv5_2'])
    h_conv5_3 = tf.nn.relu(conv2d(h_conv5_2, weights['conv5_3']) + bias['conv5_3'])
    h_pool5 = max_pool(h_conv5_3, 2)

    shape = int(np.prod(h_pool5.get_shape()[1:]))
    weights_fc1 = weight_variable([shape, 4096])

    # fully connected layer-1
    h_pool5_flat = tf.reshape(h_pool5, [-1, weights_fc1.get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, weights_fc1) + bias['fc1'])
    # drop out
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fully connected layer-2
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, weights['fc2']) + bias['fc2'])
    # drop out
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # fully connected layer-3
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, weights['fc3']) + bias['fc3'])
    # drop out
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    # output layer
    y_conv = tf.nn.softmax(tf.matmul(h_fc3_drop, weights['out']) + bias['out'], name='probability')
    print("y_conv:", y_conv)
    return y_conv


def linear(x):
    W = tf.Variable(tf.zeros([48 * 48, 7]))
    b = tf.Variable(tf.zeros([7]))
    pred = tf.nn.softmax(tf.matmul(x, W) + b, name='probability')
    return pred


def optimize(pred, label):
    # cross_entropy = -tf.reduce_sum(label * tf.log(pred))
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    return cross_entropy, train_step


def evaluate(pred, label):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    return correct_prediction, accuracy



# use linear model to predict
def predict_linear(data):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(CNN_PATH + '.meta')
        loader.restore(sess, CNN_PATH)

        x = loaded_graph.get_tensor_by_name('data:0')
        y = loaded_graph.get_tensor_by_name('probability:0')
        y_ = loaded_graph.get_tensor_by_name('label:0')
        logit = sess.run(y, feed_dict={
            x: data, y_: np.zeros((8, 7))
        })
        result = sess.run(tf.argmax(logit, 1))

    return result

def train(mode):
    x_test, y_test, x_train, y_train = facial_data.read()
    x = tf.placeholder(tf.float32, [None, 48 * 48], name='data')
    y_ = tf.placeholder(tf.float32, [None, 7], name='label')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    if mode == 0:
        y = linear(x)
    if mode == 1:
        y = cnn_net(x, Weights, Bias, keep_prob)

    cross_entropy, train_step = optimize(y, y_)
    correct_prediction, accuracy = evaluate(y, y_)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(0, 100):
            total_test_loss = 0
            total_test_acc = 0

            for i in range(600):
                batch_xs = x_train[i * 50: (i + 1) * 50]
                batch_ys = y_train[i * 50: (i + 1) * 50]
                # Run optimization op (backprop)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.6})

                if i % 10 == 0:
                    predict, actual = sess.run([tf.argmax(y, 1), tf.argmax(y_, 1)],
                                               feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                    #print("predict:", sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0}))
                    print("predict emotion:", predict)
                    print("actual emotion:", actual)
                    # Calculate loss and accuracy
                    loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs,
                                                                      y_: batch_ys, keep_prob: 1.0})

                    print("Epoch: " + str(epoch + 1) + ", Batch: " + str(i) + ", Loss= "
                          + "{:.3f}".format(loss) + ", Training Accuracy= "
                          + "{:.3f}".format(acc))

            # Calculate test loss and test accuracy
            for test_batch in range(0, 100):
                batch_xs = x_test[test_batch * 50: (test_batch + 1) * 50]
                batch_ys = y_test[test_batch * 50: (test_batch + 1) * 50]

                test_loss, test_acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs,
                                                                            y_: batch_ys, keep_prob: 1.0})
                total_test_loss += test_loss
                total_test_acc += test_acc

            total_test_acc = total_test_acc / 100
            total_test_loss = total_test_loss / 100

            print("Epoch: " + str(epoch + 1) + ", Test Loss= " + \
                  "{:.3f}".format(total_test_loss) + ", Test Accuracy= " + \
                  "{:.3f}".format(total_test_acc))
        model_saver = tf.train.Saver()
        model_saver.save(sess, CNN_PATH)
        print('Model trained completed.')


if __name__ == '__main__':
    train(1)
