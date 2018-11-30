import tensorflow as tf
import numpy as np
import facial_data

CNN_PATH = "model/cnnmodel"
LINEAR_PATH = "model/linear"


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=5e-2)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


Weights = {
    # convolution layers
    'conv1': weight_variable([3, 3, 1, 128]),  # 5,5,32,32
    'conv2': weight_variable([3, 3, 128, 64]),  # 3,3,32,32
    'conv3': weight_variable([3, 3, 64, 32]),  #5,5,32,64
    # fully connected layers
    'fc1': weight_variable([6 * 6 * 32, 200]), #6*6*64,1024
    'out': weight_variable([200, 7]) #1024,7
}

Bias = {
    # convolution layers
    'conv1': bias_variable([128]), #32
    'conv2': bias_variable([64]), #32
    'conv3': bias_variable([32]), #32
    # fully connected layers
    'fc1': bias_variable([200]), #1024
    'out': bias_variable([7])
}


# cnn network
def cnn_net(x, weights, bias, keep_prob):
    x = tf.reshape(x, [-1, 48, 48, 1])

    # layer-1
    h_conv1 = tf.nn.relu(conv2d(x, weights['conv1']) + bias['conv1'])
    h_pool1 = max_pool(h_conv1, 2)

    # layer-2
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights['conv2']) + bias['conv2'])
    h_pool2 = max_pool(h_conv2, 2)

    # layer-3
    h_conv3 = tf.nn.relu(conv2d(h_pool2, weights['conv3']) + bias['conv3'])
    h_pool3 = max_pool(h_conv3, 2)

    # fully connected layer
    h_pool3_flat = tf.reshape(h_pool3, [-1, weights['fc1'].get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, weights['fc1']) + bias['fc1'])

    # drop out
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print("h_fc1_drop:", h_fc1_drop)
    # output layer
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, weights['out']) + bias['out'], name='probability')
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
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    return cross_entropy, train_step


def evaluate(pred, label):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    return correct_prediction, accuracy


def train_cnn():
    x_test, y_test, x_train, y_train = facial_data.read()

    x = tf.placeholder(tf.float32, [None, 48 * 48], name='data')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_ = tf.placeholder(tf.float32, [None, 7], name='label')
    y = cnn_net(x, Weights, Bias, keep_prob)
    cross_entropy, train_step = optimize(y, y_)
    correct_prediction, accuracy = evaluate(y, y_)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for epoch in range(0, 10):
        for i in range(600):
            batch_xs = x_train[i * 50: (i + 1) * 50]
            batch_ys = y_train[i * 50: (i + 1) * 50]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                emotions = sess.run(tf.argmax(y, 1), feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                print(emotions)
                print("step %d, loss %g, training accuracy %g" % (i, loss, train_accuracy))
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))

    model_saver = tf.train.Saver()
    model_saver.save(sess, CNN_PATH)
    print('CNN Model trained completed.')


def train_linear():
    x_test, y_test, x_train, y_train = facial_data.read()
    x = tf.placeholder(tf.float32, [None, 48 * 48], name='data')
    y_ = tf.placeholder(tf.float32, [None, 7], name='label')
    y = linear(x)
    cross_entropy, train_step = optimize(y, y_)
    correct_prediction, accuracy = evaluate(y, y_)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        batch_xs = x_train[i * 10: (i + 1) * 10]
        batch_ys = y_train[i * 10: (i + 1) * 10]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        emotions = sess.run(tf.argmax(y, 1), feed_dict={x: batch_xs, y_: batch_ys})
        print("emotions:", emotions)

    print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
    model_saver = tf.train.Saver()
    model_saver.save(sess, LINEAR_PATH)
    print('Linear model trained completed.')

'''
# use linear model to predict
def predict_linear(data):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(LINEAR_PATH + '.meta')
        loader.restore(sess, LINEAR_PATH)

        x = loaded_graph.get_tensor_by_name('data:0')
        y = loaded_graph.get_tensor_by_name('probability:0')
        y_ = loaded_graph.get_tensor_by_name('label:0')
        logit = sess.run(y, feed_dict={
            x: data, y_: np.zeros((8, 7))
        })
        result = sess.run(tf.argmax(logit, 1))

    return result

# use cnn model to predict 
def predict_cnn(data):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(LINEAR_PATH + '.meta')
        loader.restore(sess, CNN_PATH)

        x = loaded_graph.get_tensor_by_name('data:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        y = loaded_graph.get_tensor_by_name('probability:0')
        y_ = loaded_graph.get_tensor_by_name('label:0')
        logit = sess.run(y, feed_dict={
            x: data, y_: np.zeros((8, 7)), keep_prob: 1.0
        })
        emotions = sess.run(tf.argmax(logit, 1))

    return emotions
'''

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
                if i % 100 == 0:
                    predict, actual = sess.run([tf.argmax(y, 1), tf.argmax(y_, 1)],
                                               feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                    print("predict emotion:", predict)
                    print("actual emotion:", actual)
                    # Calculate loss and accuracy
                    loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs,
                                                                      y_: batch_ys, keep_prob: 1.0})

                    print("Epoch: " + str(epoch + 1) + ", Batch: " + str(i) + ", Loss= " + \
                          "{:.3f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

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


if __name__ == '__main__':
    train(1)
