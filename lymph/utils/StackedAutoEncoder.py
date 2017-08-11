import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from lymph.utils.autoencoder import Autoencoder


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]


def train_stacked_autoencoders(in_data, stack_shape):
    """
    训练堆栈式自编码器
    :param in_data: 输入数据，形状应为[None, n],其中n为一个正整数
    :param stack_shape: 堆栈式自编码器中各个层大小, 例如:
        输入为784（MNIST手写字符集的每个图像为28×28=784），第一个自编码器隐层有300个单元，第二个有100个单元的参数设置：
            stack_shape = [784, 300, 100]
    :return: 一个list，其中中包含了堆栈式自编码器中的每个实例
    """
    # 定义训练参数
    training_epochs = 200
    batch_size = 128
    # 初始化stacked autoencoders列表
    stacked_autoencoders = [Autoencoder(stack_shape[i], stack_shape[i + 1]) for i in range(len(stack_shape) - 1)]
    # 逐层训练自编码器（贪婪）
    for i in range(len(stack_shape) - 1):
        train_data = in_data if i == 0 else stacked_autoencoders[i - 1].transform(train_data)
        for epoch in range(training_epochs):
            for k in range(train_data.shape[0] // batch_size):
                batch_xs = get_random_block_from_data(train_data, batch_size)
                stacked_autoencoders[i].partial_fit(batch_xs)
            if epoch % 10 == 0:  # 显示epoch信息
                test_cost = stacked_autoencoders[i].calc_total_cost(train_data)
                print("[Encoder_{}] Epoch_{}, cost={}".format(i, epoch, test_cost))
    return stacked_autoencoders


def train_MNIST(stacked_autoecoders: list):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    w1 = tf.Variable(stacked_autoecoders[0].getWeights(), dtype=tf.float32)
    b1 = tf.Variable(stacked_autoecoders[0].getBiases(), dtype=tf.float32)
    w2 = tf.Variable(stacked_autoecoders[1].getWeights(), dtype=tf.float32)
    b2 = tf.Variable(stacked_autoecoders[1].getBiases(), dtype=tf.float32)
    w3 = tf.Variable(tf.zeros([100, 10]), dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([10]), dtype=tf.float32)
    [encoder.close_session() for encoder in stacked_autoecoders]

    h1 = tf.nn.softplus(tf.matmul(x, w1) + b1)
    h2 = tf.nn.softplus(tf.matmul(h1, w2) + b2)
    y = tf.nn.softmax(tf.matmul(h2, w3) + b3)  # softmax layer
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
    train_step = tf.train.AdamOptimizer().minimize(loss)
    prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            train_loss, _ = sess.run([loss, train_step], feed_dict={x: batch_xs, y_: batch_ys})
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
                print("[step %d] train_loss=%.3f, train_accuracy=%.3f" % (i, train_loss, train_accuracy), end=" ")
                test_accuracy, test_loss = sess.run([accuracy, loss],
                                                    feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print("|| test_loss=%.3f, test_accuracy=%.3f" % (i, test_loss, test_accuracy))


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    sas = train_stacked_autoencoders(in_data=mnist.train.images, stack_shape=[784, 300, 100])
    train_MNIST(sas)
