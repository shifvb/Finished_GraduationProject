import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import random


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)


def KL_divergence(x, y):
    return x * tf.log(x / y) + (1 - x) * tf.log((1 - x) / (1 - y))


class Autoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                 lambda_=3e-3, beta=3, sparsity_param=0.1):
        """
        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数，默认为softplus
        :param optimizer: 优化器，默认为adam
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.weights = self._initialize_weights()
        self.lambda_ = lambda_
        self.beta = beta
        self.sparsity_param = sparsity_param

        # 定义网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = self.transfer(tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2']))

        # 定义自编码器的损失函数
        self.cost = (tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))) + \
                    (self.lambda_ / 2) * (
                        tf.reduce_sum(tf.square(self.weights['w1'])) + tf.reduce_sum(tf.square(self.weights['w2'])))

        self.optimizer = optimizer.minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
        all_weights['w2'] = tf.Variable(xavier_init(self.n_hidden, self.n_input))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        """
        执行一步训练
        :param X: 一个batch数据
        :return: 当前损失cost
        """
        cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        """
        让Session执行一个计算图节点self.cost，传入的参数和函数partial_fit一致
        *用于自编码器训练完毕后，在测试集上对模型性能进行评测
        *不会像partial_fit那样触发训练操作
        :param X: 测试用batch数据
        :return: 损失cost
        """
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X):
        """
        返回自编码器隐含层的输出结果
        :param X: 输入数据
        :return: 隐含层输出
        """
        return self.sess.run(self.hidden, feed_dict={
            self.x: X,
        })

    def generate(self, hidden=None):
        """
        将隐含层的结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据。
        *这个接口和前面的transform正好将整个自编码器拆分为两部分，这里的generate接口是后半部分
            （即将高阶特征特征复原为原始数据的步骤）
        :param hidden: 隐含层的输出结果
        :return: 复原后的原始数据
        """
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        """
        整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据，即包括transform和generate两块。
        输入数据是原数据，输出数据是复原后的数据
        :param X: 原数据
        :return: 复原后的数据
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        """
        获取隐含层的权重w1
        :return: 隐含层权重w1
        """
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        """
        获取隐含层的偏置系数b1
        :return: 偏置系数b1
        """
        return self.sess.run(self.weights['b1'])

    def close_session(self):
        """wrapper function to close the object's internal TensorFlow session"""
        self.sess.close()

    def __del__(self):
        self.close_session()


if __name__ == '__main__':
    mnist = input_data.read_data_sets(r"C:\Users\anonymous\PycharmProjects\ufldl_tutorial\tf_version\MNIST_data",
                                      one_hot=True)

    # 定义参数
    traning_epochs = 20
    batch_size = 128
    auto_encoder = Autoencoder(n_input=784, n_hidden=196, transfer_function=tf.nn.softplus)

    # 开始训练过程
    for epoch in range(traning_epochs):
        for i in range(int(mnist.train.images.shape[0] / batch_size)):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            cost = auto_encoder.partial_fit(batch_xs)
        if epoch % 1 == 0:
            print("Epoch:{}".format(epoch), "cost={}".format(cost))

    # 对训练完的模型进行性能测试
    print("Total cost: " + str(auto_encoder.calc_total_cost(mnist.test.images)))

    # 结果可视化
    data = auto_encoder.reconstruct(mnist.test.images)
    index = random.randint(0, 10000)
    pre_image_data = (mnist.test.images[index] * 255).reshape([28, 28]).astype(np.uint8)
    Image.fromarray(pre_image_data, "L").show()
    data = (data[index] * 255).reshape([28, 28]).astype(np.uint8)
    print(pre_image_data.astype(np.int16) - data.astype(np.int16))
    Image.fromarray(data, "L").show()
