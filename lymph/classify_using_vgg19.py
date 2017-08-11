import os
import pickle
from random import randint
from random import shuffle

import numpy as np
import tensorflow as tf

import lymph.utils.StackedAutoEncoder as SAE


def gen_batches(images: np.ndarray, labels: np.ndarray, examples_list: list):
    """

    :param images: image data
    :param labels: label data
    :param examples_list:
                [num_liver_examples: int, num_left_kidney_examples: int,
                num_right_kidney_examples: int, num_bladder_examples: int,
                num_lymphoma_examples: int, num_background_examples: int]
    :return: generated image and label batches
    """
    # 产生随机的index，这样既可以保证image和label对应，而且具有随机性
    random_index = list(range(images.shape[0]))
    shuffle(random_index)

    # 算是闭包函数吧
    def _get_batches_of_type(_num: int, feature_list: list):
        counter = 0
        images_to_be_generated, labels_to_be_generated = [], []
        for x in random_index:
            if counter == _num:
                break
            if (labels[x] == feature_list).all():
                images_to_be_generated.append(images[x])
                labels_to_be_generated.append(labels[x])
                counter += 1
        return np.stack(images_to_be_generated, axis=0), np.stack(labels_to_be_generated, axis=0)

    def _shuffle(images: np.ndarray, labels: np.ndarray):
        L = list(range(images.shape[0]))
        shuffle(L)
        _images, _labels = [], []
        for x in L:
            _images.append(images[x])
            _labels.append(labels[x])
        return np.stack(_images, axis=0), np.stack(_labels, axis=0)

    temp_images, temp_labels = [], []
    for i, num in enumerate(examples_list):
        if not num == 0:
            result = _get_batches_of_type(num, [1 if i == _ else 0 for _ in range(len(examples_list))])
            temp_images.append(result[0])
            temp_labels.append(result[1])

    return _shuffle(np.concatenate(temp_images, axis=0), np.concatenate(temp_labels, axis=0))


def show_num_each_class(labels: np.ndarray, msg: str) -> None:
    _num_classes = labels.shape[1]  # 动态判定到底有多少种分类
    type_list = [0 for _ in range(_num_classes)]
    labels_argmax = np.argmax(labels, axis=1)
    for i in range(labels.shape[0]):
        type_list[labels_argmax[i]] += 1
    print("[DEBUG] {}: {}".format(msg, type_list))


def train(train_folder: str, test_folder: str):
    # 加载训练集文件
    images = pickle.load(open(os.path.join(train_folder, "images.pydump"), 'rb'))
    labels = pickle.load(open(os.path.join(train_folder, "labels.pydump"), 'rb'))
    show_num_each_class(labels, "训练集输入构成")
    # 生成训练集
    train_images, train_labels = gen_batches(images, labels, [574, 123, 184, 81, 898, 898])
    show_num_each_class(train_labels, "生成训练集构成")
    # 加载测试集文件
    images = pickle.load(open(os.path.join(test_folder, "images.pydump"), 'rb'))
    labels = pickle.load(open(os.path.join(test_folder, "labels.pydump"), 'rb'))
    show_num_each_class(labels, "测试集构成")
    # 生成测试集
    test_images, test_labels = gen_batches(images, labels, [0, 0, 0, 0, -1, 0])
    show_num_each_class(test_labels, "生成测试集构成")
    # 构建稀疏自编码器
    sas = SAE.train_stacked_autoencoders(in_data=train_images, stack_shape=[4096, 512, 64, 6])

    # 构建网络
    x = tf.placeholder(dtype=tf.float32, shape=[None, 4096])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 6])
    # 全舰装填权值
    w1 = tf.Variable(sas[0].getWeights(), dtype=tf.float32)
    b1 = tf.Variable(sas[0].getBiases(), dtype=tf.float32)
    w2 = tf.Variable(sas[1].getWeights(), dtype=tf.float32)
    b2 = tf.Variable(sas[1].getBiases(), dtype=tf.float32)
    w3 = tf.Variable(sas[2].getWeights(), dtype=tf.float32)
    b3 = tf.Variable(sas[2].getBiases(), dtype=tf.float32)
    # 关闭session节省内存
    [_.close_session() for _ in sas]

    h1 = tf.nn.softplus(tf.matmul(x, w1) + b1)
    h2 = tf.nn.softplus(tf.matmul(h1, w2) + b2)
    y = tf.nn.softplus(tf.matmul(h2, w3) + b3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 20
        for i in range(1000):
            r_index = randint(0, train_images.shape[0] - batch_size)
            batch_xs = train_images[r_index: r_index + batch_size]
            batch_ys = train_labels[r_index: r_index + batch_size]
            if i % 10 == 0:
                train_loss, train_accuracy, train_y, train_y_ = sess.run([loss, accuracy, y, y_],
                                                                         feed_dict={x: batch_xs, y_: batch_ys})
                print("[DEBUG] i=%d, loss=%.3f, accuracy=%.3f" % (i, train_loss, train_accuracy), end=" ")
                test_loss, test_accuracy, test_y, test_y_ = sess.run([loss, accuracy, y, y_],
                                                                     feed_dict={x: test_images, y_: test_labels})
                print("loss=%.3f, accuracy=%.3f" % (test_loss, test_accuracy))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


if __name__ == '__main__':
    train(train_folder=r"C:\LYMPH_data\mat224dataUintPT06548", test_folder=r"C:\LYMPH_data\mat224dataUintPT38904")
