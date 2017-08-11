import numpy as np
from used.load_MNIST import load_MNIST_images
from PIL import Image
from random import choice
from used import load_MNIST
from used.train import train
from used.sparse_autoencoder import sparse_autoencoder
import pickle
from used.sample_images import sample_MNIST_images


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def analyze_theta():
    # set/get visible_size, hidden_size
    global visible_size
    global hidden_size

    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.
    data_theta = pickle.load(open("test.pydump", 'rb'))
    W1 = data_theta[0: hidden_size * visible_size].reshape(hidden_size, visible_size)
    b1 = data_theta[2 * hidden_size * visible_size: 2 * hidden_size * visible_size + hidden_size]
    W2 = data_theta[hidden_size * visible_size: 2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
    b2 = data_theta[2 * hidden_size * visible_size + hidden_size:]

    # data
    data = load_MNIST_images(r'D:\MNIST\train-images.idx3-ubyte')
    data = sample_MNIST_images(data)

    # Number of training examples
    m = data.shape[1]

    # z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    # a2 = sigmoid(z2)
    a2 = sparse_autoencoder(data_theta, hidden_size=hidden_size, visible_size=visible_size, data=data)

    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    h = sigmoid(z3)

    from used.display_network import display_network
    display_network(W1.transpose(), filename="weights.jpg", opt_normalize=False)
    display_network(W2, filename="weights2.jpg", opt_normalize=False)

    # arr = (h * 255).astype(np.uint8).transpose()
    # for i in range(1):
    #     index = choice(range(0, 60000))
    #     print(index)
    #     img = arr[index].reshape(28, 28).astype(np.uint8)
    #     prev_img = (data[:, index].reshape([28, 28]) * 255).astype(np.uint8)
    #     prev_img_ = Image.fromarray(prev_img, 'L')
    #     prev_img_.show()
    #     # prev_img_.save("prev_img_.png")
    #     img_ = Image.fromarray(img, 'L')
    #     img_.show()
    #     # img_.save("img_.png")


def main():
    # set/get visible_size, hidden_size
    global visible_size
    global hidden_size

    # get input data & save input data
    data = load_MNIST.load_MNIST_images(r'D:\MNIST\train-images.idx3-ubyte')
    data = sample_MNIST_images(data)
    pickle.dump(data, open("input_data.pydump", 'wb'))

    # train sparse autoencoder & save weights/bias
    the_output_theta = train(input_data=data, visible_size=visible_size, hidden_size=hidden_size)
    pickle.dump(the_output_theta, open("test.pydump", 'wb'))

    # get sparse autoencode output(the value of hidden units)
    # hidden_output = sparse_autoencoder(the_output_theta, hidden_size=hidden_size, visible_size=visible_size, data=data)


if __name__ == '__main__':
    # set global arguments
    visible_size = 8 * 8
    hidden_size = 36

    main()
    # analyze_theta()
