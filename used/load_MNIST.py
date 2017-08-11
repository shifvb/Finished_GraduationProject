import numpy as np


def load_MNIST_images(filename):
    """
    returns a 28x28x[number of MNIST images] matrix containing
    the raw MNIST images
    :param filename: input data file
    """
    with open(filename, "rb") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        assert magic == 2051
        num_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        num_rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        num_cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((num_images, num_rows * num_cols)).transpose()
        images = images.astype(np.float64) / 255
        return images


def load_MNIST_labels(filename):
    """
    returns a [number of MNIST images]x1 matrix containing
    the labels for the MNIST images
    :param filename: input file with labels
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        assert magic == 2049
        num_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        labels = np.fromfile(f, dtype=np.ubyte)
        return labels


if __name__ == '__main__':
    from PIL import Image
    import random

    arr = load_MNIST_images(r'D:\MNIST\train-images.idx3-ubyte')
    arr = arr.transpose([1, 0]).reshape([60000, 28, 28])
    arr *= 255
    arr = arr.astype(np.uint8)
    arr2 = load_MNIST_labels(r'D:\MNIST\train-labels.idx1-ubyte')
    for i in range(1):
        index = random.choice(range(60000))
        Image.fromarray(arr[index], 'L').show()
        print(arr2[index])
