import random

import numpy as np
import scipy.io
import math


# Returns 10000 image patches for training
# Each column contains grayscale value for the image
# Squash data to [0.1, 0.9]
def normalize_data(images):
    # Subtract mean of each image from its individual values
    mean = images.mean(axis=0)
    images = images - mean

    # Truncate to +/- 3 standard deviations and scale to -1 and +1
    pstd = 3 * images.std()
    images = np.maximum(np.minimum(images, pstd), -pstd) / pstd

    # Rescale from [-1,+1] to [0.1,0.9]
    images = (1 + images) * 0.4 + 0.1

    return images


# Returns 10000 patches for training
#  IMAGES is a 3D array containing 10 images
#  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
#  (The contrast on these images look a bit off because they have
#  been preprocessed using using "whitening."  See the lecture notes for
#  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
#  patch corresponding to the pixels in the block (21,21) to (30,30) of
#  Image 1
def sample_images():
    patch_size = 8
    num_patches = 10000
    num_images = 10
    image_size = 512

    image_data = scipy.io.loadmat('data/IMAGES.mat')['IMAGES']

    # Initialize patches with zeros.
    patches = np.zeros(shape=(patch_size * patch_size, num_patches))

    for i in range(num_patches):
        image_id = random.randint(0, num_images - 1)
        image_x = random.randint(0, image_size - patch_size)
        image_y = random.randint(0, image_size - patch_size)

        img = image_data[:, :, image_id]
        patch = img[image_x:image_x + patch_size, image_y:image_y + patch_size].reshape(patch_size * patch_size)
        patches[:, i] = patch

    return normalize_data(patches)


# sampleIMAGESRAW
# Returns 10000 "raw" unwhitened  patches
def sample_images_raw():
    image_data = scipy.io.loadmat('data/IMAGES_RAW.mat')['IMAGESr']

    patch_size = 12
    num_patches = 10000
    num_images = image_data.shape[2]
    image_size = image_data.shape[0]

    patches = np.zeros(shape=(patch_size * patch_size, num_patches))

    for i in range(num_patches):
        image_id = random.randint(0, num_images - 1)
        image_x = random.randint(0, image_size - patch_size)
        image_y = random.randint(0, image_size - patch_size)

        img = image_data[:, :, image_id]
        patch = img[image_x:image_x + patch_size, image_y:image_y + patch_size].reshape(patch_size * patch_size)
        patches[:, i] = patch

    return patches


def sample_MNIST_images(data: np.ndarray, patch_size=8, num_patches=10000):
    num_images = data.shape[1]
    image_size = math.floor(math.sqrt(data.shape[0]))
    patches = np.empty(shape=[patch_size * patch_size, num_patches], dtype=np.float64)
    for i in range(num_patches):
        img_id = random.choice(range(0, num_images))
        img_x = random.choice(range(0, image_size - patch_size))
        img_y = random.choice(range(0, image_size - patch_size))
        img = data[:, img_id].reshape([image_size, image_size])
        patch = img[img_x: img_x + patch_size, img_y: img_y + patch_size].reshape([patch_size ** 2])
        patches[:, i] = patch
    return patches

if __name__ == '__main__':
    from used.load_MNIST import load_MNIST_images
    from PIL import Image
    data = load_MNIST_images(r'D:\MNIST\train-images.idx3-ubyte')

    data2 = sample_MNIST_images(data)
    img = (data2[:, 0] * 255).astype(np.uint8).reshape([8, 8])
    Image.fromarray(img, "L").show()
    print(data2.shape)
    print(data2.dtype)