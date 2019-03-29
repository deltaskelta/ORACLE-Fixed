"""
data loader to load the MNIST data
"""

import gzip
import sys
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=150)


def load_data():
    '''load the data form the mnist dataset'''
    img_f = gzip.open('./data/train-images-idx3-ubyte.gz', 'r')
    label_f = gzip.open('./data/train-labels-idx1-ubyte.gz', 'r')

    # images are stored in this dimension
    image_size = 28
    num_images = 6000

    img_f.read(16)  # skip non image data
    label_f.read(8)  # skip non label data

    # read image data into memory, and reshape into list of image pixels
    img_buf = img_f.read(image_size * image_size * num_images)
    img_data = np.frombuffer(img_buf, dtype=np.uint8).astype(np.uint8)
    img_data = img_data.reshape(num_images, image_size, image_size, 1)

    label_buf = label_f.read(1 * num_images)
    label_data = np.frombuffer(label_buf, dtype=np.uint8).astype(np.uint8)
    label_data = label_data.reshape(num_images, 1)

    cont = True
    i = 0
    while cont:
        print("labeled as {}".format(label_data[i]))
        image = np.asarray(img_data[i]).squeeze()

        print("pixels")
        print(image)

        # uncomment to plot the image
        # plt.imshow(image)
        # plt.show()
        cont = input(
            "press letter<enter> to see more examples, <enter> to continue: ")
        i += 1

    data = {i: [] for i in range(10)}
    for i in range(num_images):
        # since the labels are all scalars, we can just get the 0 index
        digit = label_data[i][0]
        data[digit].append(img_data[i])

    for i in data.keys():
        print(len(data[i]))

    see = input("see sorted examples (y/n): ")  # sanity check
    if see == 'y':
        for i in data.keys():
            for j in range(5):
                print("image is a {}".format(i))
                image = np.asarray(data[i][j]).squeeze()
                print(image)

    return data
