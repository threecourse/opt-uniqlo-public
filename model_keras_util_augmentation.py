import h5py
import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, flip_axis
import matplotlib.pyplot as plt

def preprocess_input(x, data_format=None):

    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
    return x

def preprocess_input_reverse(x, data_format=None):

    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
       raise Exception
    else:
        # Zero-center by mean pixel
        x[:, :, :, 0] += 103.939
        x[:, :, :, 1] += 116.779
        x[:, :, :, 2] += 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

def load_x_train():
    f = h5py.File("../model/feature/train_resized.hdf5", 'r')
    x = np.copy(f['feature'][:100])
    f.close()
    return x


class ImageAugmentor(object):

    def __init__(self, no_transform=0.25, seed=71):

        self.random = np.random.RandomState(seed)
        self.no_transform = no_transform

        data_format = K.image_data_format()
        self.rotation_range = 15
        self.width_shift_range = 0.05
        self.height_shift_range = 0.05
        self.zoom_range = [1.05, 1.15]
        self.fill_mode = "constant"
        self.cval = 255.0
        self.horizontal_flip = True
        self.vertical_flip = False

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('data_format should be "channels_last" (channel after row and '
                             'column) or "channels_first" (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

    def transform_batch_train_image(self, x):
        # prevent updating x
        x = x.copy()
        x0 = preprocess_input_reverse(x)
        ary = []
        for i in range(x0.shape[0]):
            ary.append(self.transform_train_image(x0[i]))
        x = preprocess_input(np.array(ary))
        return x

    def transform_batch_test_image(self, x, n=1):
        # prevent updating x
        x = x.copy()
        # returns: (batch_size, augment_images, image)
        # reverse preprocess
        x0 = preprocess_input_reverse(x)
        # print x0.shape
        ary = []
        for i in range(x0.shape[0]):
            xs = self.transform_test_images(x0[i], n=n)
            xs = preprocess_input(np.array(xs))
            ary.append(xs)
            # print xs.shape
        return np.array(ary)

    def transform_train_image(self, x):
        if self.random.rand() <= self.no_transform:
            # print self.random.rand()
            return x
        else:
            return self.random_transform(x)

    def transform_test_images(self, x, n=1):
        images = []
        images.append(x)
        for i in range(n-1):
            images.append(self.random_transform(x))
        return images

    def random_transform(self, x):

        x = x.copy()

        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * self.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = self.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = self.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = self.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.horizontal_flip:
            if self.random.rand() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if self.random.rand() < 0.5:
                x = flip_axis(x, img_row_axis)

        return x


if __name__ == "__main__":
    pass