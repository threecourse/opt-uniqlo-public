from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import h5py
import os

import skimage.transform
from skimage import io as skio

def transform1(path):
    x = image.load_img(path, target_size=(64, 64))
    x = image.img_to_array(x)
    return x

def transform2(path):
    resize = (64, 64)
    img = skio.imread(path)
    img = skimage.transform.resize(img, resize, mode='reflect') * 255
    mask = (img[:, :, 0] == 255) + (img[:, :, 1] == 255) + (img[:, :, 2] == 255)
    silhouette =  np.where(mask, 1, 0)
    return silhouette

def transform_batch(file_dir, file_names):
    xx = [os.path.join(file_dir, fname) for fname in file_names]
    xx = [transform1(x) for x in xx]
    xx = np.stack(xx)
    print xx.shape
    # xx = preprocess_input(xx)
    return xx

def transform_batch2(file_dir, file_names):
    xx = [os.path.join(file_dir, fname) for fname in file_names]
    xx = [transform2(x) for x in xx]
    xx = np.stack(xx)
    print xx.shape
    return xx

def transform_hdf5(out_path, file_dir, file_names):

    f = h5py.File(out_path, 'w')
    f.create_dataset('file_name', (0,), maxshape=(None,), dtype='|S54')
    f.create_dataset('feature', (0, 64, 64, 3), maxshape=(None, 64, 64, 3))
    f.create_dataset('feature2', (0, 64, 64), maxshape=(None, 64, 64))
    f.close()

    n = len(file_names)
    batch_size = 2000

    for i in range(0, n, batch_size):
        fnames = file_names[i: min(i + batch_size, n)]
        features = transform_batch(file_dir, fnames)
        features2 = transform_batch2(file_dir, fnames)

        print "{} processed".format(i)
        num_done = i + features.shape[0]

        f = h5py.File(out_path, 'r+')
        f['file_name'].resize((num_done,))
        f['file_name'][i: num_done] = np.array(fnames).astype(str)
        f['feature'].resize((num_done, 64, 64, 3))
        f['feature'][i:num_done, :] = features
        f['feature2'].resize((num_done, 64, 64))
        f['feature2'][i:num_done, :] = features2
        f.close()

        if num_done % 20000 == 0 or num_done == n:
            print "images processed: ", num_done

if __name__ == "__main__":

    Train = True
    Test = True

    if Train:
        df = pd.read_csv("../model/filenames_index_train.csv", sep="\t")
        transform_hdf5("../model/feature/train_64_raw.hdf5", "../data/train", df["file_name"])

    if Test:
        df = pd.read_csv("../model/filenames_index_test.csv", sep="\t")
        transform_hdf5("../model/feature/test_64_raw.hdf5", "../data/test", df["file_name"])
