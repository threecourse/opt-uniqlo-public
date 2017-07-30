from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import h5py
import os

def transform_batch(file_dir, file_names):
    # preprocessed for vgg16, resnet50

    xx = [os.path.join(file_dir, fname) for fname in file_names]
    xx = [image.load_img(x, target_size=(224, 224)) for x in xx]
    xx = [image.img_to_array(x) for x in xx]
    xx = np.stack(xx)
    print xx.shape
    xx = preprocess_input(xx)
    print xx.shape
    return xx

def transform_hdf5(out_path, file_dir, file_names):

    f = h5py.File(out_path, 'w')
    f.create_dataset('file_name', (0,), maxshape=(None,), dtype='|S54')
    f.create_dataset('feature', (0, 224, 224, 3), maxshape=(None, 224, 224, 3))
    f.close()

    n = len(file_names)
    batch_size = 2000

    for i in range(0, n, batch_size):
        fnames = file_names[i: min(i + batch_size, n)]
        features = transform_batch(file_dir, fnames)

        print "{} processed".format(i)
        num_done = i + features.shape[0]

        f = h5py.File(out_path, 'r+')
        f['file_name'].resize((num_done,))
        f['file_name'][i: num_done] = np.array(fnames).astype(str)
        f['feature'].resize((num_done, 224, 224, 3))
        f['feature'][i:num_done, :] = features
        f.close()

        if num_done % 20000 == 0 or num_done == n:
            print "images processed: ", num_done

if __name__ == "__main__":

    Train = True
    Test = True

    if Train:
        df = pd.read_csv("../model/filenames_index_train.csv", sep="\t")
        transform_hdf5("../model/feature/train_resized.hdf5", "../data/train", df["file_name"])

    if Test:
        df = pd.read_csv("../model/filenames_index_test.csv", sep="\t")
        transform_hdf5("../model/feature/test_resized.hdf5", "../data/test", df["file_name"])
