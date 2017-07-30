import pandas as pd
import numpy as np
from model_keras_cnn5_2 import ModelKerasCNN5_2
from model import ModelRunnerBase
import copy
import h5py

from keras import backend as K

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

class RunnerKerasCNN5_2(ModelRunnerBase):

    def build_model(self, run_fold_name):
        return ModelKerasCNN5_2(run_fold_name)

    def load_x_train(self):
        f = h5py.File("../model/feature/train_64_raw.hdf5", 'r')
        x = np.copy(f['feature'])
        x2 = np.copy(f['feature2'])
        x2 = -1.0 * (x2 - 1)
        f.close()
        x = preprocess_input(x)
        return [x, x2]

    def load_x_test(self):
        f = h5py.File("../model/feature/test_64_raw.hdf5", 'r')
        x = np.copy(f['feature'])
        x2 = np.copy(f['feature2'])
        x2 = -1.0 * (x2 - 1)
        f.close()
        x = preprocess_input(x)

        return [x, x2]

if __name__ == "__main__":

    for e in [73, 74, 75, 76]:
        prms = {"nb_epoch":e,
                "batch_size":128, "patience":99,
                "optimizer":"adam", "lr":0.000125, "beta_1":0.915,
                "ch1":50, "ch2":200, "d1":250,
                "dropout1":0.1, "dropout2":0.1, "dropout3":0.2,
                "aug_seed": 271 + e, "test_time_augmentation": 4,
                "strides1":2, "strides2":2, "k1":1, "k1_2":2, "k2":2,
                "bn1":1, "bn2":1 }
        prms["steps_per_epoch"] = 3200 / prms["batch_size"]
        prms["bags"] = 5

        runner = RunnerKerasCNN5_2("keras_cnn5_2_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()
