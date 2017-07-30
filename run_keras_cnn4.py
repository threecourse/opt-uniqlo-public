import pandas as pd
import numpy as np
from model_keras_cnn4 import ModelKerasCNN4
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

class RunnerKerasCNN4(ModelRunnerBase):

    def build_model(self, run_fold_name):
        return ModelKerasCNN4(run_fold_name)

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

    prms = {"nb_epoch":65,
            "batch_size":128, "patience":99,
            "optimizer":"adam", "lr":0.0015, "beta_1":0.865,
            "filters":3, "bn":2, "ch1":120, "ch2":110, "ch3":70, "d1":80, "d2":30,
            "dropout1":0.05, "dropout2":0.05, "dropout3":0.05}
    prms["steps_per_epoch"] = 3200 / prms["batch_size"]
    prms["bags"] = 5

    runner = RunnerKerasCNN4("keras_cnn4", flist=None, prms=prms)
    runner.run_train()
    runner.run_test()

    prms = {"nb_epoch":62,
            "batch_size":128, "patience":99,
            "optimizer":"adam", "lr":0.0016, "beta_1":0.865,
            "filters":3, "bn":2, "ch1":120, "ch2":110, "ch3":70, "d1":80, "d2":30,
            "dropout1":0.05, "dropout2":0.05, "dropout3":0.05}
    prms["steps_per_epoch"] = 3200 / prms["batch_size"]
    prms["bags"] = 5

    runner = RunnerKerasCNN4("keras_cnn4_2", flist=None, prms=prms)
    runner.run_train()
    runner.run_test()

    prms = {"nb_epoch": 65,
            "batch_size": 128, "patience": 99,
            "optimizer": "adam", "lr": 0.0015, "beta_1": 0.865,
            "filters": 3, "bn": 2, "ch1": 130, "ch2": 100, "ch3": 70, "d1": 80, "d2": 30,
            "dropout1": 0.05, "dropout2": 0.05, "dropout3": 0.05}
    prms["steps_per_epoch"] = 3200 / prms["batch_size"]
    prms["bags"] = 5

    runner = RunnerKerasCNN4("keras_cnn4_3", flist=None, prms=prms)
    runner.run_train()
    runner.run_test()

    prms = {"nb_epoch": 62,
            "batch_size": 128, "patience": 99,
            "optimizer": "adam", "lr": 0.0017, "beta_1": 0.865,
            "filters": 3, "bn": 2, "ch1": 130, "ch2": 100, "ch3": 70, "d1": 80, "d2": 30,
            "dropout1": 0.05, "dropout2": 0.05, "dropout3": 0.05}
    prms["steps_per_epoch"] = 3200 / prms["batch_size"]
    prms["bags"] = 5

    runner = RunnerKerasCNN4("keras_cnn4_4", flist=None, prms=prms)
    runner.run_train()
    runner.run_test()