import pandas as pd
import numpy as np
from model_keras_vgg2_augmentation import ModelKerasVGG2Augmentation
from model import ModelRunnerBase
import copy
import h5py

class RunnerKerasVGG2Augmentation(ModelRunnerBase):

    def build_model(self, run_fold_name):
        return ModelKerasVGG2Augmentation(run_fold_name)

    def load_x_train(self):
        f = h5py.File("../model/feature/train_resized.hdf5", 'r')
        x = np.copy(f['feature'])
        f.close()
        return x

    def load_x_test(self):
        f = h5py.File("../model/feature/test_resized.hdf5", 'r')
        x = np.copy(f['feature'])
        f.close()
        return x

if __name__ == "__main__":

    """
    for e in [1]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00007, "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True,
                "aug_seed": 171 + e, "test_time_augmentation": 2}
        runner = RunnerKerasVGG2Augmentation("vgg2_aug_e{}_temp".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()

    prms = {"structure":"dense_dropout", "nb_epoch":100, "batch_size":32, "steps_per_epoch":3200/32,
            "lr":0.00007, "momentum":0.9, "patience":4, "decay":0.0, "nesterov":True,
            "aug_seed": 1712, "test_time_augmentation": 2}
    runner = RunnerKerasVGG2Augmentation("vgg2_aug_temp_checkepoch", flist=None, prms=prms)
    runner.run_train()
    runner.run_test()
    """
    for e in [24, 25, 26, 27, 28]:
        prms = {"structure": "dense_dropout", "nb_epoch": e, "batch_size": 32, "steps_per_epoch": 3200 / 32,
                "lr": 0.00007, "momentum": 0.9, "patience": 99, "decay": 0.0, "nesterov": True,
                "aug_seed": 2712 + e, "test_time_augmentation": 4}
        runner = RunnerKerasVGG2Augmentation("vgg2_aug_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()

    for e in [24, 25, 26, 27, 28]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00008, "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True,
                "aug_seed": 1712 + e, "test_time_augmentation": 4}
        runner = RunnerKerasVGG2Augmentation("vgg2_aug_2_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()
