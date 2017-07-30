import pandas as pd
import numpy as np
from model_keras_resnet_augmentation import ModelKerasResnetAugmentation
from model import ModelRunnerBase
import copy
import h5py

class RunnerKerasResnetAugmentation(ModelRunnerBase):

    def build_model(self, run_fold_name):
        return ModelKerasResnetAugmentation(run_fold_name, self.num_classes)

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

    for e in [25, 26, 27, 28]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00027, "dropout1":0.05, "dropout2":0.08,
                "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True,
                "aug_seed":71+e, "test_time_augmentation":4}
        runner = RunnerKerasResnetAugmentation("resnet_aug_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()

    for e in [25, 26, 27, 28, 29]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00030, "dropout1":0.05, "dropout2":0.08,
                "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True,
                "aug_seed":171+e, "test_time_augmentation":4}
        runner = RunnerKerasResnetAugmentation("resnet_aug_2_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()
    """

    for e in [29]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00027, "dropout1":0.05, "dropout2":0.08,
                "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True,
                "aug_seed":71+e, "test_time_augmentation":4}
        runner = RunnerKerasResnetAugmentation("resnet_aug_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()

