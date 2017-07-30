import pandas as pd
import numpy as np
from model_keras_resnet import ModelKerasResnet
from model import ModelRunnerBase
import copy
import h5py

class RunnerKerasResnet(ModelRunnerBase):

    def build_model(self, run_fold_name):
        return ModelKerasResnet(run_fold_name, self.num_classes)

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
    for e in [18, 19, 20, 21, 22]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00027, "dropout1":0.05, "dropout2":0.08,
                "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True}
        runner = RunnerKerasResnet("resnet_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()

    for e in [18, 19, 20, 21, 22]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00030, "dropout1":0.1, "dropout2":0.1,
                "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True}
        runner = RunnerKerasResnet("resnet_2_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()


    for e in [18, 19, 20, 21, 22]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00028, "dropout1":0.1, "dropout2":0.1,
                "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True}
        runner = RunnerKerasResnet("resnet_3_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()

    for e in [18, 19, 20, 21]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00029, "dropout1":0.1, "dropout2":0.1,
                "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True}
        runner = RunnerKerasResnet("resnet_4_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()

    """

    for e in [22]:
        prms = {"structure":"dense_dropout", "nb_epoch":e, "batch_size":32, "steps_per_epoch":3200/32,
                "lr":0.00029, "dropout1":0.1, "dropout2":0.1,
                "momentum":0.9, "patience":99, "decay":0.0, "nesterov":True}
        runner = RunnerKerasResnet("resnet_4_e{}".format(e), flist=None, prms=prms)
        runner.run_train()
        runner.run_test()