import pandas as pd
import numpy as np
from model_keras_mlp import ModelKerasMLP
from model import ModelRunnerBase
import copy
from keras.utils import np_utils

class RunnerKerasMLPStack3(ModelRunnerBase):

    def build_model(self, run_fold_name):
        return ModelKerasMLP(run_fold_name)

    def load_xx(self, path):
        df = pd.read_csv(path, sep="\t")
        xx = df.values
        if self.prms["apply_ln"]:
            xx = self.apply_ln(xx)
        return xx

    def apply_ln(self, xx):
        xx = np.log(xx)
        xx = xx - np.mean(xx, axis=1).reshape(-1, 1)
        return xx

    def load_x_train(self):
        xs = []
        for run_name in self.flist:
            xs.append(self.load_xx("../model/pred/pred_values_{}_train.csv".format(run_name)))
        x = np.hstack(xs)
        return x

    def load_x_test(self):
        xs = []
        for run_name in self.flist:
            xs.append(self.load_xx("../model/pred/pred_values_{}_test.csv".format(run_name)))
        x = np.hstack(xs)
        return x


if __name__ == "__main__":
    params = {}
    params["optimizer"] = "adam"
    params["scaling"] = True
    params["apply_ln"] = True
    params["lr"] = 0.00015
    params["momentum"] = -1.000
    params["rho"] = -1
    params["beta_1"] = 0.94
    params["hidden_layers"] = 2
    params["h1"] = 250
    params["h2"] = 64
    params["h3"] = -1
    params["dropout1"] = 0.05
    params["dropout2"] = 0.05
    params["dropout3"] = 0.05
    params["weight_decay"] = None
    params["batch_size"] = 256
    params["nb_epoch"] = 200
    params["patience"] = 5
    params["seed"] = 71
    params["bags"] = 20

    params2 = {}
    params2["optimizer"] = "adam"
    params2["scaling"] = True
    params2["apply_ln"] = True
    params2["lr"] = 0.000141
    params2["momentum"] = -1.000
    params2["rho"] = -1
    params2["beta_1"] = 0.9858
    params2["hidden_layers"] = 2
    params2["h1"] = 376
    params2["h2"] = 50
    params2["h3"] = -1
    params2["dropout1"] = 0.0510
    params2["dropout2"] = 0.1483
    params2["dropout3"] = 0.05
    params2["weight_decay"] = 0.0017
    params2["batch_size"] = 256
    params2["nb_epoch"] = 225
    params2["patience"] = 99
    params2["seed"] = 71
    params2["bags"] = 20

    flist1 = ["mix_fin1_cnn_color_20", "mix_fin1_mix_vgg2_aug_10", "mix_fin1_resnet_20", "mix_fin1_resnet_aug_10",
              "mix_fin1_vgg2_10", "mix_fin1_greedy_cnns"]
    flist2 = ["mix_fin1_resnet_aug_10", "mix_fin1_greedy_cnns"]
    flist3 = ["mix_fin1_resnet_20", "mix_fin1_resnet_aug_10", "mix_fin1_vgg2_10", "mix_fin1_greedy_cnns"]

    runner = RunnerKerasMLPStack3("mix_fin1_stack1", flist=flist1, prms=params)
    runner.run_train()
    runner.run_test()

    runner = RunnerKerasMLPStack3("mix_fin1_stack2", flist=flist2, prms=params)
    runner.run_train()
    runner.run_test()

    runner = RunnerKerasMLPStack3("mix_fin1_stack3", flist=flist3, prms=params)
    runner.run_train()
    runner.run_test()

    # TODO params2?











