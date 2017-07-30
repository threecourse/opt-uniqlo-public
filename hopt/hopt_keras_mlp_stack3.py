import sys
sys.path += ["."]
import os

from util import Util
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import copy
from run_keras_mlp2 import RunnerKerasMLPStack3

class HoptRunner:

    def run(self, max_evals=50):
        self.fpath = self.get_fpath()
        self.max_evals = max_evals
        self.space = self.get_space()

        trials = Trials()
        self.scores = []
        Util.mkdir_file(self.fpath)
        with open(self.fpath, "w") as f:
            f.write("\t".join(["loss"] + self.get_prms_header_list()) + "\n")

        best = fmin(self.score, self.space, algo=tpe.suggest, trials=trials, max_evals=self.max_evals)
        print "hopt finished", self.fpath, best

    def score(self, params):
        print "Training with params : "
        print params
        prms = dict(params)
        loss, model = self.loss_func(prms)

        with open(self.fpath, "a") as f:
            f.write("\t".join([str(s) for s in [loss] + self.get_prms_values_list(prms, model)]) + "\n")
        return {'loss': loss, 'status': STATUS_OK}

    def get_fpath(self):
        raise NotImplementedError

    def get_prms_header_list(self):
        raise NotImplementedError

    def get_prms_values_list(self, prms):
        raise NotImplementedError

    def get_space(self):
        raise NotImplementedError

    def loss_func(self, prms):
        raise NotImplementedError


class HoptRunnerKerasMLPStack3(HoptRunner):

    def get_fpath(self):
        return "../model/hopt/hopt_keras_mlp_stack3.txt"

    def get_prms_header_list(self):
        return ["optimizer", "lr", "momentum", "rho", "beta_1",
                "hidden_layers", "h1", "h2", "h3", "dropout1", "dropout2", "dropout3", "weight_decay",
                "batch_size", "nb_epoch", "patience", "seed", "bags",
                "epochs"]

    def get_prms_values_list(self, prms, model):
        hists = model.hists
        epochs = np.mean([float(len(h.history["val_loss"])) for h in hists])

        info = copy.deepcopy(self.runner.prms)
        info["epochs"] = epochs
        return [info[c] for c in self.get_prms_header_list()]

    def get_space(self):

        space = {
            # optimizer
            "set_optimizer": hp.choice('set_optimizer',
                                   [
                                    {"optimizer": "adam",
                                     "lr": hp.loguniform("lr_adam", np.log(0.0001), np.log(0.01)),
                                     "momentum": -1,
                                     "rho": -1,
                                     "beta_1": hp.loguniform("beta_1", np.log(0.8), np.log(0.99)),
                                     },
                                   {"optimizer": "adadelta",
                                    "lr": hp.loguniform("lr_adadelta", np.log(0.1), np.log(2.0)),
                                    "momentum": -1,
                                    "rho": hp.loguniform("rho", np.log(0.9), np.log(0.99)),
                                    "beta_1": -1,
                                    },
                                   ]),
            "layers": hp.choice('layers',
                                       [
                                           {"hidden_layers": 2,
                                            "dropout3":-1,
                                            "h3": -1},
                                           {"hidden_layers": 3,
                                            "dropout3":  hp.choice("dropout3", [0.01, 0.1, 0.2]),
                                            "h3": hp.loguniform("h3", np.log(32), np.log(128))},
                                       ]),
            "weight_decay": hp.choice("weight_decay", [None, 0.0001, 0.001, 0.01, 0.1]),
            "h1": hp.loguniform("h1", np.log(125), np.log(500)),
            "h2": hp.loguniform("h2", np.log(32), np.log(128)),
            "dropout1": hp.choice("dropout1", [0.01, 0.1, 0.2]),
            "dropout2": hp.choice("dropout2", [0.01, 0.1, 0.2]),

        }
        return space

    params = {}

    def loss_func(self, prms):
        params = {}
        params["optimizer"] = prms["set_optimizer"]["optimizer"]
        params["lr"] = prms["set_optimizer"]["lr"]
        params["momentum"] = prms["set_optimizer"]["momentum"]
        params["rho"] = prms["set_optimizer"]["rho"]
        params["beta_1"] = prms["set_optimizer"]["beta_1"]

        params["h1"]  = int(prms["h1"])
        params["h2"]  = int(prms["h2"])

        params["dropout1"] = prms["dropout1"]
        params["dropout2"] = prms["dropout2"]

        params["hidden_layers"] = prms["layers"]["hidden_layers"]
        params["dropout3"] = prms["layers"]["dropout3"]
        params["h3"]  = int(prms["layers"]["h3"])
        params["weight_decay"] = prms["weight_decay"]

        params["batch_size"] = 256
        params["nb_epoch"] = 150
        params["patience"] = 5
        params['seed'] = 71
        params["steps_per_epoch"] = 3200 / params["batch_size"]
        params["scaling"] = True
        params["apply_ln"] = True

        params["bags"] = 1

        flist = ["mix_resnet_aug_10", "mix_cnn5_1_20", "mix_cnn5_2_20", "mix_resnet_20", "mix_vgg2_10",
                 "mix_cnn_color_20"]

        self.runner = RunnerKerasMLPStack3("keras_mlp_stack3_hopt", flist=flist, prms=params, Exp=False)
        loss, model = self.runner.run_hopt()
        return loss, model

hoptrunner = HoptRunnerKerasMLPStack3()
hoptrunner.run(max_evals=250)
