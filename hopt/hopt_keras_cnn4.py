import sys
sys.path += ["."]
import os

from util import Util
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import copy
from run_keras_cnn4 import RunnerKerasCNN4

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


class HoptRunnerKerasCNN(HoptRunner):

    def get_fpath(self):
        return "../model/hopt/hopt_keras_cnn.txt"

    def get_prms_header_list(self):
        return ["optimizer", "lr", "momentum", "rho", "beta_1",
                "filters", "ch1", "ch2", "ch3", "d1", "d2", "bn",
                "dropout1", "dropout2", "dropout3",
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
                                   [{"optimizer":"sgd",
                                     "lr": hp.loguniform("lr_sgd", np.log(0.001), np.log(0.05)),
                                     "momentum": hp.loguniform("momentum", np.log(0.8), np.log(0.99)),
                                     "rho": -1,
                                     "beta_1": -1,
                                     },
                                    {"optimizer": "adadelta",
                                     "lr": hp.loguniform("lr_adadelta", np.log(0.1), np.log(2.0)),
                                     "momentum": -1,
                                     "rho":  hp.loguniform("rho", np.log(0.9), np.log(0.99)),
                                     "beta_1": -1,
                                     },
                                    {"optimizer": "adam",
                                     "lr": hp.loguniform("lr_adam", np.log(0.0001), np.log(0.01)),
                                     "momentum": -1,
                                     "rho": -1,
                                     "beta_1": hp.loguniform("beta_1", np.log(0.8), np.log(0.99)),
                                     }
                                    ]),

            # layer
            "set_filters": hp.choice('set_filters',
                                       [{'filters':2, "ch3": -1},
                                        {'filters':3, "ch3": hp.loguniform("ch3", np.log(50), np.log(250))}
                                       ]),
            "ch1": hp.loguniform("ch1", np.log(50), np.log(250)),
            "ch2": hp.loguniform("ch2", np.log(50), np.log(250)),
            "d1": hp.loguniform("d1", np.log(50), np.log(250)),
            "d2": hp.loguniform("d2", np.log(25), np.log(150)),

            # others
            "dropout": hp.loguniform("dropout", np.log(0.01), np.log(0.3)),
            "bn": hp.choice("bn", [0,1,2]),  # 0-2
        }
        return space

    def loss_func(self, prms):
        params = {}
        params["optimizer"] = prms["set_optimizer"]["optimizer"]
        params["lr"] = prms["set_optimizer"]["lr"]
        params["momentum"] = prms["set_optimizer"]["momentum"]
        params["rho"] = prms["set_optimizer"]["rho"]
        params["beta_1"] = prms["set_optimizer"]["beta_1"]

        params["filters"] = int(prms["set_filters"]["filters"])
        params["ch1"]  = int(prms["ch1"])
        params["ch2"]  = int(prms["ch2"])
        params["ch3"]  = int(prms["set_filters"]["ch3"])
        params["d1"]  = int(prms["d1"])
        params["d2"]  = int(prms["d2"])
        params["bn"]  = int(prms["bn"])

        params["dropout1"] = prms["dropout"]
        params["dropout2"] = prms["dropout"]
        params["dropout3"] = prms["dropout"]

        params["batch_size"] = 128
        params["nb_epoch"] = 150
        params["patience"] = 5
        params['seed'] = 71
        params["steps_per_epoch"] = 3200 / params["batch_size"]

        # params["scaling"] = prms["scaling"]
        # params["apply_ln"] = prms["apply_ln"]

        params["bags"] = 1

        self.runner = RunnerKerasCNN4("keras_cnn_temp", flist=None, prms=params, Exp=False)
        loss, model = self.runner.run_hopt()
        return loss, model

hoptrunner = HoptRunnerKerasCNN()
hoptrunner.run(max_evals=250)
