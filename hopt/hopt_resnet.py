import sys
sys.path += ["."]
import os

from util import Util
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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

from run_keras_resnet import RunnerKerasResnet
import copy
import numpy as np

class HoptRunnerKerasResnet(HoptRunner):

    def get_fpath(self):
        return "../model/hopt/resnet.hopt.txt"

    def get_prms_header_list(self):
        return ["structure", "lr", "momentum", "decay", "dropout1", "dropout2", "epochs"]

    def get_prms_values_list(self, prms, model):
        hist = model.hist.history
        info = copy.deepcopy(prms)
        info["epochs"] = len(hist["val_loss"])
        return [info[c] for c in self.get_prms_header_list()]

    def get_space(self):
        space = {
            "structure": "dense_dropout",
            "lr": hp.loguniform("lr", np.log(0.00001), np.log(0.001)),
            "momentum": 0.9,
            "decay": 0.0,
            "dropout1": hp.loguniform("dropout1", np.log(0.01), np.log(0.3)),
            "dropout2": hp.loguniform("dropout2", np.log(0.01), np.log(0.3)),
            "nesterov": True
        }
        return space

    def loss_func(self, prms):
        prms = copy.deepcopy(prms)
        prms["batch_size"] = 32
        prms["steps_per_epoch"] = 3200 / 32
        prms["patience"] = 3
        prms["nb_epoch"] = 60 # 20 epochs maximum
        runner = RunnerKerasResnet("resnet_hopt", flist=None, prms=prms, Exp=False)
        loss, model = runner.run_hopt()
        return loss, model

hoptrunner = HoptRunnerKerasResnet()
hoptrunner.run(max_evals=50)
