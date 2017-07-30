import sys
sys.path.append(".")

import h5py
import numpy as np
import pandas as pd
from util import Util
import re
from util_evaluation import Evaluation
from model import ModelBase, ModelRunnerBase
from util_log import Logger
logger = Logger()

class ModelAdjuster(ModelBase):

    def adjusted_pred(self, x, ratio, f1, f2):
        x = x.copy()
        r = (np.power(ratio, f2) / ratio).reshape(-1, 24)
        x = np.power(x, f1) * r
        d = np.sum(x, axis=1).reshape(-1, 1)
        # print r, d
        x = x / d
        return x

    def train(self, prms, x_tr, y_tr, x_te, y_te):
        self.prms = prms

        ratio = np.bincount(y_tr) / float(len(y_tr))
        assert (len(ratio) == 24)

        best_loss = 99.99
        best_bacc = 0.0
        best_f1, best_f2 = None, None
        for f1 in np.arange(0.90, 1.3, 0.01):
            for f2 in np.arange(0.90, 1.1, 0.01):
                x = self.adjusted_pred(x_tr, ratio, f1, f2)

                loss = Evaluation.log_loss(y_tr, x)
                # bacc = Evaluation.balanced_accuracy_adjust(y_te, x, y_tr)
                bacc = 0.0

                if self.prms["criteria"] == "logloss":
                    if loss < best_loss:
                        best_loss = loss
                        best_f1, best_f2 = f1, f2
                elif self.prms["criteria"] == "bacc":
                    if -bacc < -best_bacc:
                        best_bacc = bacc
                        best_f1, best_f2 = f1, f2
                else:
                    raise Exception

        logger.info("{} {}".format(best_f1, best_f2))
        self.adjuster = (ratio, best_f1, best_f2)

    def train_without_validation(self, prms, x_tr, y_tr):
        raise NotImplementedError

    def save_model(self):
        Util.dumpc(self.prms, "../model/model/{}_prms.pkl".format(self.run_fold_name))
        Util.dumpc(self.adjuster, "../model/model/{}_adjuster.pkl".format(self.run_fold_name))

    def load_model(self):
        self.prms = Util.load("../model/model/{}_prms.pkl".format(self.run_fold_name))
        self.adjuster = Util.load("../model/model/{}_adjuster.pkl".format(self.run_fold_name))

    def predict(self, x_te):
        ratio, best_f1, best_f2 = self.adjuster
        return self.adjusted_pred(x_te, ratio, best_f1, best_f2)
