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

class ModelMixGreedy2(ModelBase):

    def weighted_pred(self, x, weights):
        pred = np.zeros_like(x[0])
        for ii, w in enumerate(weights):
            pred += x[ii] * w / weights.sum()
        return pred

    def train(self, prms, x_tr, y_tr, x_te, y_te):

        n = len(x_tr)
        weights = np.zeros(n)
        weights[0] = 1.0

        while True:
            loss0 = Evaluation.log_loss(y_tr, self.weighted_pred(x_tr, weights))
            best_loss = loss0

            for i in range(1, n):
                _weights = weights.copy()
                _weights[i] += 0.1
                loss = Evaluation.log_loss(y_tr, self.weighted_pred(x_tr, _weights))
                if loss < best_loss:
                    best_loss = loss
                    best_weights = _weights

            weights = best_weights
            if best_loss == loss0: break

        logger.info(best_weights)
        self.weights = best_weights

    def train_without_validation(self, prms, x_tr, y_tr):
        raise NotImplementedError

    def save_model(self):
        Util.dumpc(self.weights, "../model/model/{}_weights.pkl".format(self.run_fold_name))

    def load_model(self):
        self.weights = Util.load("../model/model/{}_weights.pkl".format(self.run_fold_name))

    def predict(self, x_te):
        return self.weighted_pred(x_te, self.weights)
