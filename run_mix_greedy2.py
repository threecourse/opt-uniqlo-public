import pandas as pd
import numpy as np
from model_mix_greedy2 import ModelMixGreedy2
from model import ModelRunnerBase
import copy
import h5py
from util import Util
from submission import Submission

class RunnerMixGreedy2(ModelRunnerBase):

    def build_model(self, run_fold_name):
        model = ModelMixGreedy2(run_fold_name)
        model.load_index_fold = self.load_index_fold
        return model

    def pred_path(self, run_name, is_train):
        if is_train:
            return "../model/pred/pred_values_{}_train.csv".format(run_name)
        else:
            return "../model/pred/pred_values_{}_test.csv".format(run_name)

    def _load_x(self, is_train):
        preds = []
        for run_name in self.flist:
            pred = Util.read_csv(self.pred_path(run_name, is_train))
            preds.append(pred.values)
        return preds

    def load_x_train(self):
        return self._load_x(is_train=True)

    def load_x_test(self):
        return self._load_x(is_train=False)

def runmix(name, flist):
    runner = RunnerMixGreedy2(name, flist=flist, prms={})
    runner.run_train()
    runner.run_test()
    Submission.make_submission(name)

if __name__ == "__main__":

    pass



