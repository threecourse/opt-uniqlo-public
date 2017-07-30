import pandas as pd
import numpy as np
from model_adjuster import ModelAdjuster
from model import ModelRunnerBase
import copy
import h5py
from util import Util
from submission import Submission

class RunnerAdjuster(ModelRunnerBase):

    def build_model(self, run_fold_name):
        model = ModelAdjuster(run_fold_name)
        model.load_index_fold = self.load_index_fold
        return model

    def pred_path(self, run_name, is_train):
        if is_train:
            return "../model/pred/pred_values_{}_train.csv".format(run_name)
        else:
            return "../model/pred/pred_values_{}_test.csv".format(run_name)

    def _load_x(self, is_train):
        pred = Util.read_csv(self.pred_path(self.flist, is_train)).values
        return pred

    def load_x_train(self):
        return self._load_x(is_train=True)

    def load_x_test(self):
        return self._load_x(is_train=False)

def runmix(name, flist, prms):
    runner = RunnerAdjuster(name, flist=flist, prms=prms)
    runner.run_train()
    runner.run_test()
    Submission.make_submission(name)

if __name__ == "__main__":
    # runmix(name="mix_greedy_10_adjust", flist="mix_greedy_10", prms={"criteria":"logloss"})
    # runmix(name="mix_fin1_greedy2_adjust_bacc", flist="mix_fin1_greedy2", prms={"criteria":"bacc"})
    # runmix(name="mix_fin1_greedy2_adjust_logloss", flist="mix_fin1_greedy2", prms={"criteria":"logloss"})
    runmix(name="mix_fin1_greedy6_logloss", flist="mix_fin1_greedy6", prms={"criteria":"logloss"})