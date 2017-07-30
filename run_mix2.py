import pandas as pd
import numpy as np
from model_mix import ModelMix
from model import ModelRunnerBase
import copy
import h5py
from util import Util
from submission import Submission

class RunnerMix2(ModelRunnerBase):

    def build_model(self, run_fold_name):
        model = ModelMix(run_fold_name)
        model.load_index_fold = self.load_index_fold
        return model

    def pred_path(self, run_name, is_train):
        if is_train:
            return "../model/pred/pred_values_{}_train.csv".format(run_name)
        else:
            return "../model/pred/pred_values_{}_test.csv".format(run_name)

    def _load_x(self, is_train):
        preds = []
        if isinstance(self.flist, tuple):
            flist, weight = self.flist
        else:
            flist = self.flist
            weight = np.ones(len(flist))

        for run_name, w in zip(flist, weight):
            pred = Util.read_csv(self.pred_path(run_name, is_train))
            preds.append(pred.values * w / np.sum(weight))
        return np.array(preds).sum(axis=0)

    def load_x_train(self):
        return self._load_x(is_train=True)

    def load_x_test(self):
        return self._load_x(is_train=False)

def runmix(name, flist):
    runner = RunnerMix2(name, flist=flist, prms={})
    runner.run_train()
    runner.run_test()
    Submission.make_submission(name)

if __name__ == "__main__":

    mix_vgg2_10 =  ["vgg2_e22", "vgg2_e23", "vgg2_e24", "vgg2_e25", "vgg2_e26",
                    "vgg2_2_e22", "vgg2_2_e23", "vgg2_2_e24", "vgg2_2_e25", "vgg2_2_e26"]
    mix_resnet_20 = ["resnet_e18", "resnet_e19", "resnet_e20", "resnet_e21", "resnet_e22",
                    "resnet_2_e18", "resnet_2_e19", "resnet_2_e20", "resnet_2_e21", "resnet_2_e22",
                    "resnet_3_e18", "resnet_3_e19", "resnet_3_e20", "resnet_3_e21", "resnet_3_e22",
                    "resnet_4_e18", "resnet_4_e19", "resnet_4_e20", "resnet_4_e21", "resnet_4_e22"]
    mix_resnet_aug_10 = ["resnet_aug_e25", "resnet_aug_e26", "resnet_aug_e27", "resnet_aug_e28", "resnet_aug_e29",
                         "resnet_aug_2_e25", "resnet_aug_2_e26", "resnet_aug_2_e27", "resnet_aug_2_e28", "resnet_aug_2_e29",]
    mix_vgg2_aug_10 = ["vgg2_aug_e24", "vgg2_aug_e25", "vgg2_aug_e26", "vgg2_aug_e27", "vgg2_aug_e28",
                       "vgg2_aug_2_e24", "vgg2_aug_2_e25", "vgg2_aug_2_e26", "vgg2_aug_2_e27", "vgg2_aug_2_e28"]
    mix_cnn_color_20 = ["keras_cnn4", "keras_cnn4_2", "keras_cnn4_3", "keras_cnn4_4"]
    mix_cnn5_1_20 = ["keras_cnn5_e73", "keras_cnn5_e74", "keras_cnn5_e75", "keras_cnn5_e76"]
    mix_cnn5_2_20 = ["keras_cnn5_e43", "keras_cnn5_e44", "keras_cnn5_e45", "keras_cnn5_e46"]
    mix_cnn5_3_20 = ["keras_cnn5_2_e73", "keras_cnn5_2_e74", "keras_cnn5_2_e75", "keras_cnn5_2_e76"]
    mix_cnn5_5_20 = ["keras_cnn5_3_st1_e58", "keras_cnn5_3_st1_e59", "keras_cnn5_3_st1_e60", "keras_cnn5_3_st1_e61"]
    mix_cnn5_7_20 = ["keras_cnn5_4_1_e56", "keras_cnn5_4_1_e57", "keras_cnn5_4_1_e58", "keras_cnn5_4_1_e59"]

    runmix("mix_fin1_vgg2_10", mix_vgg2_10)
    runmix("mix_fin1_resnet_20", mix_resnet_20)
    runmix("mix_fin1_resnet_aug_10", mix_resnet_aug_10)
    runmix("mix_fin1_mix_vgg2_aug_10", mix_vgg2_aug_10)
    runmix("mix_fin1_cnn_color_20", mix_cnn_color_20)
    runmix("mix_fin1_cnn5_1_20", mix_cnn5_1_20)
    runmix("mix_fin1_cnn5_2_20", mix_cnn5_2_20)
    runmix("mix_fin1_cnn5_3_20", mix_cnn5_3_20)
    runmix("mix_fin1_cnn5_5_20", mix_cnn5_5_20)
    runmix("mix_fin1_cnn5_7_20", mix_cnn5_7_20)
