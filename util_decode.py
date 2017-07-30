import sys, os
sys.path.append("")

import numpy as np
import pandas as pd
import util

from scipy.optimize import minimize
from scipy.optimize import fmin_powell

class Decode:

    @classmethod
    def raw(cls, pred, act=None):
        return np.argmax(pred, axis=1)

    @classmethod
    def inverse(cls, pred, act, num_classes=24):
        # TODO: take care of ratio 0.0
        ratio = np.bincount(act) / float(len(act))
        assert(len(ratio) == num_classes)
        pred = pred / ratio.reshape(-1, num_classes)
        return np.argmax(pred, axis=1)

    @classmethod
    def inverse_by_train(cls, pred):
        df = pd.read_csv("../model/filenames_index_train.csv", sep="\t")
        return cls.inverse(pred, df["cid"])

    @classmethod
    def cid_counts(cls):
        df = pd.read_csv("../model/filenames_index_train.csv", sep="\t")
        counts = np.bincount(df["cid"])
        return counts