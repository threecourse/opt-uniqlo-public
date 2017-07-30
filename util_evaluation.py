import numpy as np
import pandas as pd
from sklearn.externals import joblib
import os
import datetime
import sklearn.metrics
from util_decode import Decode

class Evaluation:

    @classmethod
    def log_loss(cls, actual, pred, num_classes=24, sample_weight=None):
        return sklearn.metrics.log_loss(actual, pred, labels=range(num_classes), sample_weight=sample_weight)

    @classmethod
    def accuracy(cls, actual, pred, sample_weight=None):
        pred_label = np.argmax(pred, axis=1)
        return sklearn.metrics.accuracy_score(actual, pred_label, sample_weight=sample_weight)

    @classmethod
    def balanced_accuracy(cls, actual, pred, num_classes=24):
        pred_label = np.argmax(pred, axis=1)
        df = pd.DataFrame({"actual":actual, "pred":pred_label})
        categories = range(num_classes)

        probs = []
        for c in categories:
            mask = actual == c
            numer = np.sum((pred_label[mask] == c))
            denom = np.size(pred_label[mask])
            if denom == 0:
                probs.append(1.0)
            else:
                probs.append(float(numer)/denom)

        return np.mean(probs)

    @classmethod
    def balanced_accuracy_adjust(cls, actual, pred, pred_y, num_classes=24):

        pred_label = Decode.inverse(pred, pred_y, num_classes)
        categories = range(num_classes)
        probs = []
        for c in categories:
            mask = actual == c
            numer = np.sum((pred_label[mask] == c))
            denom = np.size(pred_label[mask])
            if denom == 0:
                probs.append(1.0)
            else:
                probs.append(float(numer)/denom)

        return np.mean(probs)

    @classmethod
    def balanced_accuracy_label(cls, actual, pred_label, num_classes=24):
        categories = range(num_classes)
        probs = []
        for c in categories:
            mask = actual == c
            numer = np.sum((pred_label[mask] == c))
            denom = np.size(pred_label[mask])
            if denom == 0:
                probs.append(1.0)
            else:
                probs.append(float(numer)/denom)
            # print c, numer, denom

        return np.mean(probs)