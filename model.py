import numpy as np
import pandas as pd
from util import Util
from util_log import Logger
from util_evaluation import Evaluation
logger = Logger()
from collections import OrderedDict

class ModelBase(object):

    def __init__(self, run_fold_name):
        self.run_fold_name = run_fold_name

    def train(self, prms, x_tr, y_tr, x_te, y_te):
        raise NotImplementedError

    def train_without_validation(self, prms, x_tr, y_tr):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def predict(self, x_te):
        raise NotImplementedError


class ModelRunnerBase(object):

    def __init__(self, run_name, flist, prms, Exp=False, map_colors=None):
        self.run_name = run_name
        self.flist = flist
        self.prms = prms
        self.Exp = Exp
        self.map_colors = map_colors
        if self.map_colors is None:
            self.num_classes = 24
        else:
            self.num_classes = len(map_colors)

    def build_model(self, run_fold_name):
        """ :rtype: ModelBase """
        raise NotImplementedError

    def load_x_train(self):
        raise NotImplementedError

    def load_x_test(self):
        raise NotImplementedError

    def load_y_train(self):
        df = pd.read_csv("../model/filenames_index_train.csv", sep="\t")
        y = df["cid"].values

        if self.map_colors is not None:
            mapper = np.zeros(24).astype(int)
            for i, cs in enumerate(self.map_colors):
                for c in cs:
                    mapper[c] = i
            y = mapper[y]

        return y

    def load_w_train(self):
        if self.map_colors is None:
            return None
        else:
            # balanced weight
            y = self.load_y_train()
            bincount =  np.bincount(y)
            w0 = 0.5 * len(y) / bincount[0]
            w1 = 0.5 * len(y) / (len(y) - bincount[0])
            print w0, w1
            weights = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                weights[i] = w0 if i == 0 else w1

            return weights[y]

    def load_index_fold(self, i_fold):

        try:
            chr, i = None, int(i_fold)
        except ValueError:
            chr, i = i_fold[0], int(i_fold[1:])

        if chr is None:
            # default
            dfidx = Util.read_csv("../model/train_index.csv")
        else:
            # "a0", "a1"..
            dfidx = Util.read_csv("../model/train_index_{}.csv".format(chr))

        if self.Exp:
            dfidx = dfidx[:1000]
        idx_tr = np.array(dfidx[dfidx["fold_idx"] != i]["idx"])
        idx_te = np.array(dfidx[dfidx["fold_idx"] == i]["idx"])

        return idx_tr, idx_te

    def load_x_fold(self, i_fold):
        idx_tr, idx_te = self.load_index_fold(i_fold)
        x = self.load_x_train()
        if isinstance(x, list):
            return ([xx[idx_tr, :] for xx in x],
                    [xx[idx_te, :] for xx in x])
        else:
            return x[idx_tr, :], x[idx_te, :]

    def load_y_fold(self, i_fold):
        idx_tr, idx_te = self.load_index_fold(i_fold)
        y = self.load_y_train()
        # return y[idx_tr, :], y[idx_te, :]
        return y[idx_tr], y[idx_te]

    def load_w_fold(self, i_fold):
        idx_tr, idx_te = self.load_index_fold(i_fold)
        w = self.load_w_train()
        if w is None:
            return None, None
        else:
            return w[idx_tr], w[idx_te]

    def run_fold_name(self, i_fold):
        return "{}_fold{}".format(self.run_name, i_fold)

    def train_fold(self, i_fold, save=True):
        logger.info( "run train_fold {} {}".format(self.run_name, i_fold))

        run_fold_name = self.run_fold_name(i_fold)

        model = self.build_model(run_fold_name)

        if i_fold == -1:
            x_tr = self.load_x_train()
            y_tr = self.load_y_train()
            w_tr = self.load_w_train()
            if w_tr is None:
                model.train_without_validation(self.prms, x_tr, y_tr)
            else:
                model.train_without_validation(self.prms, x_tr, y_tr, w_tr)
        else:
            x_tr, x_te = self.load_x_fold(i_fold)
            y_tr, y_te = self.load_y_fold(i_fold)
            w_tr, w_te = self.load_w_fold(i_fold)
            if w_tr is None:
                model.train(self.prms, x_tr, y_tr, x_te, y_te)
            else:
                model.train(self.prms, x_tr, y_tr, x_te, y_te, w_tr, w_te)
        if save:
            model.save_model()
        return model

    def run_hopt(self, i_fold = "a0"):
        logger.info( "run train {}".format(self.run_name))

        model = self.train_fold(i_fold, save=False)

        x_tr, x_te = self.load_x_fold(i_fold)
        y_tr, y_te = self.load_y_fold(i_fold)
        w_tr, w_te = self.load_w_fold(i_fold)
        idx_tr, idx_te = self.load_index_fold(i_fold)

        pred = model.predict(x_te)
        dfpred = pd.DataFrame(idx_te, columns=["idx"])
        dfpred = pd.concat([dfpred, pd.DataFrame(pred, columns=range(self.num_classes))], axis=1)

        acc = Evaluation.accuracy(y_te, pred, w_te)
        logloss = Evaluation.log_loss(y_te, pred, self.num_classes, w_te)
        # TODO consider weights
        bacc = Evaluation.balanced_accuracy_adjust(y_te, pred, y_tr, self.num_classes)

        return logloss, model

    def run_train(self):
        logger.info( "run train {}".format(self.run_name))

        preds = []
        accs, loglosses, baccs = [], [], []

        folds = range(5)
        for i_fold in folds:
            model = self.train_fold(i_fold, save=True)

            x_tr, x_te = self.load_x_fold(i_fold)
            y_tr, y_te = self.load_y_fold(i_fold)
            w_tr, w_te = self.load_w_fold(i_fold)
            idx_tr, idx_te = self.load_index_fold(i_fold)

            pred = model.predict(x_te)
            dfpred = pd.DataFrame(idx_te, columns=["idx"])
            dfpred = pd.concat([dfpred, pd.DataFrame(pred, columns=range(self.num_classes))], axis=1)

            acc = Evaluation.accuracy(y_te, pred, w_te)
            accs.append(acc)
            logloss = Evaluation.log_loss(y_te, pred, self.num_classes, w_te)
            loglosses.append(logloss)
            # TODO consider weights
            bacc = Evaluation.balanced_accuracy_adjust(y_te, pred, y_tr, self.num_classes)
            baccs.append(bacc)

            d = OrderedDict([('run_name', self.run_name), ('fold', i_fold), ('acc', acc), ('logloss', logloss), ('bacc', bacc)])
            logger.result_ltsv_time(d)
            preds.append(dfpred)

        preds = pd.concat(preds, axis=0)
        preds = preds.sort_values("idx").drop("idx", axis=1)
        logger.info("acc:{}, logloss:{}, bacc:{}".format(np.mean(accs), np.mean(loglosses), np.mean(baccs)))
        Util.to_csv(preds, "../model/pred/pred_values_{}_train.csv".format(self.run_name))

    def run_test(self):
        logger.info("run test {}".format(self.run_name))

        # by average fold models

        x_te = self.load_x_test()
        preds = []

        # get average of folds
        folds = range(5)
        for i_fold in folds:
            run_fold_name = self.run_fold_name(i_fold)

            model = self.build_model(run_fold_name)
            model.load_model()
            pred = model.predict(x_te)
            preds.append(pred)
            logger.info("test predicted {}".format(i_fold))

        preds_mean = np.mean(preds, axis=0)
        preds_mean = pd.DataFrame(preds_mean, columns=range(self.num_classes))
        Util.to_csv(preds_mean, "../model/pred/pred_values_{}_test.csv".format(self.run_name))

    def run_train_alltrain(self):
        logger.info("run train_alltrain {}".format(self.run_name))
        self.train_fold(-1, save=True)

    def run_test_alltrain(self):
        logger.info("run test_alltrain {}".format(self.run_name))
        x_te = self.load_x_test()

        run_fold_name = self.run_fold_name(-1)
        model = self.build_model(run_fold_name)
        model.load_model()
        pred = model.predict(x_te)
        pred = pd.DataFrame(pred, columns=range(self.num_classes))
        Util.to_csv(pred, "../model/pred/pred_values_{}_test.csv".format(self.run_name))