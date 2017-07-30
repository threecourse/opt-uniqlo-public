import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from util import Util
from sklearn.preprocessing import StandardScaler

class UtilCV:

    @classmethod
    def filenames_index(cls):
        df = pd.read_csv("../data/train_master.tsv", sep="\t")
        df = df.rename(columns={"category_id":"cid"})
        df["idx"] = np.array(range(len(df)))

        df_test = pd.read_csv("../data/sample_submit.csv", sep=",", header=None)
        df_test.columns = ["file_name", "cid"]
        df_test["cid"] = 0
        df_test["idx"] = np.array(range(len(df_test)))

        Util.to_csv(df, "../model/filenames_index_train.csv")
        Util.to_csv(df_test, "../model/filenames_index_test.csv")

    @classmethod
    def write_cv_index(cls):
        df = pd.read_csv("../model/filenames_index_train.csv", sep="\t")
        n_folds = 5
        folds = list(StratifiedKFold(df["cid"], n_folds=n_folds, shuffle=True, random_state=71))

        dfidx = pd.DataFrame(np.array(range(len(df))), columns=["idx"])
        dfidx["fold_idx"] = 0
        for i, fold in enumerate(folds):
            dfidx.iloc[fold[1], 1] = i

        Util.to_csv(dfidx, "../model/train_index.csv")

    @classmethod
    def write_cv_index_chr(cls, seed, chr):
        df = pd.read_csv("../model/filenames_index_train.csv", sep="\t")
        n_folds = 5
        folds = list(StratifiedKFold(df["cid"], n_folds=n_folds, shuffle=True, random_state=seed))

        dfidx = pd.DataFrame(np.array(range(len(df))), columns=["idx"])
        dfidx["fold_idx"] = 0
        for i, fold in enumerate(folds):
            dfidx.iloc[fold[1], 1] = i

        Util.to_csv(dfidx, "../model/train_index_{}.csv".format(chr))

if __name__ == "__main__":
    UtilCV.filenames_index()
    UtilCV.write_cv_index()
    UtilCV.write_cv_index_chr(72, "a")