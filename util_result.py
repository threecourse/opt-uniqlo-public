import numpy as np
import pandas as pd
import re
from util import Util

def to_dict(line):
    vals = line.rstrip().split("\t")
    d = {}
    for v in vals:
        m = re.match("(.+?):(.+)", v)
        if m:
            d[m.group(1)] = m.group(2)
    return d

def parse_result():
    path = "../model/result.log"
    columns = ["time", "run_name", "fold", "acc", "logloss", "bacc"]
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [to_dict(line) for line in lines]
        df = pd.DataFrame(lines, columns=columns)
        for c in ["acc", "logloss", "bacc"]:
            df[c] = df[c].astype(float)

        g = df.groupby("run_name").agg({'time':["first"], "fold":["count"],
                                        'acc':['mean', "std"], 'logloss':['mean', "std"], 'bacc':['mean', "std"]})

        g = g.sort_values(('time', 'first'))
        g = g.reindex(columns=[("fold", "count"),
                               ("bacc", "mean"), ("bacc", "std"),
                               ("acc", "mean"), ("acc", "std"),
                               ("logloss", "mean"), ("logloss", "std"),
                               ("time", "first"),])
        g = g.reset_index()
        g.columns = [' '.join(col).strip() for col in g.columns.values]

        return df, g

if __name__ == "__main__":
    df, g = parse_result()
    Util.to_csv(df, "../model/result.df.txt")
    Util.to_csv(g, "../model/result.g.txt")