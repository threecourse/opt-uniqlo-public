import pandas as pd
import numpy as np
from skimage import io as skio
import skimage

class UtilMisc:
    cmaster = pd.read_csv("../data/category_master.tsv", sep="\t")
    cmaster = cmaster.rename(columns={"catgory_name": "cname", "category_id":"cid"})
    cdict = [(r["cname"], r["cid"]) for i, r in cmaster.iterrows()]

    @classmethod
    def cname(cls, cid):
        return cls.cdict[cid]

class UtilImage:

    train_dir = "../data/train/"

    @classmethod
    def loadimage(cls, path, resize=None):
        img = skio.imread(path)
        if resize is not None:
            img = skimage.transform.resize(img, resize, mode='reflect')
        return img
