import sys
sys.path.append(".")
from util import Util
import pandas as pd
import numpy as np
from skimage import io as skio
import skimage
import skimage.transform

def get_feature(path):
    resize = (64, 64)
    img = skio.imread(path)
    img = skimage.transform.resize(img, resize, mode='reflect') * 255
    mask = (img[:, :, 0] == 255) + (img[:, :, 1] == 255) + (img[:, :, 2] == 255)
    silhouette =  np.where(mask, 1, 0)
    return silhouette.reshape(-1)

def run():

    if Train:
        df = pd.read_csv("../model/filenames_index_train.csv", sep="\t")
        files = df["file_name"]
        if Exp:
            n = 1000
        else:
            n = len(files)

        vectors = []
        for i, fname in enumerate(files[:n]):
            feat = get_feature("../data/train/" + fname)
            vectors.append(feat)
            if i % 100 == 0: print "extracted", i
        vectors = np.array(vectors)
        print vectors.shape

        Util.dumpc(vectors, "../model/feature/silhouette_train.pkl")

    if Test:
        df = pd.read_csv("../model/filenames_index_test.csv", sep="\t")
        files = df["file_name"]
        if Exp:
            n = 1000
        else:
            n = len(files)

        vectors = []
        for i, fname in enumerate(files[:n]):
            feat = get_feature("../data/test/" + fname)
            vectors.append(feat)
            if i % 100 == 0: print "extracted", i
        vectors = np.array(vectors)
        print vectors.shape

        Util.dumpc(vectors, "../model/feature/silhouette_test.pkl")

if __name__ == "__main__":
    Train = True
    Test = True
    Exp = False
    run()

