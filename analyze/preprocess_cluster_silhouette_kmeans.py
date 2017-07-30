import sys
sys.path.append(".")
import os
from util import Util
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from preprocess_util_tsp_greedy import TSP

def run_clustering(n_clusters=50):

    train_feature_path = "../model/feature/silhouette_train.pkl"
    model_path = "../model/feature/silhouette_model_{}.pkl".format(n_clusters)
    tsp_path = "../model/feature/silhouette_tsp_path_{}.pkl".format(n_clusters)
    test_feature_path = "../model/feature/silhouette_test.pkl"
    cls_csv_path_train = "../model/feature/silhouette_cls_{}_train.csv".format(n_clusters)
    cls_csv_path_test = "../model/feature/silhouette_cls_{}_test.csv".format(n_clusters)

    if Train:
        vectors = Util.load(train_feature_path)
        print "start clustering"
        model = KMeans(n_clusters=n_clusters, random_state=71, n_jobs=4).fit(vectors)
        clusters = model.predict(vectors)
        print "finished clustering"

        def distances(ary):
            dists = []
            for i in range(ary.shape[0]):
                dists.append(np.sqrt(((centers - centers[i, :]) ** 2).sum(axis=1)))
            dists = np.array(dists)
            print dists.shape
            return dists

        centers = model.cluster_centers_
        center_dists = distances(centers)
        tsp = np.array(TSP.solve_tsp(center_dists))

        Util.dumpc(model, model_path)
        Util.dumpc(tsp, tsp_path)

        # use argsort
        clusters_tsp = np.argsort(tsp)[clusters]

        files = pd.read_csv("../model/filenames_index_train.csv", sep="\t")["file_name"]
        n = vectors.shape[0]
        fnames = files[:n]
        df = pd.DataFrame(zip(fnames, clusters_tsp, clusters), columns=["file_name", "cls", "cls_raw"])

        Util.to_csv(df, cls_csv_path_train)

    if Test:
        vectors = Util.load(test_feature_path)
        model = Util.load(model_path)
        tsp = Util.load(tsp_path)

        clusters = model.predict(vectors)
        clusters_tsp = np.argsort(tsp)[clusters]

        files = pd.read_csv("../model/filenames_index_test.csv", sep="\t")["file_name"]
        n = vectors.shape[0]
        fnames = files[:n]
        df = pd.DataFrame(zip(fnames, clusters_tsp, clusters), columns=["file_name", "cls", "cls_raw"])
        Util.to_csv(df, cls_csv_path_test)

if __name__ == "__main__":
    Train = True
    Test = True
    run_clustering()
