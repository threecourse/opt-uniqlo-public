import sys
sys.path.append(".")
from util import Util
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from create_cluster_images import ClusterImagesCreator
from tsp_greedy import TSP

def run_clustering():

    if Train:
        vectors = Util.load("../model/analysis/feature_silhouette_train.pkl")
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
        tsp_path = np.array(TSP.solve_tsp(center_dists))

        Util.dumpc(model, "../model/analysis/cluster_silhouette_model.pkl")
        Util.dumpc(tsp_path, "../model/analysis/cluster_silhouette_tsp_path.pkl")

        # use argsort
        clusters_tsp = np.argsort(tsp_path)[clusters]

        files = pd.read_csv("../model/filenames_index_train.csv", sep="\t")["file_name"]
        n = vectors.shape[0]
        fnames = files[:n]
        df = pd.DataFrame(zip(fnames, clusters_tsp, clusters), columns=["file_name", "cls", "cls_raw"])

        Util.to_csv(df, "../model/analysis/cluster_silhouette_train.csv")

    if Test:
        vectors = Util.load("../model/analysis/feature_silhouette_test.pkl")
        model = Util.load("../model/analysis/cluster_silhouette_model.pkl")
        clusters = model.predict(vectors)
        tsp_path = Util.load("../model/analysis/cluster_silhouette_tsp_path.pkl")

        clusters_tsp = np.argsort(tsp_path)[clusters]

        files = pd.read_csv("../model/filenames_index_test.csv", sep="\t")["file_name"]
        n = vectors.shape[0]
        fnames = files[:n]
        df = pd.DataFrame(zip(fnames, clusters_tsp, clusters), columns=["file_name", "cls", "cls_raw"])
        Util.to_csv(df, "../model/analysis/cluster_silhouette_test.csv")

def run_create_pdf():
    if Train:
        df = Util.read_csv("../model/analysis/cluster_silhouette_train.csv")
        df_labels = Util.read_csv("../model/filenames_index_train.csv")
        df = df.merge(df_labels, on="file_name")
        ClusterImagesCreator.run("../model/analysis/silhouette_train.pdf", df, n_clusters, folder="../data/train/", cid_actual=True)
    if Test:
        df = Util.read_csv("../model/analysis/cluster_silhouette_test.csv")
        df_labels = pd.read_csv("../submission/subm_2017-05-07-09-04_vgg_adjust.csv", header=None, sep=",")
        df_labels.columns = ["file_name", "cid"]
        df = df.merge(df_labels, on="file_name")
        ClusterImagesCreator.run("../model/analysis/silhouette_test.pdf", df, n_clusters, folder="../data/test/", cid_actual=False)

if __name__ == "__main__":
    n_clusters = 50
    Train = False
    Test = True
    # run_clustering()
    run_create_pdf()