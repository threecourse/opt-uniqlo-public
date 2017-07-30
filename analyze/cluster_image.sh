#!/usr/bin/env bash

python analyze/preprocess_cluster_silhouette_feature_.py
python analyze/preprocess_cluster_silhouette_kmeans.py
python analyze/create_cluster_images.py