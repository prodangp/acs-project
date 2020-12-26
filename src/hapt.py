from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import glob
from random import randint
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

MEAS_NO = 61
CLUSTERS_NO = 6
PATH = r'C:\Users\info\Desktop\Info Anul III\ACS\Proiect\HAPT\RawData'


def get_hapt_data():
    activities = [None] * MEAS_NO  # create array for storing the measurements
    for i in range(0, MEAS_NO):
        activities[i] = []

    f = open(PATH + r'\labels.txt')
    for line in f.readlines():
        data = line.split()
        data = [int(d) for d in data]
        activities[data[0] - 1].append(data[1:])  # reading measurements details (see HAPT/README)
    f.close()

    dataset = [None] * CLUSTERS_NO
    sk_dataset = []
    activity_labels = []
    for i in range(0, CLUSTERS_NO):
        dataset[i] = []

    for exp_no in range(0, MEAS_NO):
        acc_path = glob.glob(PATH + r'\acc_*' + str(exp_no + 1) + '_user*')
        gyro_path = glob.glob(PATH + r'\gyro_*' + str(exp_no + 1) + '_user*')
        for activity in activities[exp_no]:
            x_acc = []
            y_acc = []
            z_acc = []
            x_gyro = []
            y_gyro = []
            z_gyro = []
            acc = open(acc_path[0])
            for no, line in enumerate(acc.readlines()):
                values = line.split()
                if activity[2] < no < activity[3]:
                    x_acc.append(float(values[0]))
                    y_acc.append(float(values[1]))
                    z_acc.append(float(values[2]))
            acc.close()
            gyro = open(gyro_path[0])
            for no, line in enumerate(gyro.readlines()):
                values = line.split()
                if activity[2] < no < activity[3]:
                    x_gyro.append(float(values[0]))
                    y_gyro.append(float(values[1]))
                    z_gyro.append(float(values[2]))
            gyro.close()
            cluster_no = randint(0, 5)
            data = [cluster_no,
                    np.mean(x_acc), np.mean(y_acc), np.mean(z_acc),
                    np.mean(x_gyro), np.mean(y_gyro), np.mean(z_gyro)]
            try:
                dataset[activity[1] - 1].append(data)
                sk_dataset.append(data[1:])
                activity_labels.append(activity[1] - 1)
            except IndexError:
                pass
    return dataset, sk_dataset, activity_labels


def compute_centroids(dataset):
    centroids = []
    clusters = [None] * CLUSTERS_NO
    for i in range(0, CLUSTERS_NO):
        clusters[i] = []
    for a in dataset:
        for el in a:
            clusters[el[0]].append(el[1:])
    for cluster in clusters:
        centroid = np.mean([el for el in cluster], axis=0)
        centroids.append(centroid)
    return centroids


def find_min(x, centroids, norm=2):
    distances = [np.linalg.norm(x - centroid, norm) for centroid in centroids]
    return np.argmin(distances)


def update_clusters(dataset, stabilize=False):
    centroids = compute_centroids(dataset)
    if stabilize:
        movers = 0
        for a in dataset:
            for el in a:
                initial = el[0]
                el[0] = find_min(np.array(el[1:]), centroids)
                if initial != el[0]:
                    movers += 1
        return dataset, movers
    else:
        for a in dataset:
            for el in a:
                el[0] = find_min(np.array(el[1:]), centroids)
        return dataset


def kmeans(dataset, iterations):
    if dataset is None:
        dataset, _, _ = get_hapt_data()
    else:
        pass
    for i in range(0, iterations):
        dataset = update_clusters(dataset)
    results = [None] * CLUSTERS_NO
    for i in range(0, CLUSTERS_NO):
        results[i] = [0] * 6
    for a_no, a in enumerate(dataset):
        for el in a:
            results[a_no][el[0]] += 1
    return np.array(results), dataset


def kmeans_stabilize(dataset):
    if dataset is None:
        dataset, _, _ = get_hapt_data()
    else:
        pass
    iterations = 0
    movers = 1
    while movers > 0:
        iterations += 1
        dataset, movers = update_clusters(dataset, stabilize=True)
    results = [None] * CLUSTERS_NO
    for i in range(0, CLUSTERS_NO):
        results[i] = [0] * 6
    for a_no, a in enumerate(dataset):
        for el in a:
            results[a_no][el[0]] += 1
    return np.array(results), dataset, iterations


def scikit_kmeans():
    _, dataset, activity_labels = get_hapt_data()
    dist = KMeans(n_clusters=6).fit(dataset)
    results = [None] * CLUSTERS_NO
    for i in range(0, CLUSTERS_NO):
        results[i] = [0] * 6
    for index, label in enumerate(activity_labels):
        results[label][dist.labels_[index]] += 1
    return np.array(results)


def scikit_hierarchical():
    _, dataset, activity_labels = get_hapt_data()
    dist = AgglomerativeClustering(n_clusters=6).fit(dataset)
    results = [None] * CLUSTERS_NO
    for i in range(0, CLUSTERS_NO):
        results[i] = [0] * 6
    for index, label in enumerate(activity_labels):
        results[label][dist.labels_[index]] += 1
    return np.array(results)
