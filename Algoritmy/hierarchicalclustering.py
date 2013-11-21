import numpy as np
import pandas as pd
import networkx as nx


class HierarichicalClustering():
    def __init__(self, points=None, distances=None, distance=None):
        self.points = points
        self.distances = distances
        self.tree = nx.Graph()
        self.distance = distance if distance else self.euclidean_distance

    def compute_distances(self):
        self.distances = pd.DataFrame(index=self.points.index, columns=self.points.index)

        for i in self.points.index:
            for j in self.points.index:
                distance = self.distance(self.points.ix[i], self.points.ix[j])
                self.distances.ix[i, j] = distance

        return self.distances

    def build_tree(self):
        cluster_distance = self.cluster_distance_min

        self.tree.add_nodes_from(self.points.index)

        working_clusters = self.points.index

        minimal_distance = None
        winner = None
        for i, first_cluster in enumerate(working_clusters):
            for second_cluster in working_clusters[i + 1:]:
                distance = cluster_distance(first_cluster, second_cluster)
                if not minimal_distance or minimal_distance > distance:
                    minimal_distance = distance
                    winner = (first_cluster, second_cluster)

        print winner
        return self.tree


    def cluster_distance_min(self, x, y):
        minimal_distance = None
        for i in x.index:
            for j in y.index:
                if not minimal_distance or minimal_distance > self.distances[x,y]:
                    minimal_distance = self.distances[x,y]
        return minimal_distance


    def euclidean_distance(self, x, y):
        div = x - y
        return np.sqrt(np.dot(div, div))
