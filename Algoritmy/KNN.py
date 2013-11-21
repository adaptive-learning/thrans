import numpy as np
import pandas as pd

class KNN():
    def __init__(self):
        pass

    def fit(self, distances, labels, k):
        predict = np.empty(len(labels))
        predict.fill(-1)

        for i in range(len(predict)):
            nearest =  [j[0] for j in sorted(enumerate(distances.ix[i]), key=lambda x:x[1]) if j[0] != i]
            nearest = nearest[:k]
            l = labels[nearest]

            predict[i] = max(set(l), key=l.tolist().count)

        return predict


    def compute_distances(self, points):
        distances = pd.DataFrame(index=points.index, columns=points.index)

        for i in points.index:
            for j in points.index:
                distance = self.euclidean_distance(points.ix[i], points.ix[j])
                distances.ix[i, j] = distance

        return distances

    def euclidean_distance(self, x, y):
        div = x - y
        return np.sqrt(np.dot(div, div))
