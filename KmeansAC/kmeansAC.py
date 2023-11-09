import pandas as pd
import numpy as np

class KMeans:
    def __init__(self, k):
        self.k = k
        self.centres = None

    def fit(self, x, eps=1e-5):
        centres = []
        for i in range(self.k):
            centres.append(x.iloc[np.random.randint(0, len(x))])
        centres = np.array(centres)
        while True:
            dists = []
            for i in range(self.k):
                dists.append(np.linalg.norm(x - centres[i], axis=1))
            dists = np.array(dists)
            clusters = np.argmin(dists, axis=0)
            new_centres = []
            for i in range(self.k):
                new_centres.append(np.mean(x.iloc[clusters == i], axis=0))
            new_centres = np.array(new_centres)
            if np.linalg.norm(new_centres - centres) < eps:
                break
            centres = new_centres
        self.centres = centres

    def predict(self, x):
        dists = []
        for i in range(self.k):
            dists.append(np.linalg.norm(x - self.centres[i], axis=1))
        dists = np.array(dists)
        clusters = np.argmin(dists, axis=0)
        return clusters.tolist()
