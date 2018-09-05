from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.features = features
        self.labels = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        from queue import PriorityQueue
        labels = []
        LOO = False
        if features == self.features:
            LOO = True
        for i in range(len(features)):
            q = PriorityQueue()
            for j in range(len(self.features)):
                if i == j and LOO:
                    continue
                dist = self.distance_function(features[i],self.features[j])
                q.put((dist,self.labels[j]))
            pos, neg = 0, 0
            for n in range(self.k):
                if q.empty():
                    break
                dist, l = q.get()
                if l > 0:
                    pos += 1
                else:
                	neg += 1
            labels.append(1) if pos >= neg else labels.append(0)
        return labels


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
