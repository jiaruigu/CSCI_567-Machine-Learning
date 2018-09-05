from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        X = []
        for i in range(len(features)):
            X.append([1] + features[i])
        w = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(numpy.transpose(X),X)),numpy.transpose(X)),values)
        self.w = w

    def predict(self, features: List[List[float]]) -> List[float]:
        X = []
        for i in range(len(features)):
            X.append([1] + features[i])
        values = numpy.matmul(self.w,numpy.transpose(X))
        return values

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        raise self.w


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        X = []
        for i in range(len(features)):
            X.append([1] + features[i])
        w = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(numpy.transpose(X),X) + self.alpha*numpy.identity(self.nb_features+1)),numpy.transpose(X)),values)
        self.w = w

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        X = []
        for i in range(len(features)):
            X.append([1] + features[i])
        values = numpy.matmul(self.w,numpy.transpose(X))
        return values

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        raise self.w


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
