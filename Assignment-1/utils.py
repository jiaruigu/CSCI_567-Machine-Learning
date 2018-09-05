from typing import List

import numpy as np

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    mse = 0
    for i in range(len(y_true)):
        mse += (y_true[i] - y_pred[i])**2
    return mse


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    dbz = 0
    assert len(real_labels) == len(predicted_labels)
    correctly_predicted, real_one_label, predicted_one_label = 0, 0, 0
    for i in range(len(real_labels)):
        if real_labels[i] == predicted_labels[i] == 1:
            correctly_predicted += 1
        if real_labels[i] == 1:
            real_one_label += 1
        if predicted_labels[i] == 1:
            predicted_one_label += 1
    if correctly_predicted == 0:
        return 0
    try:
        precision = correctly_predicted/predicted_one_label
    except:
        dbz += 1
    try:
        recall = correctly_predicted/real_one_label
    except:
        dbz += 2
    if dbz == 0:
        return 2*(precision*recall/(precision+recall))
    elif dbz == 1:
        return 2*recall
    elif dbz == 2:
        return 2*precision
    else:
        return 0


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    assert k > 0
    if k == 1 or len(features) == 0:
        return features
    features_extended=[]
    for i in range(len(features)):
        features_extended.append([])
        for j in range(len(features[i])):
            features_extended[i].append(features[i][j])
    for i in range(len(features)):
        for j in range(1,k):
            for n in range(len(features[0])):
                features_extended[i].append(features[i][n]**(j + 1))
    return features_extended


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.inner(np.subtract(point1, point2), np.subtract(point1, point2))**0.5


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.inner(point1, point2)


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    import math
    a = np.subtract(point1, point2)
    sum = 0
    for i in a:
        sum += i**2
    return -math.exp(-0.5*sum)


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized_features = []
        for i in range(len(features)):
            try:
                normalized_features.append(features[i]/np.inner(features[i],features[i])**0.5)
            except:
                return features
        return normalized_features


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        normalized_features = []
        for i in range(len(features)):
            try:
                normalized_features.append(np.divide(np.subtract(features[i], min(features[i])),max(features[i]) - min(features[i])))
            except e:
                return features
        return normalized_features
