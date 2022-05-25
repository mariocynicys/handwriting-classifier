from features import FEATURES

import pickle
import numpy as np
import sklearn.ensemble
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class GenderClassifier(sklearn.ensemble.TryDifferentEnsembleClassifiers):
    def __init__(self,):
        pass

    def predict(self, features: dict):
        assert all(feature_name in FEATURES
                   for feature_name in features.keys()), "Encountered an unknown feature!"
        return 1

    def pickle(self, file_name='classifier.pkl'):
        with open(file_name, 'wb') as clf_file:
            pickle.dump(self, clf_file)
