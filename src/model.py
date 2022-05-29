from features import FEATURES

import pickle
import sklearn.ensemble
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier, VotingClassifier

class SVM(SVC):
    pass


class RF(RandomForestClassifier):
    pass


class ANN(MLPClassifier):
    pass


class KNN(KNeighborsClassifier):
    pass


class GenderClassifier(sklearn.ensemble.AdaBoostClassifier):
    def __init__(self,):
        pass

    def predict(self, features: dict) -> int:
        assert all(feature_name in FEATURES
                   for feature_name in features.keys()), "Encountered an unknown feature!"
        return super().predict(features)

    def pickle(self, file_name='classifier.pkl'):
        with open(file_name, 'wb') as clf_file:
            pickle.dump(self, clf_file)
